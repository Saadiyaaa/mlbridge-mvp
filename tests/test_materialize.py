"""materialized_inputs 的聚焦测试：只覆盖物化语义本身，不触碰 Drive/Colab。"""
from __future__ import annotations

import errno
import os
import shutil
import stat
import subprocess
import json
from pathlib import Path

import pytest

from mlbridge import core
from mlbridge import materialize as mz


@pytest.fixture
def ws(tmp_path):
    """工作区 root + 外部模型目录。root 只放一个哨兵文件，便于断言未被修改。"""
    root = tmp_path / 'proj'
    root.mkdir()
    (root / 'keep.txt').write_text('SENTINEL')
    external = tmp_path / 'external'
    (external / 'dataset').mkdir(parents=True)
    (external / 'weights.bin').write_bytes(b'W' * 128)
    (external / 'dataset' / 'a.csv').write_text('a')
    (external / 'dataset' / 'nested').mkdir()
    (external / 'dataset' / 'nested' / 'b.csv').write_text('b')
    return root


def _no_links_under(base):
    return [p for p in base.rglob('*') if p.is_symlink()]


def test_local_restic_snapshot_restores_real_dir_over_existing_link(ws, tmp_path):
    if not shutil.which('restic'):
        pytest.skip('restic unavailable')
    (ws / 'models').mkdir()
    (ws / 'models/model').symlink_to('../external/dataset', target_is_directory=True)
    # The link target lives outside the repo, like a local model cache.
    (ws / 'models/model').unlink()
    (ws / 'models/model').symlink_to(tmp_path / 'external/dataset', target_is_directory=True)
    repo = tmp_path / 'restic-repo'
    clone = tmp_path / 'clone'
    (clone / 'models').mkdir(parents=True)
    (clone / 'models/model').symlink_to(tmp_path / 'external/dataset', target_is_directory=True)
    env = {**os.environ, 'RESTIC_PASSWORD': 'test-password-only'}
    subprocess.run(['restic', '-r', str(repo), 'init'], check=True, capture_output=True,
                   env=env)
    with mz.materialized_inputs(ws, ['models/model']) as (stage, rels):
        p = subprocess.run(['restic', '-r', str(repo), 'backup', '--json', *rels],
                           cwd=stage, env=env, check=True, text=True, capture_output=True)
        snap = next(json.loads(line)['snapshot_id'] for line in p.stdout.splitlines()
                    if json.loads(line).get('message_type') == 'summary')
    subprocess.run(['restic', '-r', str(repo), 'restore', snap, '--target', str(clone)],
                   check=True, capture_output=True, env=env)
    assert (clone / 'models/model').is_dir()
    assert not (clone / 'models/model').is_symlink()
    assert (clone / 'models/model/a.csv').read_text() == 'a'
    assert (ws / 'models/model').is_symlink()


def test_file_and_dir_symlink_materialized(ws):
    (ws / 'w.bin').symlink_to('../external/weights.bin')
    (ws / 'data').symlink_to('../external/dataset')
    with mz.materialized_inputs(ws, ['w.bin', 'data']) as (stage, rels):
        assert rels == ['w.bin', 'data']
        # staging 在 root.parent（工作区之外、同文件系统）
        assert stage.parent == ws.parent
        assert not stage.resolve().is_relative_to(ws.resolve())
        assert stage.stat().st_mode & 0o777 == 0o700
        # 逻辑相对路径在 staging 中重现，且全部是真实文件/目录，无软链接
        w = stage / 'w.bin'
        assert w.is_file() and not w.is_symlink()
        assert w.read_bytes() == b'W' * 128
        assert (stage / 'data' / 'a.csv').read_text() == 'a'
        assert (stage / 'data' / 'nested' / 'b.csv').read_text() == 'b'
        assert not _no_links_under(stage)
        # 同一文件系统上应优先硬链接（inode 相同），避免磁盘成本
        src = (ws / 'w.bin').stat()
        assert (w.stat().st_ino, w.stat().st_dev) == (src.st_ino, src.st_dev)
    assert not stage.exists()


def test_validation_without_materialization_rejects_external_symlink(ws):
    # 禁用物化时，越界软链接被拒绝。
    (ws / 'w.bin').symlink_to('../external/weights.bin')
    with pytest.raises(core.BridgeError):
        core.validate_paths(ws, ['w.bin'])


def test_nested_git_metadata_is_not_staged(ws):
    (ws / 'model').symlink_to('../external/dataset')
    hidden = ws.parent / 'external/dataset/.git'
    hidden.mkdir()
    (hidden / 'config').write_text('internal')
    with mz.materialized_inputs(ws, ['model']) as (stage, _):
        assert (stage / 'model/a.csv').is_file()
        assert not (stage / 'model/.git').exists()


def test_regular_in_root_path_also_staged(ws):
    (ws / 'plain').mkdir()
    (ws / 'plain' / 'x.txt').write_text('x')
    with mz.materialized_inputs(ws, ['plain']) as (stage, _):
        assert (stage / 'plain' / 'x.txt').read_text() == 'x'
        assert (stage / 'plain').is_dir() and not (stage / 'plain').is_symlink()


def test_nested_symlink_inside_selected_dir_rejected(ws):
    (ws / 'd').mkdir()
    (ws / 'd' / 'real.txt').write_text('r')
    (ws / 'd' / 'link').symlink_to('../external/weights.bin')
    with pytest.raises(core.BridgeError, match='嵌套软链接'):
        with mz.materialized_inputs(ws, ['d']):
            pass


def test_git_component_rejected(ws):
    (ws / '.git').mkdir()
    (ws / '.git' / 'HEAD').write_text('ref')
    for path in ['.git', '.git/HEAD', 'models/.git']:
        with pytest.raises(core.BridgeError):
            with mz.materialized_inputs(ws, [path]):
                pass


def test_broken_symlink_rejected(ws):
    (ws / 'broken').symlink_to('../external/does-not-exist')
    with pytest.raises(core.BridgeError):
        with mz.materialized_inputs(ws, ['broken']):
            pass


def test_symlink_ancestor_rejected(ws):
    # 显式路径的中间组件是软链接：即使末段真实存在也拒绝。
    (ws / 'alias').symlink_to('../external')
    with pytest.raises(core.BridgeError, match='祖先'):
        with mz.materialized_inputs(ws, ['alias/dataset/a.csv']):
            pass


def test_overlap_and_duplicate_rejected(ws):
    (ws / 'data').mkdir()
    (ws / 'data' / 'sub').mkdir()
    for paths in [['data', 'data/sub'], ['data', 'data'], ['data/sub', 'data/sub']]:
        with pytest.raises(core.BridgeError, match='重叠|重复'):
            with mz.materialized_inputs(ws, paths):
                pass


def test_empty_paths_rejected(ws):
    with pytest.raises(core.BridgeError, match='至少'):
        with mz.materialized_inputs(ws, []):
            pass


def test_traversal_and_absolute_rejected(ws):
    for path in ['../external', '/etc/passwd', 'a/../../b']:
        with pytest.raises(core.BridgeError):
            with mz.materialized_inputs(ws, [path]):
                pass


def test_special_file_rejected(ws, tmp_path):
    os.mkfifo(ws / 'fifo')
    with pytest.raises(core.BridgeError, match='特殊文件'):
        with mz.materialized_inputs(ws, ['fifo']):
            pass
    # 软链接指向 FIFO 也不行
    (ws / 'fifo-link').symlink_to(tmp_path / 'external' / 'weights.bin')
    os.mkfifo(tmp_path / 'other.fifo')
    (ws / 'fifo-link2').symlink_to(tmp_path / 'other.fifo')
    with pytest.raises(core.BridgeError, match='特殊文件'):
        with mz.materialized_inputs(ws, ['fifo-link2']):
            pass


def test_symlink_target_other_special_type_rejected(ws, tmp_path, monkeypatch):
    # 字符设备等在测试里无法安全创建：伪装 dest.stat() 报告 socket 类型，
    # 验证既非普通文件也非目录时 fail closed。
    (ws / 'm').symlink_to('../external/weights.bin')
    real_stat = os.stat

    def fake_stat(path, *, follow_symlinks=True, **kw):
        st = real_stat(path, follow_symlinks=follow_symlinks, **kw)
        if follow_symlinks and Path(path).name == 'm':
            fake = os.stat_result((stat.S_IFSOCK | 0o600,) + tuple(st)[1:])
            return fake
        return st

    monkeypatch.setattr(mz.os, 'stat', fake_stat)
    with pytest.raises(core.BridgeError, match='特殊文件'):
        with mz.materialized_inputs(ws, ['m']):
            pass


def test_exdev_fallback_to_copy2(ws, monkeypatch):
    (ws / 'w.bin').symlink_to('../external/weights.bin')
    calls = []
    real_copy2 = shutil.copy2

    def fake_link(src, dst, **kw):
        calls.append('link')
        raise OSError(errno.EXDEV, 'Invalid cross-device link')

    def fake_copy2(src, dst, **kw):
        calls.append('copy2')
        return real_copy2(src, dst, **kw)

    monkeypatch.setattr(mz.os, 'link', fake_link)
    monkeypatch.setattr(mz.shutil, 'copy2', fake_copy2)
    with mz.materialized_inputs(ws, ['w.bin']) as (stage, _):
        assert calls == ['link', 'copy2']
        assert (stage / 'w.bin').read_bytes() == b'W' * 128
        # 回退路径不是硬链接
        assert (stage / 'w.bin').stat().st_ino != (ws / 'w.bin').stat().st_ino


def test_link_implementation_linking_symlink_instead_of_target_is_corrected(
        ws, monkeypatch):
    (ws / 'w.bin').symlink_to('../external/weights.bin')
    def link_symlink_inode(src, dst, **kwargs):
        Path(dst).symlink_to(os.readlink(src))
    monkeypatch.setattr(mz.os, 'link', link_symlink_inode)
    with mz.materialized_inputs(ws, ['w.bin']) as (stage, _):
        assert (stage / 'w.bin').read_bytes() == b'W' * 128
        assert not (stage / 'w.bin').is_symlink()


def test_eperm_fallback_to_copy2(ws, monkeypatch):
    (ws / 'w.bin').symlink_to('../external/weights.bin')
    monkeypatch.setattr(mz.os, 'link',
                        lambda *a, **k: (_ for _ in ()).throw(
                            OSError(errno.EPERM, 'Operation not permitted')))
    copied = []
    real_copy2 = shutil.copy2
    monkeypatch.setattr(mz.shutil, 'copy2',
                        lambda s, d, **k: (copied.append(1), real_copy2(s, d, **k))[1])
    with mz.materialized_inputs(ws, ['w.bin']) as (stage, _):
        assert copied == [1]
        assert (stage / 'w.bin').read_bytes() == b'W' * 128


def test_unexpected_oserror_becomes_bridge_error(ws, monkeypatch):
    (ws / 'w.bin').symlink_to('../external/weights.bin')
    monkeypatch.setattr(mz.os, 'link',
                        lambda *a, **k: (_ for _ in ()).throw(
                            OSError(errno.EIO, 'I/O error')))
    with pytest.raises(core.BridgeError):
        with mz.materialized_inputs(ws, ['w.bin']):
            pass


def test_cleanup_on_success_and_on_restic_failure(ws, tmp_path):
    (ws / 'w.bin').symlink_to('../external/weights.bin')
    before = sorted(p.name for p in ws.parent.iterdir())
    with mz.materialized_inputs(ws, ['w.bin']) as (stage, _):
        assert stage.exists()
        stage_name = stage.name
    after = sorted(p.name for p in ws.parent.iterdir())
    assert before == after and stage_name not in after

    # 模拟 restic 在 with 体内失败（main 的真实调用形态）：staging 仍必须清理。
    (ws / 'w2.bin').symlink_to('../external/weights.bin')
    with pytest.raises(core.BridgeError, match='restic'):
        with mz.materialized_inputs(ws, ['w2.bin']) as (stage2, _):
            raise core.BridgeError('restic backup failed')
    assert not stage2.exists()
    assert sorted(p.name for p in ws.parent.iterdir()) == after


def test_sources_and_workspace_unchanged(ws):
    (ws / 'w.bin').symlink_to('../external/weights.bin')
    (ws / 'data').symlink_to('../external/dataset')
    target = ws.parent / 'external' / 'weights.bin'
    target_bytes = target.read_bytes()
    root_tree = sorted(str(p.relative_to(ws)) + ('/' if p.is_dir() else '')
                       for p in ws.rglob('*'))
    links = {str(p.relative_to(ws)): os.readlink(p) for p in ws.rglob('*')
             if p.is_symlink()}
    target_stat = target.stat()

    with mz.materialized_inputs(ws, ['w.bin', 'data']):
        pass

    # 工作区目录树完全一致（不新增任何文件/目录）
    assert sorted(str(p.relative_to(ws)) + ('/' if p.is_dir() else '')
                  for p in ws.rglob('*')) == root_tree
    assert (ws / 'keep.txt').read_text() == 'SENTINEL'
    # 软链接仍是软链接，指向不变
    assert {str(p.relative_to(ws)): os.readlink(p)
            for p in ws.rglob('*') if p.is_symlink()} == links
    # 外部目标内容与 mtime/nsize 未被触碰
    assert target.read_bytes() == target_bytes
    after = target.stat()
    assert (after.st_mtime_ns, after.st_size) == (target_stat.st_mtime_ns,
                                                  target_stat.st_size)
