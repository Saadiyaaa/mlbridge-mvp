"""在工作区外临时物化选中的软链接目标，供 restic 一次备份。

restic 默认备份软链接本身；物化后备份其指向的文件内容。
在 root.parent 创建 0700 的暂存目录，按项目相对路径放置真实文件；
普通路径与软链接目标一起暂存，项目工作区和原软链接保持不变。
拒绝软链接祖先、断链、嵌套软链接、.git、特殊文件、越界路径及重叠选择。
优先硬链接；不支持时复制，可能占用与源文件等量的临时空间。
源文件在检查与复制之间发生变动时，备份内容可能不一致。
"""
from __future__ import annotations

import contextlib
import errno
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path

from .core import BridgeError, checked_relative

# 允许硬链接失败后回退复制的 errno：跨文件系统 / 权限 / 文件系统不支持。
_HARDLINK_FALLBACK = frozenset({errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP})


def _lstat(path: Path, what: str) -> os.stat_result:
    try:
        return path.lstat()
    except FileNotFoundError as exc:
        raise BridgeError(f"{what}不存在或为断链: {path}") from exc
    except OSError as exc:
        raise BridgeError(f"{what}不可访问: {path} ({exc})") from exc


def _link_file(src: Path, dst: Path) -> None:
    """硬链接普通文件；跨设备/不支持时回退 copy2（注意磁盘成本）。"""
    try:
        os.link(src, dst)
        # Link traversal differs across platforms; restic must never receive
        # the symlink inode instead of the selected model bytes.
        if dst.is_symlink():
            dst.unlink()
            shutil.copy2(src, dst)
        if not dst.is_file() or dst.is_symlink():
            raise BridgeError(f"暂存文件不是普通文件: {dst}")
    except OSError as exc:
        if exc.errno not in _HARDLINK_FALLBACK:
            raise BridgeError(f"无法实体化文件 {src} -> {dst} ({exc})") from exc
        try:
            shutil.copy2(src, dst)
            if not dst.is_file() or dst.is_symlink():
                raise BridgeError(f"暂存文件不是普通文件: {dst}")
        except OSError as exc2:
            raise BridgeError(f"复制文件失败 {src} -> {dst} ({exc2})") from exc2


def _materialize_tree(src_dir: Path, dst_dir: Path) -> None:
    """把真实目录树实体化到 staging；目录建成真实目录，不跟随任何链接。"""
    dst_dir.mkdir()
    try:
        entries = list(os.scandir(src_dir))
    except OSError as exc:
        raise BridgeError(f"无法读取目录 {src_dir} ({exc})") from exc
    for entry in entries:
        if entry.name == '.git':
            continue  # Git 元数据不属于模型/数据输入；绝不注入到 Colab 克隆仓库
        child = Path(entry.path)
        if entry.is_symlink():
            raise BridgeError(f"选中目录内存在嵌套软链接，拒绝跟随: {child}")
        try:
            st = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise BridgeError(f"无法读取条目 {child} ({exc})") from exc
        if stat.S_ISREG(st.st_mode):
            _link_file(child, dst_dir / entry.name)
        elif stat.S_ISDIR(st.st_mode):
            _materialize_tree(child, dst_dir / entry.name)
        else:
            raise BridgeError(f"选中目录内存在特殊文件(FIFO/socket/设备等): {child}")


def _materialize_one(stage_root: Path, root: Path, rel: str) -> None:
    dest = root / rel
    target = stage_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)

    # 1) 显式路径的祖先组件必须是真实目录，绝不允许经过软链接。
    parts = Path(rel).parts
    for i in range(1, len(parts)):
        ancestor = root / Path(*parts[:i])
        st = _lstat(ancestor, f"路径祖先 {Path(*parts[:i])}")
        if stat.S_ISLNK(st.st_mode):
            raise BridgeError(f"路径经过软链接祖先，拒绝: {ancestor}")
        if not stat.S_ISDIR(st.st_mode):
            raise BridgeError(f"路径祖先不是目录: {ancestor}")

    # 2) 末段组件：允许软链接（仅末段），也允许普通文件/目录。
    st = _lstat(dest, f"路径 {rel}")
    if stat.S_ISLNK(st.st_mode):
        try:
            resolved_path = dest.resolve(strict=True)
            if '.git' in resolved_path.parts:
                raise BridgeError(f'软链接目标属于 .git，拒绝发布: {rel}')
            resolved = dest.stat()
        except (OSError, RuntimeError) as exc:
            raise BridgeError(f"软链接目标不可读（断链？）: {dest} ({exc})") from exc
        if stat.S_ISREG(resolved.st_mode):
            _link_file(dest, target)
        elif stat.S_ISDIR(resolved.st_mode):
            _materialize_tree(dest, target)
        else:
            raise BridgeError(f"软链接目标是特殊文件(FIFO/socket/设备等): {dest}")
    elif stat.S_ISREG(st.st_mode):
        _link_file(dest, target)
    elif stat.S_ISDIR(st.st_mode):
        _materialize_tree(dest, target)
    else:
        raise BridgeError(f"路径是特殊文件(FIFO/socket/设备等): {dest}")


def _check_overlap(rels: list[str]) -> list[str]:
    out: list[str] = []
    seen: list[tuple[str, ...]] = []
    for rel in rels:
        parts = Path(rel).parts
        if parts in seen:
            raise BridgeError(f"重复选中同一路径: {rel}")
        for prev in seen:
            if prev == parts[:len(prev)] or parts == prev[:len(parts)]:
                raise BridgeError(f"选中路径互相重叠: {rel}")
        seen.append(parts)
        out.append(rel)
    return out


@contextlib.contextmanager
def materialized_inputs(root: Path, paths: list[str]) -> Iterator[tuple[Path, list[str]]]:
    """把选中的输入（末段软链接或普通路径）实体化到工作区外的临时目录。

    用法（main 在 --materialize-links 时调用）::

        with materialized_inputs(root, paths) as (stage, rels):
            snapshot = restic_backup(cwd=stage, paths=rels, ...)

    yield 的 (stage, rels)：staging 目录与**原样保留的逻辑相对路径**，调用方
    只需把 cwd 从 root 换成 stage，argv 路径与 tag 完全不变。退出（无论正常
    还是异常，包括 restic 失败）时一定删除 staging 目录。
    """
    if not root.is_dir():
        raise BridgeError(f"项目根目录不存在: {root}")
    if not paths:
        raise BridgeError(
            "--materialize-links 至少要选择一个路径；没有链接需要物化时请直接用普通 publish")
    root = root.resolve()
    rels = _check_overlap([checked_relative(p) for p in paths])
    # checked_relative 之外的 OSError（极端文件系统错误）也要收敛为 BridgeError。
    try:
        stage = Path(tempfile.mkdtemp(prefix=".mlbridge-materialize-",
                                      dir=str(root.parent)))
    except OSError as exc:
        raise BridgeError(f"无法在 {root.parent} 创建临时物化目录 ({exc})") from exc
    try:
        # 防御：staging 必须在 root 之外（绝不等于 root 或位于 root 内部），
        # 例如 root == "/" 时 root.parent 就是 root 自身，必须拒绝。
        resolved_stage = stage.resolve()
        if resolved_stage == root or resolved_stage.is_relative_to(root):
            raise BridgeError(f"临时物化目录落在项目根目录内: {resolved_stage}")
        resolved_stage.chmod(0o700)
        for rel in rels:
            _materialize_one(stage, root, rel)
        yield stage, rels
    except BridgeError:
        raise
    except OSError as exc:
        raise BridgeError(f"物化输入失败: {exc}") from exc
    finally:
        # 即使 restic 失败/任何异常也要清理；删除失败不掩盖原始错误。
        with contextlib.suppress(OSError):
            shutil.rmtree(stage)
