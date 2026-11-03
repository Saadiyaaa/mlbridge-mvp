"""foreground 执行模式（plan.execution_mode v1）与执行中断后的诚实恢复。

全部离线（fake colab CLI + fake restic + 本地 git clone）：
- starter/exec/超时语义：detached 文本逐字节不变；foreground 无
  start_new_session、不 import mlbridge；本地 exec 显式长超时、超时仅
  观测中断（绝不改判远端失败）；
- 真实子进程路径（fake exec 把 starter 脚本作为真实本地子进程运行，
  站在 /content 位置，只做目录注入）：确认状态/回执在 exec 结束前可见
  （哨兵之前）、worker 输出全部进日志文件、batch_worker/remote_worker
  保持 standalone（不 import mlbridge）；
- 连接断开/超时后 worker 若仍在运行：后续 watch 仍按远端事实对账提交，
  绝不 abandon；
- foreground + 中断后的恢复流程：单 unit 中断 → 观测 last_known → 手动
  stop → 中断对账写 execution_abandoned → --retry-failed 显式重跑 →
  新 attempt 成功提交（旧 abandoned attempt 保留、不改写）。

未宣称：真实 Colab 环境验证；任一模式不承诺平台不回收、不承诺成功。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

import mlbridge.protocol as P
from mlbridge import colab_runner as CR
from mlbridge import core
from mlbridge.cli import execute as cli_execute
from mlbridge.cli import parser
from mlbridge.ledger import Ledger
from mlbridge import remote_worker as RW

REPO = None  # 由 make_repo 注入真实本地仓库路径（git clone 可用）


def make_repo(tmp_path):
    """本地 git 仓库：真实 clone/checkout（无网络）。"""
    repo = tmp_path / 'git-src'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.name', 'T'], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.email', 't@e.c'],
                   check=True)
    (repo / 'readme.md').write_text('demo workspace\n')
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), 'commit', '-qm', 'init'], check=True)
    commit = subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', '-C', str(repo), 'remote', 'add', 'origin', str(repo)],
                   check=True)
    subprocess.run(['git', '-C', str(repo), 'update-ref',
                    'refs/remotes/origin/master', commit], check=True)
    subprocess.run(['git', '-C', str(repo), 'branch', '--set-upstream-to',
                    'origin/master'], check=True, capture_output=True)
    return repo, commit


def _imports_mlbridge(source: str) -> bool:
    """是否出现真实的 `import mlbridge` / `from mlbridge...` 语句（仅看代码，
    忽略 docstring 与注释中提及）。"""
    try:
        import ast
    except ImportError:  # pragma: no cover
        return False
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(a.name.split('.')[0] == 'mlbridge' for a in node.names):
                return True
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.split('.')[0] == 'mlbridge':
                return True
    return False


FAKE_RESTIC = r'''#!/bin/sh
sub=""
for a in "$@"; do
  case "$a" in backup|restore|version|snapshots) sub="$a"; break;; esac
done
case "$sub" in
  version) echo "restic 0.17.0 compiled with go1.22.5 on linux/amd64"; exit 0;;
  restore) exit 0;;
  backup)
    n=$(cat "${FAKE_RESTIC_COUNT:?required}" 2>/dev/null || echo 0); n=$((n+1))
    printf '%s' "$n" > "${FAKE_RESTIC_COUNT:?required}"
    printf '{"message_type":"summary","snapshot_id":"b%063d"}\n' "$n"
    exit 0;;
esac
exit 0
'''


def wait_for(pred, deadline_s=30.0):
    start = time.monotonic()
    while time.monotonic() - start < deadline_s:
        if pred():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------------------
# 基础设施：XDG 隔离 + 会产真实子进程的 fake colab CLI
# ---------------------------------------------------------------------------

class FakeVM:
    """模拟 Colab VM 的 staged 文件与代际（stop 后视为已释放）。"""

    def __init__(self, prefix: Path):
        self.prefix = str(prefix)
        self.gen = 0
        self.staged: dict[str, Path] = {}
        self.jobs: list[dict] = []
        self.exec_results: list = []

    @property
    def root(self) -> Path:
        return Path(f'{self.prefix}-gen{self.gen}')

    def upload(self, src: Path, name: str) -> None:
        self.staged[name] = src
        if name == 'mlbridge_job.json':
            self.jobs.append(json.loads(src.read_text()))

    def materialize(self) -> Path:
        """每次 exec 前重建该代的 /content（同代内新 exec = 新 VM 会话）。"""
        root = self.root
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        for name, src in self.staged.items():
            data = src.read_text()
            if name == 'mlbridge_worker.py':
                # 测试镜像远端 patch：/content → 本测试目录（与真实上传
                # 无关；对上传集合/内容契约只做目录绑定替换）。
                # 远端 worker 的 BASE 以环境变量覆盖（缺省 /content）；
                # 这里把整行重绑到本测试目录（离线镜像，不改变上传契约）。
                data, replaced = re.subn(
                    r"BASE = Path\(os\.environ\.get\('MLBRIDGE_WORKER_BASE'\)"
                    r" or '/content'\)",
                    f"BASE = Path({str(root)!r})", data)
                assert replaced == 1, 'staged worker 的 BASE 定义形状变了' 
                assert str(root) in data
                assert _imports_mlbridge(data) is False
            if name == 'batch_worker.py':
                assert _imports_mlbridge(data) is False
            (root / name).write_text(data)
        (root / 'mlbridge_rclone.conf').chmod(0o600)
        (root / 'mlbridge_restic_password').chmod(0o600)
        return root


@pytest.fixture
def fg_env(tmp_path, monkeypatch):
    """独立 XDG 配置 + 已 push 的本地 git 工作区 + fake colab CLI。"""
    repo, commit = make_repo(tmp_path)
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    monkeypatch.delenv('MLBRIDGE_GITHUB_USERNAME', raising=False)
    monkeypatch.delenv('MLBRIDGE_GITHUB_TOKEN', raising=False)
    monkeypatch.setattr(CR, '_github_secrets_path',
                        lambda: tmp_path / 'isolated-secrets.env')
    monkeypatch.setattr(CR, 'list_active_sessions', lambda: [])
    monkeypatch.setattr(CR.time, 'sleep', lambda s: None)
    cfg = tmp_path / 'cfg' / 'mlbridge'
    cfg.mkdir(parents=True)
    scratch = tmp_path / 'creds'
    scratch.mkdir()
    rconf = scratch / 'rclone.conf'
    passwd = scratch / 'passwd'
    rconf.write_text('[gdrive]\ntype=drive\n')
    passwd.write_text('topsecret')
    (cfg / 'settings.toml').write_text(
        '[storage]\nremote="gdrive"\nprefix="data"\n'
        f'password_file="{passwd}"\nrclone_config="{rconf}"\n\n'
        '[colab]\ndefault_gpu="CPU"\n')
    core.register('demo', str(repo), str(repo))
    setup = [[sys.executable, '-c', "import pathlib\n"
             "pathlib.Path('SETUP.log').open('a').write('x')"]]
    (core.project_file('demo')).write_text(
        'name = "demo"\n'
        f'workspace = {json.dumps(str(repo))}\n'
        f'repository = {json.dumps(str(repo))}\n'
        '\n[jobs.example]\n'
        'command = ["python", "train.py"]\n'
        f'setup = {json.dumps(setup)}\n'
        'outputs = ["outputs"]\n')
    # fake restic / fake rclone 进 worker PATH。
    fakebin = tmp_path / 'fakebin'
    fakebin.mkdir()
    count = tmp_path / 'restic-count.txt'
    count.write_text('0')
    restic_log = tmp_path / 'restic.log'
    restic_log.write_text('')
    fake_restic = fakebin / 'restic'
    fake_restic.write_text(FAKE_RESTIC)
    fake_restic.chmod(0o755)
    fake_rclone = fakebin / 'rclone'
    fake_rclone.write_text('#!/bin/sh\nexit 0\n')
    fake_rclone.chmod(0o755)
    monkeypatch.setenv('PATH', str(fakebin) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('FAKE_RESTIC_LOG', str(restic_log))
    monkeypatch.setenv('FAKE_RESTIC_COUNT', str(count))

    vm = FakeVM(tmp_path / 'vm')
    state: dict = {'calls': [], 'exec_calls': [], 'exec_stdout': []}

    def cli(*args, capture=False, timeout=None):
        state['calls'].append(args)
        verb = args[0]
        if verb == 'new':
            return subprocess.CompletedProcess(args, 0)
        if verb == 'exec':
            script = Path(args[4]).read_text().replace('/content',
                                                       str(vm.root))
            root = vm.materialize()
            hook = state.get('exec_hook')
            if hook is not None:
                result = hook(args, script, root, timeout)
                state['exec_stdout'].append(
                    getattr(result, 'stdout', None))
                return result
            r = subprocess.run([sys.executable, '-c', script], cwd=root,
                               env=dict(**os.environ),
                               capture_output=True, text=True,
                               timeout=timeout)
            state['exec_stdout'].append(r.stdout)
            return subprocess.CompletedProcess(args, r.returncode,
                                               stdout=r.stdout,
                                               stderr=r.stderr)
        if verb == 'download':
            session, remote_path, dest = args[2], args[3], Path(args[4])
            if state.get('stopped') or vm.gen > 0 and vm.root.exists() is False:
                raise core.BridgeError(f'download failed: {remote_path}')
            rel = remote_path.removeprefix('/content/')
            target = vm.root / rel
            if not target.is_file():
                raise core.BridgeError(f'download failed: {remote_path}')
            dest.write_text(target.read_text(encoding='utf-8'))
            return subprocess.CompletedProcess(args, 0)
        if verb == 'stop':
            return subprocess.CompletedProcess(args, 0)
        return subprocess.CompletedProcess(args, 0)

    def stage(session, src, name):
        state['staged'] = state.get('staged', [])
        state['staged'].append(name)
        if name in ('mlbridge_rclone.conf', 'mlbridge_restic_password',
                    'mlbridge_git_auth'):
            # 凭据同样进入 staged（真实 exec 时以 0600 落盘）。
            pass
        vm.upload(src, name)

    def put_script(session, contents):
        state['put_scripts'] = state.get('put_scripts', [])
        state['put_scripts'].append(contents)

    monkeypatch.setattr(CR, 'cli', cli)
    monkeypatch.setattr(CR, 'stage', stage)
    monkeypatch.setattr(CR, 'put_script', put_script)
    state.update(vm=vm, tmp_path=tmp_path)
    return state


def _spec(command, outputs, setup, repo, commit, validator_timeout=30.0,
          **spec_extra):
    payload = {
        'inputs': {'data': 'a' * 64},
        'command': command,
        'params': {'lr': 1e-3},
        'outputs': outputs,
        'validator': {'kind': 'command', 'command': [sys.executable, '-c',
                                                     'pass'],
                      'timeout_seconds': validator_timeout},
    }
    payload.update(spec_extra)
    unit = P.WorkUnit.from_dict({**payload, 'schema_version': 2,
                                 'project': 'demo',
                                 'repository': str(repo), 'commit': commit,
                                 'environment': {'python': '3.12',
                                  'setup_sha256': P.canonical_digest(setup)}})
    return unit.to_dict()


def _batch_file(units, path, timeout=20.0):
    payload = {
        'schema_version': 2,
        'project': 'demo',
        'environment': {'python': '3.12'},
        'units': [dict(item, execution={'timeout_seconds': timeout})
                  for item in units],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


# ---------------------------------------------------------------------------
# 单元级：starter/超时/远端校验
# ---------------------------------------------------------------------------

def test_detached_starter_text_unchanged():
    script = CR.starter_script()
    assert 'start_new_session=True' in script and '--rm' not in script
    assert 'import mlbridge' not in script


def test_foreground_starter_is_text_free_and_synchronous():
    script = CR.starter_script('foreground')
    assert 'start_new_session' not in script
    assert 'import mlbridge' not in script
    assert CR.FOREGROUND_DONE in script
    # worker 输出只落日志文件；cell 只打印哨兵（输出有界）。
    assert 'stdout=f' in script and 'stderr=f' in script
    with pytest.raises(core.BridgeError):
        CR.starter_script('bogus')


def test_foreground_exec_timeout_is_explicit_and_bounded():
    repo = 'https://github.com/demo/demo.git'  # 不真正访问
    content = {
        'batch_id': 'bt1', 'schema_version': 1,
        'publish_timeout_seconds': 300,
        'units': [{'unit_id': 'x' * 64,
                   'execution': {'timeout_seconds': 120},
                   'spec': {'validator': {'kind': 'command',
                                          'command': ['python', '-c', 'pass'],
                                          'timeout_seconds': 40}}}],
    }
    total = CR.foreground_exec_timeout(content)
    # 120 + 40 + manifest 默认 600 + publish 300 + 余量。
    assert total == pytest.approx(
        120 + 40 + 600 + 300 + CR.FOREGROUND_EXEC_MARGIN)
    with_limits = CR.foreground_exec_timeout({
        **content,
        'manifest_limits': {'max_files': 10, 'max_bytes': 1 << 20,
                            'timeout_seconds': 5},
    })
    assert with_limits == pytest.approx(
        120 + 40 + 5 + 300 + CR.FOREGROUND_EXEC_MARGIN)
    with pytest.raises(core.BridgeError):
        CR.foreground_exec_timeout({'units': [{'execution': {}}]})


def test_remote_worker_rejects_bad_and_legacy_foreground_execution_mode():
    assert RW.validate_execution_mode({}) == 'detached'
    assert RW.validate_execution_mode({'execution_mode': 'detached'}) \
        == 'detached'
    assert RW.validate_execution_mode({'execution_mode': 'foreground',
                                       'unit_batch': {'units': []}}) \
        == 'foreground'
    with pytest.raises(RuntimeError, match='仅.*批模式|unit_batch'):
        RW.validate_execution_mode({'execution_mode': 'foreground'})
    with pytest.raises(RuntimeError, match='execution_mode'):
        RW.validate_execution_mode({'execution_mode': 'evil'})
    # 非法 plan-execution_mode 必须在任何副作用之前被拒绝：main 内先于
    # 心跳（这里只验纯函数；main 集成由 harness 用合法值覆盖）。


# ---------------------------------------------------------------------------
# 集成 A：foreground 全程成功（真实子进程）
# ---------------------------------------------------------------------------

def test_foreground_run_success_real_subprocess(fg_env):
    state = fg_env
    ws = core.project('demo')['workspace']
    repo_path = Path(ws)
    commit = subprocess.check_output(['git', '-C', str(repo_path),
                                      'rev-parse', 'HEAD'],
                                     text=True).strip()
    setup = [[sys.executable, '-c',
              "import pathlib\npathlib.Path('SETUP.log').open('a').write('x')"]]
    unit1_cmd = [sys.executable, '-c',
                 "import pathlib\np=pathlib.Path('out1'); p.mkdir(parents=True)\n"
                 "(p/'a.txt').write_text('ONE')"]
    # unit2 等 GO 文件再写输出，测试守护线程先确认 unit1 回执可见。
    unit2_cmd = [sys.executable, '-c',
                 "import os,time,pathlib\n"
                 "while not os.path.exists('GO'):\n    time.sleep(0.05)\n"
                 "p=pathlib.Path('out2'); p.mkdir(parents=True)\n"
                 "(p/'b.txt').write_text('TWO')"]

    spec1 = _spec(unit1_cmd, ['out1'], setup, repo_path, commit)
    # v2 无独立 seed 字段：两个 unit 由 command/outputs 自然区分
    spec2 = _spec(unit2_cmd, ['out2'], setup, repo_path, commit)
    path = _batch_file([{'spec': spec1}, {'spec': spec2}],
                       state['tmp_path'] / 'fg.json')

    observed: list[str] = []
    done = threading.Event()
    receipts = None
    errors = []
    holder = {'receipts': None}

    def probe():
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not done.is_set():
            try:
                st = json.loads(
                    (Path(state['vm'].root) / 'mlbridge_state.json')
                    .read_text())
            except (OSError, ValueError):
                time.sleep(0.02)
                continue
            observed.append(st.get('phase', '?'))
            receipt_dir = Path(state['vm'].root) / 'mlbridge_receipts'
            try:
                if receipt_dir.is_dir() and any(receipt_dir.rglob('*.json')):
                    holder['receipts'] = True
                    # unit1 已有回执：GO 允许 unit2 继续（仍在 exec 内）。
                    (Path(state['vm'].root) / 'mlbridge-workspace'
                     / 'GO').write_text('go')
                    break
            except OSError as exc:
                errors.append(exc)
            time.sleep(0.02)
    t = threading.Thread(target=probe, daemon=True)
    t.start()

    code = cli_execute(parser().parse_args(
        ['run', 'demo', '--unit-batch', str(path),
         '--execution-mode', 'foreground']))
    done.set()
    t.join(timeout=10)
    assert not errors and t.is_alive() is False, errors
    assert code == 0
    # 状态/回执在 exec 结束前可见（哨兵之前）。
    assert holder['receipts'] is True
    assert observed and ('RUNNING' in observed or 'PUBLISHING' in observed)
    # exec 输出含哨兵且 worker 输出未进 cell（有界）。
    out = state['exec_stdout'][-1] or ''
    assert CR.FOREGROUND_DONE in out and 'worker_exit=0' in out
    # cell 输出有界（两行）；研究输出只进日志文件，不进 cell stdout。
    assert len(out.splitlines()) <= 2
    assert 'b.txt' not in out and 'a.txt' not in out
    # 载荷/记录携带 execution_mode；远端 standalone 文件未 import mlbridge。
    payload = state['vm'].jobs[-1]
    assert payload['execution_mode'] == 'foreground'
    session = payload['session']
    record = CR.get_local(session)
    assert record['execution_mode'] == 'foreground'
    assert record['foreground_exit']['worker_exit'] == 0
    # 全部落到台账；auto stop 发生。
    ledger = Ledger('demo')
    for raw in (spec1, spec2):
        uid = P.unit_id(raw)
        view = ledger.get(uid)
        assert view['state'] == 'completed'
        assert view['attempts'][f'att-{uid[:12]}-1']['resolution'][
            'outcome'] == 'task_succeeded'
    assert any(c[0] == 'stop' for c in state['calls'])
    # vm 上留有回执与终态（供本地 watch 独立读取）。
    assert (Path(state['vm'].root) / 'mlbridge_state.json').is_file()


# ---------------------------------------------------------------------------
# 集成 B：本地 exec 超时 = 观测中断；worker 不受影响，后续 watch 仍提交
# ---------------------------------------------------------------------------

def _spawn_fg(script: str, root: Path) -> subprocess.Popen:
    """模拟独立 VM 环境：exec 子进程自建会话（测试只在自杀组内动作）。"""
    outfile = tempfile.NamedTemporaryFile('w+', delete=False)
    proc = subprocess.Popen([sys.executable, '-c', script], cwd=root,
                            env=dict(**os.environ), start_new_session=True,
                            stdout=outfile, stderr=subprocess.STDOUT)
    return proc, outfile


def test_foreground_exec_timeout_is_observation_lost_not_failure(fg_env):
    state = fg_env
    ws = core.project('demo')['workspace']
    commit = subprocess.check_output(['git', '-C', str(ws), 'rev-parse',
                                      'HEAD'], text=True).strip()
    setup = [[sys.executable, '-c',
              "import pathlib\npathlib.Path('SETUP.log').open('a').write('x')"]]
    slow_cmd = [sys.executable, '-c',
                "import time,pathlib\n"
                "p=pathlib.Path('out'); p.mkdir(parents=True)\n"
                "time.sleep(0.8)\n(p/'a.txt').write_text('LATE')"]
    spec = _spec(slow_cmd, ['out'], setup, Path(ws), commit)
    path = _batch_file([{'spec': spec}], state['tmp_path'] / 'fg.json')

    def slow_exec(args, script, root, timeout):
        proc, outfile = _spawn_fg(script, root)
        time.sleep(0.15)          # 本地连接在远端执行中途“断开”
        raise core.ProcessTimeout('命令超时: colab')

    state['exec_hook'] = slow_exec
    code = cli_execute(parser().parse_args(
        ['run', 'demo', '--unit-batch', str(path),
         '--execution-mode', 'foreground', '--detach']))
    assert code == 0
    session = state['vm'].jobs[-1]['session']
    record = CR.get_local(session)
    # 仅观测中断（绝不判远端失败/不 abandoned/不 stop）。
    assert record.get('observation_lost', {}).get('stage') == 'foreground-exec'
    assert record.get('stopped') is not True
    assert any(c[0] == 'stop' for c in state['calls']) is False
    # 远端 worker 不受影响，继续跑到终态。
    assert wait_for(lambda:
                    (Path(state['vm'].root) / 'mlbridge_state.json').is_file()
                    and json.loads(
                        (Path(state['vm'].root) / 'mlbridge_state.json')
                        .read_text()).get('phase') == 'SUCCEEDED',
                    deadline_s=30)
    # 之后 watch 按远端事实正常对账提交（没有 ABANDONED）。
    state['exec_hook'] = None
    code = cli_execute(parser().parse_args(['watch', session]))
    assert code == 0
    view = Ledger('demo').get(P.unit_id(spec))
    attempt = f"att-{P.unit_id(spec)[:12]}-1"
    assert view['state'] == 'completed'
    res = view['attempts'][attempt]['resolution']
    assert res['outcome'] == 'task_succeeded'
    assert not any(
        a['resolution']['outcome'] == 'execution_abandoned'
        for a in view['attempts'].values())


# ---------------------------------------------------------------------------
# 集成 C：foreground + 中断后的恢复流程
# ---------------------------------------------------------------------------

def test_foreground_interrupt_then_honest_abandon_and_retry(fg_env,
                                                            monkeypatch):
    state = fg_env
    from mlbridge import batch_control as BC
    from mlbridge import colab_runner as CR2
    ws = core.project('demo')['workspace']
    commit = subprocess.check_output(['git', '-C', str(ws), 'rev-parse',
                                      'HEAD'], text=True).strip()
    setup = [[sys.executable, '-c',
              "import pathlib\npathlib.Path('SETUP.log').open('a').write('x')"]]
    # bounded wait：attempt1 在等待 GO 期间被中断；重跑时 GO 缺失，
    # bounded wait 结束后正常写输出并退出（同一 spec、同一 batch 身份）。
    hang_cmd = [sys.executable, '-c',
                "import time,pathlib\n"
                "p=pathlib.Path('out'); p.mkdir(parents=True)\n"
                "(p/'started').write_text('x')\n"
                "for _ in range(80):\n"
                "    if pathlib.Path('GO').exists():\n        break\n"
                "    time.sleep(0.05)\n"
                "(p/'a.txt').write_text('RETRIED')\n"]
    spec = _spec(hang_cmd, ['out'], setup, Path(ws), commit)
    path = _batch_file([{'spec': spec}], state['tmp_path'] / 'fg.json')

    def killer(args, script, root, timeout):
        (root.parent / 'killer-ran').write_text('start')
        proc, outfile = _spawn_fg(script, root)
        deadline = time.monotonic() + 60
        started = False
        while time.monotonic() < deadline:
            try:
                st = json.loads((root / 'mlbridge_state.json').read_text())
            except (OSError, ValueError):
                time.sleep(0.02)
                continue
            if st.get('batch', {}).get('current_unit'):
                started = True
                break
            time.sleep(0.02)
        assert started, 'unit 未启动'
        time.sleep(0.2)
        (root.parent / 'killer-ran').write_text('after-kill')
        # cell 进程终止（中断可能随 cell 杀死 worker——foreground 代价）。
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        outfile.close()
        raise core.ProcessTimeout('命令超时: colab')
    state['exec_hook'] = killer
    code = cli_execute(parser().parse_args(
        ['run', 'demo', '--unit-batch', str(path),
         '--execution-mode', 'foreground', '--detach']))
    assert code == 0
    session = state['vm'].jobs[-1]['session']
    uid = P.unit_id(spec)
    att1 = f'att-{uid[:12]}-1'
    record = CR.get_local(session)
    assert record.get('observation_lost', {}).get('stage') == 'foreground-exec'
    assert record.get('stopped') is not True
    ledger = Ledger('demo')
    assert ledger.get(uid)['attempts'][att1]['resolution'] is None

    # 有 last_known 观测、无任何执行终局证据：watch 只报告未知，
    # 台账零改写（观测失联绝不放弃）。
    box = {'download_ok': 1, 'divert': True}
    fixture_cli = CR2.cli

    def divert_cli(*args, capture=False, timeout=None):
        if box['divert'] and args[0] == 'download':
            if box['download_ok'] <= 0:
                raise core.BridgeError('download failed: link lost')
            box['download_ok'] -= 1
        return fixture_cli(*args, capture=capture, timeout=timeout)
    monkeypatch.setattr(CR2, 'cli', divert_cli)
    code = cli_execute(parser().parse_args(['watch', session,
                                            '--interval', '5']))
    assert code == 1
    assert box['download_ok'] == 0
    assert ledger.get(uid)['attempts'][att1]['resolution'] is None

    # 用户确认远端已死 → 手动 stop（VM 释放 = 正面执行终局证据）。
    code = cli_execute(parser().parse_args(['stop', session]))
    assert code == 0
    assert CR.get_local(session).get('stopped') is True

    # 中断对账：RUNNING（本批身份匹配、last_known 兜底）+ tags 无快照
    # → execution_abandoned。
    monkeypatch.setattr(BC, 'restic_attempt_snapshots',
                        lambda ns, bid, u, a: [])
    box['download_ok'] = 0      # VM 已释放：任何 download 都不可达
    state['calls'] = []
    code = cli_execute(parser().parse_args(['watch', session]))
    print('watch2 returns', code, flush=True)
    assert code == 1
    view = ledger.get(uid)
    res1 = view['attempts'][att1]['resolution']
    assert res1['outcome'] == 'execution_abandoned'
    assert res1['terminal_state'] == 'ABANDONED'
    assert res1['source'] == 'controller/stop-confirmed'
    assert '--retry-failed' in res1['note']
    # 对账不再触发任何 stop（不会重复释放/误停其他实例）。
    assert not any(c[0] == 'stop' for c in state['calls'])

    # 无 --retry-failed 阻止重跑（旧 attempt 已确定终止）。
    with pytest.raises(core.BridgeError, match='--retry-failed'):
        cli_execute(parser().parse_args(
            ['run', 'demo', '--unit-batch', str(path)]))

    # 显式重跑（fg exec 正常完成）：新 attempt seq=2 成功提交。
    box['divert'] = False
    state['exec_hook'] = None
    code = cli_execute(parser().parse_args(
        ['run', 'demo', '--unit-batch', str(path),
         '--execution-mode', 'foreground', '--retry-failed']))
    assert code == 0
    view = ledger.get(uid)
    att2 = f'att-{uid[:12]}-2'
    assert att2 in view['attempts'] and view['attempts'][att2]['seq'] == 2
    assert view['state'] == 'completed'
    assert view['result'] is not None
    assert view['result']['attempt_id'] == att2
    res2 = view['attempts'][att2]['resolution']
    assert res2['outcome'] == 'task_succeeded'
    # 旧 abandoned attempt 不被改写、未被删除。
    assert view['attempts'][att1]['resolution']['outcome'] == \
        'execution_abandoned'



def test_legacy_run_rejects_execution_mode():
    """--execution-mode 仅 --unit-batch 批模式有效；legacy 一律拒绝。"""
    with pytest.raises(core.BridgeError, match='--execution-mode|unit-batch'):
        cli_execute(parser().parse_args(
            ['run', 'demo', '--execution-mode', 'foreground']))


def test_run_batch_rejects_unknown_execution_mode(tmp_path):
    from mlbridge import batch_control as BC
    with pytest.raises(core.BridgeError, match='execution-mode'):
        BC.run_batch(project_name='demo', batch_file=tmp_path / 'no.json',
                     commit='a' * 40, repository='r', setup=[],
                     restic_namespace='n', retry_failed=False,
                     on_launch=lambda *a, **k: None,
                     execution_mode='evil')
