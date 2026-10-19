"""私有 GitHub 仓库克隆:环境变量凭据 + GIT_ASKPASS。

不访问网络、不分配 Colab 资源、不需要真实 GitHub 账号。
重点验证:token 不进 job plan / argv / 脚本内容 / 日志;
权限收紧;克隆后(无论成败)立即清理;公开仓库路径不受影响。
"""
import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import stat
import subprocess

import pytest

from mlbridge import colab_runner, core, remote_worker

USER = 'octocat'
TOKEN = 'github_pat_11AAAAAAA0secret0token0123456789'
GOOD_URL = 'https://github.com/demo/demo.git'


@pytest.fixture(autouse=True)
def no_real_credentials(monkeypatch, tmp_path):
    """隔离真实 secrets.env，不读取或改写用户令牌。"""
    monkeypatch.delenv('MLBRIDGE_GITHUB_USERNAME', raising=False)
    monkeypatch.delenv('MLBRIDGE_GITHUB_TOKEN', raising=False)
    monkeypatch.setattr(colab_runner, '_github_secrets_path',
                        lambda: tmp_path / 'isolated-secrets.env')


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))


def _prepare_project(tmp_path, monkeypatch, repository=GOOD_URL):
    """独立临时 XDG 配置 + 带提交的 git 工作区,不触碰真实用户配置。"""
    root = tmp_path / 'proj'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), 'config', 'user.name', 'T'], check=True)
    subprocess.run(['git', '-C', str(root), 'config', 'user.email', 't@example.com'],
                   check=True)
    (root / 'keep.txt').write_text('KEEP')
    subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(root), 'commit', '-qm', 'init'], check=True)
    branch = subprocess.check_output(['git', '-C', str(root), 'branch', '--show-current'],
                                     text=True).strip()
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    subprocess.run(['git', '-C', str(root), 'remote', 'add', 'origin', repository],
                   check=True)
    subprocess.run(['git', '-C', str(root), 'update-ref',
                    f'refs/remotes/origin/{branch}', commit], check=True)
    subprocess.run(['git', '-C', str(root), 'branch', '--set-upstream-to',
                    f'origin/{branch}'], check=True, capture_output=True)
    cfg = tmp_path / 'cfg'
    (cfg / 'mlbridge').mkdir(parents=True)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(cfg))
    scratch = tmp_path / 'creds'
    scratch.mkdir()
    rconf, passwd = scratch / 'rclone.conf', scratch / 'passwd'
    rconf.write_text('[gdrive]\ntype=drive\n')
    passwd.write_text('topsecret')
    (cfg / 'mlbridge' / 'settings.toml').write_text(
        '[storage]\nremote="gdrive"\nprefix="data"\n'
        f'password_file="{passwd}"\nrclone_config="{rconf}"\n')
    core.register('demo', str(root), repository)
    return commit


def _forbid_allocation(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('credential/URL preflight must fail before allocation')
    monkeypatch.setattr(colab_runner, 'cli', deny)


# ---------------------------------------------------------------- preflight

def test_single_username_fails_before_allocation(tmp_path, monkeypatch):
    _prepare_project(tmp_path, monkeypatch)
    monkeypatch.setenv('MLBRIDGE_GITHUB_USERNAME', USER)
    _forbid_allocation(monkeypatch)
    with pytest.raises(core.BridgeError, match='MLBRIDGE_GITHUB'):
        colab_runner.launch('demo', 'example', [], 'CPU', allow_empty_input=True)


@pytest.mark.parametrize('style', [
    'MLBRIDGE_GITHUB_USERNAME={user}\nMLBRIDGE_GITHUB_TOKEN={token}\n',
    'export MLBRIDGE_GITHUB_USERNAME="{user}"\nexport MLBRIDGE_GITHUB_TOKEN="{token}"\n',
    "MLBRIDGE_GITHUB_USERNAME='{user}'\nMLBRIDGE_GITHUB_TOKEN='{token}'\n",
])
def test_awk_reads_private_file_without_sourcing(tmp_path, monkeypatch, style):
    path = tmp_path / 'isolated-secrets.env'
    # Non-target assignments are ignored and never evaluated.
    path.write_text('IGNORED=$(touch /tmp/mlbridge-unwanted-side-effect)\n'
                    + style.format(user=USER, token=TOKEN))
    path.chmod(0o600)
    assert colab_runner.github_auth() == (USER, TOKEN)
    assert not os.path.exists('/tmp/mlbridge-unwanted-side-effect')


def test_awk_file_env_precedence_and_partial_env_rejection(tmp_path, monkeypatch):
    path = tmp_path / 'isolated-secrets.env'
    path.write_text(f'MLBRIDGE_GITHUB_USERNAME={USER}\nMLBRIDGE_GITHUB_TOKEN={TOKEN}\n')
    path.chmod(0o600)
    monkeypatch.setenv('MLBRIDGE_GITHUB_USERNAME', 'other')
    with pytest.raises(core.BridgeError, match='MLBRIDGE_GITHUB'):
        colab_runner.github_auth()  # no mixing sources
    monkeypatch.setenv('MLBRIDGE_GITHUB_TOKEN', 'github_pat_override')
    assert colab_runner.github_auth() == ('other', 'github_pat_override')


def test_awk_file_security_and_duplicate_checks(tmp_path):
    path = tmp_path / 'isolated-secrets.env'
    path.write_text(f'MLBRIDGE_GITHUB_USERNAME={USER}\nMLBRIDGE_GITHUB_TOKEN={TOKEN}\n')
    path.chmod(0o644)
    with pytest.raises(core.BridgeError, match='0600'):
        colab_runner.github_auth()
    path.chmod(0o600)
    with path.open('a') as f:
        f.write(f'MLBRIDGE_GITHUB_TOKEN={TOKEN}\n')
    with pytest.raises(core.BridgeError, match='重复'):
        colab_runner.github_auth()


def test_awk_secrets_file_fails_before_vm_allocation(tmp_path, monkeypatch):
    _prepare_project(tmp_path, monkeypatch)
    path = tmp_path / 'isolated-secrets.env'
    path.write_text(f'MLBRIDGE_GITHUB_USERNAME={USER}\nMLBRIDGE_GITHUB_TOKEN={TOKEN}\n')
    path.chmod(0o600)
    monkeypatch.setattr(colab_runner, 'stage',
                        lambda *args: (_ for _ in ()).throw(AssertionError('not allocated')))
    _forbid_allocation(monkeypatch)
    with pytest.raises(AssertionError, match='allocation'):
        colab_runner.launch('demo', 'example', [], 'CPU', allow_empty_input=True)


def test_token_with_newline_fails_before_allocation(tmp_path, monkeypatch):
    _prepare_project(tmp_path, monkeypatch)
    monkeypatch.setenv('MLBRIDGE_GITHUB_USERNAME', USER)
    monkeypatch.setenv('MLBRIDGE_GITHUB_TOKEN', TOKEN + '\nextra')
    _forbid_allocation(monkeypatch)
    with pytest.raises(core.BridgeError, match='单行'):
        colab_runner.launch('demo', 'example', [], 'CPU', allow_empty_input=True)


def test_single_token_fails_before_allocation(tmp_path, monkeypatch):
    _prepare_project(tmp_path, monkeypatch)
    monkeypatch.setenv('MLBRIDGE_GITHUB_TOKEN', TOKEN)
    _forbid_allocation(monkeypatch)
    with pytest.raises(core.BridgeError, match='MLBRIDGE_GITHUB'):
        colab_runner.launch('demo', 'example', [], 'CPU', allow_empty_input=True)


@pytest.mark.parametrize('url', [
    'http://github.com/demo/demo.git',
    'git@github.com:demo/demo.git',
    'https://gitlab.com/demo/demo.git',
    'https://user:pass@github.com/demo/demo.git',
    'https://token@github.com/demo/demo.git',
    'https://github.com/demo/demo.git?token=x',
    'https://github.com/demo/demo.git#frag',
    'https://github.com/demo/',
    'https://github.com//demo.git',
    '',
])
def test_invalid_private_url_fails_before_allocation(tmp_path, monkeypatch, url):
    _prepare_project(tmp_path, monkeypatch)
    # 直接改写项目 toml 的 repository 行,模拟用户配置了带凭据/畸形 URL 的仓库
    toml = core.project_file('demo')
    lines = toml.read_text().splitlines()
    at = next(i for i, l in enumerate(lines) if l.startswith('repository'))
    lines[at] = f'repository = {json.dumps(url)}'
    toml.write_text('\n'.join(lines) + '\n')
    monkeypatch.setenv('MLBRIDGE_GITHUB_USERNAME', USER)
    monkeypatch.setenv('MLBRIDGE_GITHUB_TOKEN', TOKEN)
    _forbid_allocation(monkeypatch)
    with pytest.raises(core.BridgeError,
                       match='github.com/<owner>|缺少 repository'):
        colab_runner.launch('demo', 'example', [], 'CPU', allow_empty_input=True)


def test_invalid_url_error_does_not_leak_embedded_credentials(tmp_path, monkeypatch):
    _prepare_project(tmp_path, monkeypatch)
    secret_in_url = 'https://user:sup3rsecret@github.com/demo/demo.git'
    toml = core.project_file('demo')
    toml.write_text(toml.read_text().replace(GOOD_URL, secret_in_url))
    monkeypatch.setenv('MLBRIDGE_GITHUB_USERNAME', USER)
    monkeypatch.setenv('MLBRIDGE_GITHUB_TOKEN', TOKEN)
    _forbid_allocation(monkeypatch)
    with pytest.raises(core.BridgeError) as exc:
        colab_runner.launch('demo', 'example', [], 'CPU', allow_empty_input=True)
    assert 'sup3rsecret' not in str(exc.value)


def test_proc_does_not_pass_pat_to_child_processes(monkeypatch):
    monkeypatch.setenv('MLBRIDGE_GITHUB_TOKEN', TOKEN)
    seen = []
    def fake_run(cmd, *, env=None, **kwargs):
        seen.append(env)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(core.subprocess, 'run', fake_run)
    core.proc(['git', 'version'])
    core.proc(['restic', 'version'], env={'MLBRIDGE_GITHUB_TOKEN': TOKEN,
                                          'RESTIC_PASSWORD_FILE': '/fake'})
    assert len(seen) == 2
    assert all('MLBRIDGE_GITHUB_TOKEN' not in env for env in seen)
    assert seen[1]['RESTIC_PASSWORD_FILE'] == '/fake'


# ------------------------------------------------------- staging + plan 内容

def test_launch_stages_auth_last_with_restrictive_mode_and_no_token_leak(
        tmp_path, monkeypatch):
    _prepare_project(tmp_path, monkeypatch)
    monkeypatch.setenv('MLBRIDGE_GITHUB_USERNAME', USER)
    monkeypatch.setenv('MLBRIDGE_GITHUB_TOKEN', TOKEN)
    stage_order, scripts, auth_info, job_plan = [], [], [], [None]
    captured_cli = []

    def fake_cli(*args, capture=False, timeout=None):
        captured_cli.append(args)
        return subprocess.CompletedProcess(args, 0)

    def fake_stage(session, src, remote_basename):
        stage_order.append(remote_basename)
        if remote_basename == colab_runner.GIT_AUTH_BASENAME:
            auth_info.append({
                'mode': stat.S_IMODE(os.stat(src).st_mode),
                'content': open(src).read(),
                'src': str(src),
            })
        if remote_basename == 'mlbridge_job.json':
            job_plan[0] = json.loads(open(src).read())

    monkeypatch.setattr(colab_runner, 'cli', fake_cli)
    monkeypatch.setattr(colab_runner, 'stage', fake_stage)
    monkeypatch.setattr(colab_runner, 'put_script',
                        lambda s, code: scripts.append(code))

    session = colab_runner.launch('demo', 'example', [], 'CPU', allow_empty_input=True)
    assert session

    # 上传顺序:凭据文件在所有其他 staging 之后
    assert stage_order[:4] == ['mlbridge_worker.py', 'mlbridge_job.json',
                               'mlbridge_rclone.conf', 'mlbridge_restic_password']
    assert stage_order[4] == colab_runner.GIT_AUTH_BASENAME

    # 本地临时凭据文件:0600、内容正确、launch 结束后本地副本已删除
    assert auth_info[0]['mode'] == 0o600
    assert auth_info[0]['content'] == f'{USER}\n{TOKEN}\n'
    assert not os.path.exists(auth_info[0]['src'])

    # 凭据上传后立即注入收紧权限 + 生成 askpass 的 setup 脚本
    setup = colab_runner._github_setup_script()
    assert scripts[1] == setup
    assert '0o600' in setup and '0o700' in setup

    # token 不在任何脚本、cli 参数(含 job plan 上传路径)或远端 URL 中
    for code in scripts:
        assert TOKEN not in code
    assert not any(TOKEN in str(a) for a in captured_cli)
    assert TOKEN not in json.dumps(job_plan[0])
    assert job_plan[0]['repository'] == GOOD_URL  # git remote URL 干净,无凭据
    # 本地任务记录(不含 plan)也必须无 token
    record = (core.state_dir() / 'jobs' / f'{session}.json').read_text()
    assert TOKEN not in record


def test_git_http_basic_challenge_uses_askpass_file_without_url_token(tmp_path):
    """真实 git 对本地 HTTP Basic challenge 发出两次 askpass 提示。"""
    headers = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            headers.append(self.headers.get('Authorization'))
            self.send_response(403 if headers[-1] else 401)
            self.send_header('WWW-Authenticate', 'Basic realm="test"')
            self.end_headers()
        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    auth = tmp_path / colab_runner.GIT_AUTH_BASENAME
    auth.write_text(f'{USER}\n{TOKEN}\n')
    auth.chmod(0o600)
    askpass = tmp_path / colab_runner.GIT_ASKPASS_BASENAME
    askpass.write_text(colab_runner._askpass_source(str(auth)))
    askpass.chmod(0o700)
    url = f'http://127.0.0.1:{server.server_port}/demo.git'
    env = {**os.environ, 'GIT_ASKPASS': str(askpass), 'GIT_TERMINAL_PROMPT': '0',
           'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null'}
    try:
        proc = subprocess.run(['git', 'ls-remote', url], env=env, text=True,
                              capture_output=True, timeout=15)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert proc.returncode != 0  # fake server denies access after challenge
    assert headers[0] is None
    assert 'Basic ' + base64.b64encode(f'{USER}:{TOKEN}'.encode()).decode() in headers
    assert TOKEN not in proc.stderr and TOKEN not in url


def test_cleanup_script_removes_github_auth():
    script = colab_runner.cleanup_script()
    assert colab_runner.GIT_AUTH_BASENAME in script
    assert colab_runner.GIT_ASKPASS_BASENAME in script
    assert 'mlbridge_rclone.conf' in script
    assert 'mlbridge_restic_password' in script


def test_askpass_script_reads_auth_file(tmp_path):
    """askpass 可执行脚本:Username 提示返回第一行,其余提示返回第二行。"""
    auth = tmp_path / colab_runner.GIT_AUTH_BASENAME
    auth.write_text(f'{USER}\n{TOKEN}\n')
    askpass = tmp_path / colab_runner.GIT_ASKPASS_BASENAME
    askpass.write_text(colab_runner._askpass_source(str(auth)))
    askpass.chmod(0o700)
    user = subprocess.run([str(askpass), "Username for 'https://github.com': "],
                          capture_output=True, text=True)
    assert user.stdout.strip() == USER
    token = subprocess.run([str(askpass), 'Password: '],
                           capture_output=True, text=True)
    assert token.stdout.strip() == TOKEN


# ------------------------------------------------------------ 远端 worker

# 最小 restic 替身:version 用于 restore_args 探测;restore/backup 只需成功。
# 输出列表为空,不触发备份路径。
MIN_RESTIC = '''#!/bin/sh
case "$1" in
  version) echo "restic 0.17.0 compiled with go1.22.5 on linux/amd64";;
esac
exit 0
'''


def _worker_fixtures(tmp_path, monkeypatch, source_repo, commit):
    """构造远端 /content 布局(含已暂存的私有凭据)+ git 环境记录包装器。"""
    base = tmp_path / 'remote-root'
    base.mkdir()
    (base / 'mlbridge_rclone.conf').write_text('[gdrive]\ntype=drive\n')
    (base / 'mlbridge_restic_password').write_text('test')
    (base / colab_runner.GIT_AUTH_BASENAME).write_text(f'{USER}\n{TOKEN}\n')
    askpass = base / colab_runner.GIT_ASKPASS_BASENAME
    askpass.write_text(
        colab_runner._askpass_source(str(base / colab_runner.GIT_AUTH_BASENAME)))
    askpass.chmod(0o700)
    plan = {'job_id': 'test', 'session': 'test', 'project': 'demo',
            'repository': source_repo, 'commit': commit,
            'restic_repository': 'rclone:gdrive:store/demo',
            'inputs': ['0' * 64], 'setup': [], 'command': ['true'], 'outputs': []}
    (base / 'mlbridge_job.json').write_text(json.dumps(plan))

    env_log = tmp_path / 'git_env.log'
    env_log.write_text('')
    real_git = subprocess.check_output(['sh', '-c', 'command -v git'],
                                       text=True).strip()
    fakebin = tmp_path / 'fakebin'
    fakebin.mkdir(exist_ok=True)
    wrapper = fakebin / 'git'
    wrapper.write_text(
        '#!/bin/sh\n'
        f'printf \'GIT_ASKPASS=%s\\n\' "$GIT_ASKPASS" >> {env_log}\n'
        f'printf \'GIT_TERMINAL_PROMPT=%s\\n\' "$GIT_TERMINAL_PROMPT" >> {env_log}\n'
        f'exec "{real_git}" "$@"\n')
    wrapper.chmod(0o755)
    (fakebin / 'restic').write_text(MIN_RESTIC)
    (fakebin / 'restic').chmod(0o755)
    (fakebin / 'rclone').write_text('#!/bin/sh\nexit 0\n')
    (fakebin / 'rclone').chmod(0o755)
    monkeypatch.setenv('PATH', str(fakebin) + os.pathsep + os.environ['PATH'])
    monkeypatch.setattr(remote_worker, 'BASE', base)
    monkeypatch.setattr(remote_worker, 'STATE', base / 'mlbridge_state.json')
    monkeypatch.setattr(remote_worker, 'PLAN', base / 'mlbridge_job.json')
    return base, env_log


def test_worker_private_clone_uses_askpass_and_cleans_up(tmp_path, monkeypatch,
                                                         capsys):
    """私有克隆走 GIT_ASKPASS,结束后立即删除凭据;token 不进日志。"""
    commit = _prepare_project(tmp_path, monkeypatch)
    # 本地 git 仓库充当克隆源(无网络);路径即 plan['repository']
    base, env_log = _worker_fixtures(tmp_path, monkeypatch, str(tmp_path / 'proj'),
                                     commit)
    remote_worker.main()
    # 只看 clone 那一次调用的环境(后续 git checkout 也会经过包装器,
    # 同样带 env 或不带,取第一次出现的带值行)
    log = env_log.read_text().splitlines()
    assert 'GIT_TERMINAL_PROMPT=0' in log
    askpass_lines = [l for l in log
                     if l.startswith('GIT_ASKPASS=') and l != 'GIT_ASKPASS=']
    assert askpass_lines == [f'GIT_ASKPASS={base / colab_runner.GIT_ASKPASS_BASENAME}']
    state = json.loads((base / 'mlbridge_state.json').read_text())
    assert state['phase'] == 'SUCCEEDED'
    assert not (base / colab_runner.GIT_AUTH_BASENAME).exists()
    assert not (base / colab_runner.GIT_ASKPASS_BASENAME).exists()
    assert TOKEN not in capsys.readouterr().out
    assert TOKEN not in json.dumps(state)


def test_worker_public_repo_ignores_askpass(tmp_path, monkeypatch):
    """没有凭据文件时(公开仓库),不设置 GIT_ASKPASS,行为与之前一致。"""
    commit = _prepare_project(tmp_path, monkeypatch)
    base, env_log = _worker_fixtures(tmp_path, monkeypatch, str(tmp_path / 'proj'),
                                     commit)
    (base / colab_runner.GIT_AUTH_BASENAME).unlink()
    (base / colab_runner.GIT_ASKPASS_BASENAME).unlink()
    remote_worker.main()
    log = env_log.read_text().splitlines()
    assert 'GIT_ASKPASS=' in log and 'GIT_TERMINAL_PROMPT=' in log
    assert not any(l.startswith('GIT_ASKPASS=/') for l in log)
    state = json.loads((base / 'mlbridge_state.json').read_text())
    assert state['phase'] == 'SUCCEEDED'


def test_worker_clone_failure_removes_credentials_and_hides_token(tmp_path,
                                                                   monkeypatch,
                                                                   capsys):
    """克隆失败也必须立即删除凭据;失败信息里不出现 token。"""
    commit = _prepare_project(tmp_path, monkeypatch)
    base, env_log = _worker_fixtures(
        tmp_path, monkeypatch, str(tmp_path / 'no-such-repository-xyz'), commit)
    with pytest.raises(subprocess.CalledProcessError):
        remote_worker.main()
    state = json.loads((base / 'mlbridge_state.json').read_text())
    assert state['phase'] == 'FAILED'
    assert TOKEN not in state.get('error', '')
    assert TOKEN not in capsys.readouterr().out
    assert not (base / colab_runner.GIT_AUTH_BASENAME).exists()
    assert not (base / colab_runner.GIT_ASKPASS_BASENAME).exists()
