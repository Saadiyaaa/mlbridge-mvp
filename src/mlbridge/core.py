"""Pure-stdlib local configuration, restic and Colab wrappers.

No function in this module writes to the tracked project files. A fetch
explicitly writes the requested data to a user-chosen workspace.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import tomllib
from pathlib import Path
from typing import Any

IDENTITY = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}$")
SNAPSHOT = re.compile(r"^[a-f0-9]{8,64}$")
GPU_CHOICES = frozenset({'T4', 'L4', 'A100', 'H100', 'G4', 'CPU', 'NONE'})


class BridgeError(RuntimeError):
    pass


def ident(value: str) -> str:
    if not IDENTITY.fullmatch(value):
        raise BridgeError("名称仅支持 1–63 位字母、数字、下划线、连字符")
    return value


def app_dir(env_name: str, fallback: str) -> Path:
    # 空字符串视为未设置：否则 Path("") 会产生相对路径 "mlbridge"，
    # 配置/状态会悄悄落到当前工作目录而不是用户主目录。
    base = os.environ.get(env_name) or str(Path.home() / fallback)
    return Path(base).expanduser() / "mlbridge"


def config_dir() -> Path:
    return app_dir("XDG_CONFIG_HOME", ".config")


def state_dir() -> Path:
    return app_dir("XDG_STATE_HOME", ".local/state")


def read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BridgeError(f"找不到配置文件: {path}")
    with path.open("rb") as f:
        return tomllib.load(f)


def project_file(name: str) -> Path:
    return config_dir() / "projects" / f"{ident(name)}.toml"


def project(name: str) -> dict[str, Any]:
    cfg = read_toml(project_file(name))
    if cfg.get("name") != name:
        raise BridgeError("项目配置 name 不匹配")
    for key in ("workspace", "repository"):
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise BridgeError(f"项目配置缺少 {key}")
    return cfg


def settings() -> dict[str, str]:
    cfg = read_toml(config_dir() / "settings.toml")
    store = cfg.get("storage", {})
    for key in ("remote", "prefix", "password_file", "rclone_config"):
        if not isinstance(store.get(key), str) or not store[key]:
            raise BridgeError(f"settings.toml 缺少 storage.{key}")
    return {k: str(v) for k, v in store.items()}


def _colab_settings() -> dict[str, Any]:
    colab = read_toml(config_dir() / "settings.toml").get("colab", {})
    if not isinstance(colab, dict):
        raise BridgeError("settings.toml 的 [colab] 必须是表")
    return colab


def default_gpu() -> str:
    """Read the global Colab default; older settings without [colab] use L4."""
    gpu = _colab_settings().get("default_gpu", "L4")
    if not isinstance(gpu, str) or gpu.upper() not in GPU_CHOICES:
        raise BridgeError("settings.toml 的 colab.default_gpu 无效；支持 T4, L4, A100, H100, G4, CPU (或 NONE)")
    return gpu.upper()


def default_high_mem() -> bool:
    """Older settings without high_mem keep the normal-memory machine shape."""
    high_mem = _colab_settings().get("high_mem", False)
    if not isinstance(high_mem, bool):
        raise BridgeError("settings.toml 的 colab.high_mem 必须为 true 或 false")
    return high_mem


def default_max_active_sessions() -> int:
    """单实例守卫的默认上限；旧配置缺省为 1（保守策略）。

    0 表示显式关闭该检查（例如多实例由用户自行管理）。负数与非法
    类型一律拒绝，绝不静默放宽。
    """
    value = _colab_settings().get("max_active_sessions", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BridgeError(
            "settings.toml 的 colab.max_active_sessions 必须是非负整数（0=关闭检查）")
    return value


def expand(v: str) -> Path:
    return Path(os.path.expandvars(v)).expanduser().resolve()


def repo_url(name: str, store: dict[str, str]) -> str:
    # 云端 remote（如 gdrive）用相对 prefix；local 类 remote 允许绝对路径。
    # 原实现无条件 strip("/")，会把绝对路径变成相对路径，导致 local 后端
    # 在当前工作目录下开新位置。
    raw = store["prefix"].strip()
    parts = [p for p in raw.split("/") if p]
    if not parts or any(p in ("..", ".") for p in parts):
        raise BridgeError("storage.prefix 不合法")
    prefix = ("/" if raw.startswith("/") else "") + "/".join(parts)
    remote = store["remote"].rstrip(":")
    ident(remote)
    return f"rclone:{remote}:{prefix}/{ident(name)}"


def checked_relative(value: str) -> str:
    p = Path(value)
    if not value or value in (".", "..") or p.is_absolute() or ".." in p.parts:
        raise BridgeError(f"拒绝项目根目录/绝对或越界路径: {value!r}")
    if any(part == ".git" for part in p.parts):
        raise BridgeError("禁止备份 .git 目录")
    return p.as_posix().removeprefix("./")


def validate_paths(root: Path, paths: list[str], *, exist: bool = True) -> list[str]:
    if not paths:
        raise BridgeError("至少需要一个输入或输出目录")
    out = []
    root = root.resolve()
    for p in paths:
        rel = checked_relative(p)
        dest = root / rel
        resolved = dest.resolve()
        if not resolved.is_relative_to(root):
            raise BridgeError(f"路径通过软链接越界: {p}")
        if exist and not dest.exists():
            raise BridgeError(f"路径不存在: {dest}")
        out.append(rel)
    return out


def proc(cmd: list[str], *, cwd: Path | None = None,
         env: dict[str, str] | None = None, capture: bool = False,
         timeout: int | None = None) -> subprocess.CompletedProcess:
    try:
        # The GitHub PAT belongs only in the short-lived upload file. Never
        # inherit it into local git/restic/rclone/colab child processes.
        child_env = dict(os.environ if env is None else env)
        child_env.pop('MLBRIDGE_GITHUB_TOKEN', None)
        return subprocess.run(cmd, cwd=cwd, env=child_env, text=True,
                              stdout=subprocess.PIPE if capture else None,
                              stderr=subprocess.PIPE if capture else None,
                              timeout=timeout, check=True)
    except FileNotFoundError as exc:
        raise BridgeError(f"找不到命令: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or exc.stdout or "").strip()
        raise BridgeError(f"命令失败({exc.returncode}): {cmd[0]} {tail[-650:]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise BridgeError(f"命令超时: {cmd[0]}") from exc


def secret_env(store: dict[str, str]) -> dict[str, str]:
    pw, rc = expand(store["password_file"]), expand(store["rclone_config"])
    if not pw.is_file() or not rc.is_file():
        raise BridgeError("找不到 restic password_file 或 rclone_config；先配置好凭据")
    env = dict(os.environ)
    env.update(RESTIC_PASSWORD_FILE=str(pw), RCLONE_CONFIG=str(rc))
    # Respect a user's separately configured rate limit; no credentials in argv.
    return env


def restic(name: str, store: dict[str, str], args: list[str],
           *, cwd: Path | None = None, capture: bool = False):
    return proc(["restic", "-r", repo_url(name, store), *args], cwd=cwd,
                env=secret_env(store), capture=capture)


def parse_backup_output(text: str) -> str:
    found = ""
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("message_type") == "summary":
            found = event.get("snapshot_id", "")
    if not SNAPSHOT.fullmatch(found):
        raise BridgeError("未从 restic JSON 摘要中获得 snapshot_id；请检查仓库快照")
    return found


def publish(name: str, paths: list[str], label: str, *,
            materialize_links: bool = True) -> str:
    """Publish selected relative paths; final-component links are real data by default.

    A single temporary staging tree is used only when one of the explicit
    paths is a symlink. Ordinary paths keep their existing backup behavior.
    """
    cfg, store = project(name), settings()
    root = expand(cfg["workspace"])
    label = ident(label)

    def backup(cwd: Path, rel: list[str]) -> str:
        r = restic(name, store, ["backup", "--json",
                                 "--tag", f"mlbridge-project-{name}",
                                 "--tag", f"mlbridge-artifact-{label}", *rel],
                   cwd=cwd, capture=True)
        return parse_backup_output(r.stdout)

    # Do not follow arbitrary ancestors or walk a parent directory to discover
    # more links: only paths explicitly named on the command line can opt in.
    rel = [checked_relative(p) for p in paths]
    for p in rel:
        parts = Path(p).parts
        for i in range(1, len(parts)):
            if (root / Path(*parts[:i])).is_symlink():
                raise BridgeError(f'发布路径经过软链接祖先，请直接选择链接本身: {p}')
    if materialize_links and any((root / p).is_symlink() for p in rel):
        from .materialize import materialized_inputs
        with materialized_inputs(root, rel) as (stage, selected):
            return backup(stage, selected)
    rel = validate_paths(root, rel)
    if materialize_links:
        # A parent selection like --path models must not silently publish a
        # dangling model symlink. Require selecting that link explicitly.
        for p in rel:
            if (root / p).is_dir():
                def on_error(exc: OSError) -> None:
                    raise BridgeError(f'无法检查发布目录中的软链接: {p} ({exc})') from exc
                for current, dirs, files in os.walk(root / p, onerror=on_error):
                    for entry_name in dirs + files:
                        if (Path(current) / entry_name).is_symlink():
                            raise BridgeError(
                                f'发布目录内有软链接 {Path(current, entry_name).relative_to(root)}；'
                                '请单独 --path 选择该链接，或使用 --no-materialize-links')
    return backup(root, rel)


# restic 在 0.17.0 才引入 restore --overwrite {always,if-changed,never}。
# 更老的版本恢复时无条件覆盖；本机/Colab 的发行版 restic 常为 0.12.x，
# 必须探测后再传参，否则会报 unknown flag。
_OVERWRITE_MIN = (0, 17, 0)
_OVERWRITE_OK: dict[str, bool] = {}


def restic_overwrite_supported() -> bool:
    cached = _OVERWRITE_OK.get("ok")
    if cached is None:
        r = proc(["restic", "version"], capture=True, timeout=15)
        m = re.search(r"restic (\d+)\.(\d+)\.(\d+)", r.stdout)
        cached = bool(m) and tuple(int(g) for g in m.groups()) >= _OVERWRITE_MIN
        _OVERWRITE_OK["ok"] = cached
    return cached


def fetch(name: str, snapshot: str, *, target: Path | None = None,
          overwrite: bool = False) -> Path:
    if not SNAPSHOT.fullmatch(snapshot):
        raise BridgeError("必须显式指定 8–64 位十六进制 snapshot ID，不允许 latest")
    cfg, store = project(name), settings()
    root = expand(cfg["workspace"]) if target is None else target.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    args = ["restore", snapshot, "--target", str(root)]
    if restic_overwrite_supported():
        args += ["--overwrite", "always" if overwrite else "never"]
    elif not overwrite:
        # 老版本 restic 无法表达“从不覆盖”，静默恢复会覆盖已有文件。
        raise BridgeError(
            "本机 restic 版本过旧（支持 --overwrite 需 ≥ 0.17.0）。"
            "默认的“从不覆盖”安全恢复无法保证；请升级 restic，"
            "或明确加 --overwrite 接受覆盖语义")
    # 老版本 restic 在显式请求 overwrite 时默认行为即无条件覆盖，无需传参。
    restic(name, store, args)
    return root


def _normalize_git_url(url: str) -> str:
    # 仅做本地区分用的一致化：去尾斜杠和 .git 后缀，不做网络请求。
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-len(".git")]
    return url


def check_git_pushed(root: Path, repository: str | None = None) -> None:
    """Fail closed：任何无法在本地证明“HEAD 已在配置远端上”的情况都拒绝。

    只用本地 git 元数据（remote-tracking 引用），不联网。唯一绕过方式是
    显式 --allow-dirty（由 git_commit 的 allow_dirty 参数控制）。
    """
    try:
        upstream = proc(["git", "rev-parse", "--abbrev-ref", "@{u}"],
                        cwd=root, capture=True).stdout.strip()
    except BridgeError as exc:
        raise BridgeError(
            "无法确定当前分支的上游跟踪分支（git rev-parse @{u} 失败，"
            "例如无 upstream 或游离 HEAD）；本地无法证明提交已推送到远端。"
            "请先配置 upstream 并 git push，或显式使用 --allow-dirty 跳过"
        ) from exc
    if not upstream:
        raise BridgeError(
            "无法确定当前分支的上游跟踪分支；本地无法证明提交已推送到远端。"
            "请先配置 upstream 并 git push，或显式使用 --allow-dirty 跳过")

    if repository:
        remote = None
        # 最长匹配优先，兼容名字含 '/' 的 remote（罕见但合法）。
        for name in sorted(proc(["git", "remote"], cwd=root,
                                capture=True).stdout.split(), key=len, reverse=True):
            if upstream.startswith(name + "/"):
                remote = name
                break
        if remote is None:
            raise BridgeError(
                f"上游 {upstream} 无法映射到任何 git remote；"
                f"本地无法证明提交已推送到配置的仓库 {repository}。"
                "请修正项目 repository 配置，或显式使用 --allow-dirty 跳过")
        try:
            actual = proc(["git", "remote", "get-url", remote],
                          cwd=root, capture=True).stdout.strip()
        except BridgeError as exc:
            raise BridgeError(
                f"无法读取 git remote {remote} 的 URL；"
                f"本地无法证明提交已推送到配置的仓库 {repository}") from exc
        if _normalize_git_url(actual) != _normalize_git_url(repository):
            raise BridgeError(
                f"项目配置的 repository 是 {repository}，但当前分支的上游是 "
                f"git remote {remote} ({actual})；本地无法证明提交已推送到配置的仓库。"
                "请推送/修改配置使其一致，或显式使用 --allow-dirty 跳过")

    try:
        unpushed = proc(["git", "rev-list", f"{upstream}..HEAD"],
                        cwd=root, capture=True).stdout.strip()
    except BridgeError as exc:
        raise BridgeError(
            f"无法验证本地提交是否已在 {upstream} 上（git rev-list 失败：{exc}）；"
            "本地无法证明提交已推送到远端。请先 git fetch/push，"
            "或显式使用 --allow-dirty 跳过") from exc
    if unpushed:
        count = len(unpushed.splitlines())
        raise BridgeError(
            f"检测到本地有 {count} 个未推送(git push)的代码提交，远端 Colab 将无法拉取；"
            "请先执行 git push，或使用 --allow-dirty 忽略")


def git_commit(root: Path, allow_dirty: bool, repository: str | None = None) -> str:
    """repository 传项目配置的 repo URL 时，同时校验上游即该仓库。

    保持前两个位置参数不变：旧调用 git_commit(root, allow_dirty) 仍有效，
    只是不做 repository 一致性校验；launch 处应传入 cfg['repository']。
    """
    if not root.is_dir():
        raise BridgeError(f"项目目录不存在: {root}")
    if not allow_dirty:
        dirty = proc(["git", "status", "--porcelain", "--untracked-files=no"],
                     cwd=root, capture=True).stdout
        if dirty.strip():
            raise BridgeError("项目存在未提交的已跟踪代码改动；先提交推送或使用 --allow-dirty")
        check_git_pushed(root, repository)
    return proc(["git", "rev-parse", "HEAD"], cwd=root, capture=True).stdout.strip()


def new_settings() -> Path:
    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "settings.toml"
    if path.exists():
        raise BridgeError(f"配置已存在，不覆盖: {path}")
    pw = directory / "restic-password"
    if not pw.exists():
        pw.write_text(secrets.token_urlsafe(36) + "\n", encoding="utf-8")
        pw.chmod(0o600)
    rc = Path.home() / ".config/rclone/rclone.conf"
    path.write_text('[storage]\nremote = "gdrive"\nprefix = "mlbridge"\n'
                    f'password_file = "{pw}"\nrclone_config = "{rc}"\n'
                    '\n[colab]\ndefault_gpu = "L4"\nhigh_mem = false\n', encoding="utf-8")
    path.chmod(0o600)
    return path


def register(name: str, workspace: str, repository: str) -> Path:
    ident(name)
    root = expand(workspace)
    if not root.is_dir():
        raise BridgeError(f"工作区不存在: {root}")
    path = project_file(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise BridgeError(f"项目已经注册: {name}")
    # JSON string literal is also a valid TOML quoted basic string.
    path.write_text(f'name = {json.dumps(name, ensure_ascii=False)}\n'
                    f'workspace = {json.dumps(str(root), ensure_ascii=False)}\n'
                    f'repository = {json.dumps(repository, ensure_ascii=False)}\n'
                    '\n[jobs.example]\ncommand = ["python", "train.py"]\n'
                    'setup = [["python", "-m", "pip", "install", "-r", "requirements.txt"]]\n'
                    'outputs = ["outputs"]\n', encoding="utf-8")
    return path


def job_spec(cfg: dict[str, Any], job: str) -> dict[str, Any]:
    ident(job)
    jobs = cfg.get("jobs", {})
    spec = jobs.get(job)
    if not isinstance(spec, dict):
        raise BridgeError(f"找不到任务 jobs.{job}")
    command = spec.get("command")
    setup = spec.get("setup", [])
    outputs = spec.get("outputs", [])
    if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
        raise BridgeError("job.command 必须为非空 argv 数组")
    if not isinstance(setup, list) or any(not isinstance(x, list) or not x or
           not all(isinstance(y, str) and y for y in x) for x in setup):
        raise BridgeError("job.setup 必须为 argv 数组的数组")
    if not isinstance(outputs, list) or any(not isinstance(x, str) for x in outputs):
        raise BridgeError("job.outputs 必须为路径数组")
    # 允许空 outputs：纯 setup/有外部副作用的任务合法；remote_worker 
    # 在 plan['outputs'] 为空时不做备份。仅对提供的路径做安全校验。
    if outputs:
        validate_paths(Path("/content/mlbridge-workspace"), outputs, exist=False)
    return {"command": command, "setup": setup, "outputs": outputs}
