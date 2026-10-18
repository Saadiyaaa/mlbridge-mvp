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


class BridgeError(RuntimeError):
    pass


def ident(value: str) -> str:
    if not IDENTITY.fullmatch(value):
        raise BridgeError("名称仅支持 1–63 位字母、数字、下划线、连字符")
    return value


def app_dir(env_name: str, fallback: str) -> Path:
    return Path(os.environ.get(env_name, str(Path.home() / fallback))).expanduser() / "mlbridge"


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
        return subprocess.run(cmd, cwd=cwd, env=env, text=True,
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


def publish(name: str, paths: list[str], label: str) -> str:
    cfg, store = project(name), settings()
    root = expand(cfg["workspace"])
    rel = validate_paths(root, paths)
    label = ident(label)
    r = restic(name, store, ["backup", "--json",
                             "--tag", f"mlbridge-project-{name}",
                             "--tag", f"mlbridge-artifact-{label}", *rel],
               cwd=root, capture=True)
    return parse_backup_output(r.stdout)


# restic 在 0.16.2 才引入 restore --overwrite {always,if-changed,never}。
# 更老的版本恢复时无条件覆盖；本机/Colab 的发行版 restic 常为 0.12.x，
# 必须探测后再传参，否则会报 unknown flag。
_OVERWRITE_MIN = (0, 16, 2)
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
            "本机 restic 版本过旧（支持 --overwrite 需 ≥ 0.16.2）。"
            "默认的“从不覆盖”安全恢复无法保证；请升级 restic，"
            "或明确加 --overwrite 接受覆盖语义")
    # 老版本 restic 在显式请求 overwrite 时默认行为即无条件覆盖，无需传参。
    restic(name, store, args)
    return root


def check_git_pushed(root: Path) -> None:
    try:
        upstream = proc(["git", "rev-parse", "--abbrev-ref", "@{u}"],
                        cwd=root, capture=True).stdout.strip()
    except BridgeError:
        # 无上游跟踪分支（例如纯本地测试仓库或新分支），不阻断
        return

    if upstream:
        try:
            unpushed = proc(["git", "rev-list", f"{upstream}..HEAD"],
                            cwd=root, capture=True).stdout.strip()
            if unpushed:
                count = len(unpushed.splitlines())
                raise BridgeError(
                    f"检测到本地有 {count} 个未推送(git push)的代码提交，远端 Colab 将无法拉取；"
                    "请先执行 git push，或使用 --allow-dirty 忽略"
                )
        except BridgeError as exc:
            if "未推送" in str(exc):
                raise


def git_commit(root: Path, allow_dirty: bool) -> str:
    if not root.is_dir():
        raise BridgeError(f"项目目录不存在: {root}")
    if not allow_dirty:
        dirty = proc(["git", "status", "--porcelain", "--untracked-files=no"],
                     cwd=root, capture=True).stdout
        if dirty.strip():
            raise BridgeError("项目存在未提交的已跟踪代码改动；先提交推送或使用 --allow-dirty")
        check_git_pushed(root)
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
                    f'password_file = "{pw}"\nrclone_config = "{rc}"\n', encoding="utf-8")
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
