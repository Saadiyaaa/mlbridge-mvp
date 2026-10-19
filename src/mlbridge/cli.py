"""Stable CLI entrypoint. No source project modifications or SDK integration."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import __version__
from .colab_runner import get_local, get_status, launch, logs, shutdown, watch
from .ssh_observe import ssh_observe
from .core import (BridgeError, config_dir, expand, fetch, new_settings,
                   proc, project, publish, register, repo_url, restic,
                   secret_env, settings, restic_overwrite_supported)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='mlbridge', description='项目外部的数据快照 / Colab 运行适配器')
    p.add_argument('--version', action='version', version=__version__)
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('init', help='首次创建全局配置和 restic 密码')
    r = sub.add_parser('register', help='注册现有项目，不修改项目文件')
    r.add_argument('name'); r.add_argument('--workspace', required=True); r.add_argument('--repo', required=True)
    d = sub.add_parser('doctor', help='检查依赖/凭据文件')
    d.add_argument('--project')
    x = sub.add_parser('repo-init', help='明确创建此项目的远程 restic 仓库（仅一次）')
    x.add_argument('project')
    x = sub.add_parser('publish', help='发布一组已完成的输入或输出目录')
    x.add_argument('project'); x.add_argument('--label', required=True)
    x.add_argument('--path', action='append', required=True, dest='paths')
    x.add_argument('--no-materialize-links', action='store_true',
                   help='不物化显式选中的软链接（外部链接将被拒绝）')
    x = sub.add_parser('artifacts', help='列出项目快照及其标签')
    x.add_argument('project')
    x = sub.add_parser('fetch', help='按 snapshot ID 恢复相对路径，不同步代码')
    x.add_argument('project'); x.add_argument('snapshot')
    x.add_argument('--target', type=Path, help='默认当前注册工作区')
    x.add_argument('--overwrite', action='store_true', help='显式允许覆盖已有文件')
    x = sub.add_parser('run', help='创建 Colab session，后台运行，默认等待成功并释放资源')
    x.add_argument('project'); x.add_argument('--job', default='example'); x.add_argument('--input', action='append', default=[])
    x.add_argument('--gpu', help='GPU 类型(T4, L4, A100, H100, G4)或 CPU/NONE；省略时使用 settings.toml 的 [colab].default_gpu（默认 L4）')
    x.add_argument('--high-mem', action=argparse.BooleanOptionalAction, default=None,
                   help='请求高内存机型（如 A100）；--no-high-mem 可覆盖配置，省略时读取 [colab].high_mem')
    x.add_argument('--session')
    x.add_argument('--max-active-sessions', type=int, default=None,
                   help='单实例守卫：分配前用 `colab sessions` 核对活跃实例数，超过则拒绝（不做任何清理）。'
                        '默认读 settings.toml [colab].max_active_sessions（缺省 1）；0 只关闭并发上限检查，'
                        '复用已在 colab 活跃的 --session 名仍会被 fail-closed 拒绝')
    x.add_argument('--detach', action='store_true', help='不等待；之后用 watch 监视/自动停止')
    x.add_argument('--allow-dirty', action='store_true', help='使用 HEAD，忽略本地未提交代码变更')
    x.add_argument('--allow-empty-input', action='store_true', help='明确确认该任务无需输入快照')
    x = sub.add_parser('status', help='查询运行时状态/最近错误')
    x.add_argument('session')
    x = sub.add_parser('logs', help='只获取 Colab 任务日志末尾，适合 AI 远程观察')
    x.add_argument('session'); x.add_argument('--lines', type=int, default=100)
    x = sub.add_parser('watch', help='重新连接已提交的任务；成功后可自动停止 GPU')
    x.add_argument('session'); x.add_argument('--interval', type=int, default=20)
    x.add_argument('--no-stop', action='store_true')
    x = sub.add_parser('stop', help='手动释放 session（如果未发布输出，可能丢失数据）')
    x.add_argument('session')
    x = sub.add_parser('ssh', help='SSH 单次读取既有实例的系统状态；支持自定义远端命令，不创建/停止实例')
    x.add_argument('session')
    x.add_argument('--timeout', type=int, default=60, help='连接与执行的总秒数（5–3600）')
    x.add_argument('--identity', help='已有的 ed25519/ecdsa 私钥路径（默认先检查 MLBridge 配置目录 ssh/id_ed25519）')
    x.add_argument('command', nargs=argparse.REMAINDER, help='可选：-- 后接远端命令及参数；不填则输出系统快照')
    return p


def execute(args):
    a = args.action
    if a == 'init':
        print('配置创建:', new_settings())
        print('下一步：修改 settings.toml 指向已经认证好的 rclone.conf，再 register / repo-init。')
    elif a == 'register':
        print('项目注册:', register(args.name, args.workspace, args.repo))
    elif a == 'doctor':
        import platform
        print('python:', platform.python_version())
        for binary in ('git', 'restic', 'rclone', 'colab'):
            print(f'{binary}: {shutil.which(binary) or "MISSING"}')
        print('配置目录:', config_dir())
        try:
            s = settings()
            print('rclone config:', 'EXISTS' if expand(s['rclone_config']).is_file() else 'MISSING')
            pw = expand(s['password_file'])
            mode = oct(pw.stat().st_mode & 0o777) if pw.is_file() else '-'
            print('restic password file:', 'EXISTS' if pw.is_file() else 'MISSING', mode)
            if shutil.which('rclone') and expand(s['rclone_config']).is_file():
                # 只捕获 BridgeError（rclone 退出码非零/超时等运行时失败）；
                # 编程错误（NameError/KeyError 等）绝不能被伪装成“网络未连通”。
                try:
                    proc(['rclone', 'about', f"{s['remote']}:"],
                         cwd=Path.home(), env=secret_env(s), capture=True, timeout=5)
                    print(f'rclone remote ({s["remote"]}): CONNECTED (已验证云端连通性)')
                except BridgeError as exc:
                    print(f'rclone remote ({s["remote"]}): UNVERIFIED '
                          f'(rclone 运行失败或网络未连通: {exc})')
            try:
                ok = restic_overwrite_supported()
                print('restic --overwrite:', 'supported' if ok else 'NOT supported (restic < 0.17.0)')
            except BridgeError:
                print('restic --overwrite: unknown (restic 未安装)')
            if args.project:
                print('工作区:', project(args.project)['workspace'])
                print('restic remote:', repo_url(args.project, s))
        except BridgeError as exc:
            print('配置待完成:', exc)
    elif a == 'repo-init':
        project(args.project)
        restic(args.project, settings(), ['init'])
        print('已初始化该项目的 restic 仓库。')
    elif a == 'publish':
        snap = publish(args.project, args.paths, args.label,
                       materialize_links=not args.no_materialize_links)
        print(json.dumps({'project': args.project, 'label': args.label,
                          'snapshot_id': snap}, ensure_ascii=False))
    elif a == 'fetch':
        target = fetch(args.project, args.snapshot,
                       target=args.target, overwrite=args.overwrite)
        print('数据已恢复到:', target)
    elif a == 'artifacts':
        project(args.project)
        out = restic(args.project, settings(), ['snapshots', '--json',
             '--tag', 'mlbridge-project-' + args.project], capture=True)
        for obj in json.loads(out.stdout):
            print(json.dumps({'id': obj.get('id'), 'time': obj.get('time'),
                              'tags': obj.get('tags')}, ensure_ascii=False))
    elif a == 'run':
        session = launch(args.project, args.job, args.input, args.gpu,
                         high_mem=args.high_mem, allow_dirty=args.allow_dirty, session=args.session,
                         allow_empty_input=args.allow_empty_input,
                         max_active_sessions=args.max_active_sessions)
        print('任务已独立启动，session:', session, flush=True)
        print(f'调试：colab ssh -s {session} （不要用 --rm）', flush=True)
        if not args.detach:
            print('监视状态直到完成；Ctrl-C 不会故意停止远端训练。', flush=True)
            result = watch(session)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result.get('phase') != 'SUCCEEDED' or result.get('stop_error'):
                return 1
        else:
            print(f'稍后执行 mlbridge watch {session}，在完成时释放 GPU。')
    elif a == 'status':
        try:
            record = get_local(args.session)
            print('本地记录:', json.dumps(record, ensure_ascii=False))
        except BridgeError:
            record = None
        current = (record.get('result') if record and record.get('stopped') and record.get('result')
                   else get_status(args.session))
        if (current.get('phase') == 'UNREACHABLE' and record and record.get('last_known')):
            # VM 消失 ≠ 运行失败：当前状态仍是 UNREACHABLE，另附消失前
            # 最后一次看到的远端状态供人工判断。
            current = {**current, 'last_known': record['last_known']}
        print('Colab 状态:', json.dumps(current, ensure_ascii=False, indent=2))
    elif a == 'logs':
        logs(args.session, lines=args.lines)
    elif a == 'watch':
        result = watch(args.session, interval=args.interval, auto_stop=not args.no_stop)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get('phase') != 'SUCCEEDED' or result.get('stop_error'):
            return 1
    elif a == 'stop':
        shutdown(args.session)
        print('Colab session 已停止:', args.session)
    elif a == 'ssh':
        if args.command[:1] != ['--'] and args.command[:1] in (['--timeout'], ['--identity']):
            raise BridgeError('ssh 的 --timeout/--identity 需写在 <session> 前；'
                              '远端命令请放在 <session> -- 后')
        cmd = args.command[1:] if args.command[:1] == ['--'] else args.command
        ssh_observe(args.session, cmd or None, timeout=args.timeout, identity=args.identity)
    return 0


def main():
    try:
        sys.exit(execute(parser().parse_args()))
    except BridgeError as exc:
        print(f'[mlbridge] 错误：{exc}', file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print('\n[mlbridge] 本地监视已取消；如远端任务仍在，请用 status/watch 检查并手动释放资源。', file=sys.stderr)
        sys.exit(130)


if __name__ == '__main__':
    main()
