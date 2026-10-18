"""Stable CLI entrypoint. No source project modifications or SDK integration."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import __version__
from .colab_runner import get_local, get_status, launch, logs, shutdown, watch
from .core import (BridgeError, config_dir, expand, fetch, new_settings,
                   project, publish, register, repo_url, restic, settings,
                   restic_overwrite_supported)


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
    x = sub.add_parser('artifacts', help='列出项目快照及其标签')
    x.add_argument('project')
    x = sub.add_parser('fetch', help='按 snapshot ID 恢复相对路径，不同步代码')
    x.add_argument('project'); x.add_argument('snapshot')
    x.add_argument('--target', type=Path, help='默认当前注册工作区')
    x.add_argument('--overwrite', action='store_true', help='显式允许覆盖已有文件')
    x = sub.add_parser('run', help='创建 Colab session，后台运行，默认等待成功并释放资源')
    x.add_argument('project'); x.add_argument('--job', default='example'); x.add_argument('--input', action='append', default=[])
    x.add_argument('--gpu', default='L4'); x.add_argument('--session')
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
            try:
                ok = restic_overwrite_supported()
                print('restic --overwrite:', 'supported' if ok else 'NOT supported (restic < 0.16.2)')
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
        snap = publish(args.project, args.paths, args.label)
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
                         allow_dirty=args.allow_dirty, session=args.session,
                         allow_empty_input=args.allow_empty_input)
        print('任务已独立启动，session:', session, flush=True)
        print(f'调试：colab ssh -s {session}  （不要用 --rm）', flush=True)
        if not args.detach:
            print('监视状态直到完成；Ctrl-C 不会故意停止远端训练。', flush=True)
            result = watch(session)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result.get('phase') != 'SUCCEEDED':
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
        print('Colab 状态:', json.dumps(current, ensure_ascii=False, indent=2))
    elif a == 'logs':
        logs(args.session, lines=args.lines)
    elif a == 'watch':
        result = watch(args.session, interval=args.interval, auto_stop=not args.no_stop)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get('phase') != 'SUCCEEDED':
            return 1
    elif a == 'stop':
        shutdown(args.session)
        print('Colab session 已停止:', args.session)
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
