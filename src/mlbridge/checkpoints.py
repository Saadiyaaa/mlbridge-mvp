"""Local read-only discovery for per-key checkpoint snapshots.

 Discovery from scratch: query the project's restic repository with exact
project/session/protocol tags (--tag is AND-filtered by restic), strictly parse
`snapshots --json`, and group by key tag. This module only discovers the
"snapshot-of-record"; never acceptance: pins (scientific acceptance content)
inside the metadata intent only count as accepted after local restore and
verification. Multiple snapshots for the same key are always a conflict; the
latest one is never arbitrarily chosen as accepted.

 Cross-host claim comparison (central control plane's responsibility, later
phase): checkpoint key exact string ownership is computed via SHA-256 into the
key tag; here's `--key` (the same key used at checkpoint time, passed by the
caller) can compare the key with the local registry mapping in
`queried.key_sha256`.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

from .core import BridgeError, project, restic, settings

SESSION = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$')
FULL64 = re.compile(r'^[a-f0-9]{64}$')

# restic tag 过滤语义（restic 0.18.0 `restic help snapshots` + 本机实证，
# 证据: mlbridge-implementation/artifacts/restic-tagsem-verification.txt）：
#   --tag A --tag B   多个 flag 之间是 OR（A 快照 ∪ B 快照）
#   --tag A,B         单个 flag 内逗号分隔才是 AND（同时含 A 与 B）
# 因此 ownership tags 必须合并成单个逗号 --tag；分写成多个 --tag 会把
# 其他 session/其他 key 的快照也查回来，既污染结果又会被严格校验拒掉。

PROJECT_TAG_PREFIX = 'mlbridge-project-'
SESSION_TAG_PREFIX = 'mlbridge-session-checkpoint-'
PROTOCOL_TAG = 'mlbridge-checkpoint-v1'
KEY_TAG_PREFIX = 'mlbridge-checkpoint-key-'
META_TAG_PREFIX = 'mlbridge-checkpoint-meta-'

_NO_ACCEPTANCE_NOTE = (
    'Only tag-based discovery of a snapshot-of-record; any snapshot only counts as '
    'accepted after local restore + acceptance; this output is not an '
    'acceptance statement; multiple snapshots for the same key '
    'must be resolved manually, never deemed latest-first-accepted')


def key_sha256(key: str) -> str:
    """checkpoint key → deterministic 64-hex identity (cross-host claim comparison key)."""
    if not isinstance(key, str) or not key or len(key) > 256:
        raise BridgeError('key must be a 1..256 character checkpoint key')
    return hashlib.sha256(key.encode('utf-8')).hexdigest()


def discover_checkpoints(session: str, project_name: str,
                         *, key: str | None = None) -> dict:
    """Read-only discovery: per checkpoint key give the full64 snapshot and
    "pins still pending acceptance" status; on conflict include all of them.

    Any CLI/timeout/malformed JSON/illegal field is a fail-closed BridgeError
    (positive exit code), never guessed as empty and never retried.
    """
    if not SESSION.fullmatch(session):
        raise BridgeError('session ID is invalid')
    project(project_name)  # Registration existence check (raises on failure)
    scope = [PROJECT_TAG_PREFIX + project_name,
             SESSION_TAG_PREFIX + session,
             PROTOCOL_TAG]
    queried: dict | None = None
    if key is not None:
        identify = key_sha256(key)
        scope.append(KEY_TAG_PREFIX + identify)
        queried = {'checkpoint_key': key, 'key_sha256': identify}
    args = ['snapshots', '--json', '--tag', ','.join(scope)]
    out = restic(project_name, settings(), args, capture=True)
    try:
        entries = json.loads(out.stdout)
    except ValueError as exc:
        raise BridgeError(f'snapshots --json output is not valid JSON: {exc}') from exc
    if not isinstance(entries, list):
        raise BridgeError('snapshots --json output is not a JSON array, strictly rejected')
    grouped: dict[str, list[dict]] = {}
    for obj in entries:
        row = _one_snapshot(obj, project_name, session)
        if queried is not None and row['key_sha256'] != identify:
            # --key 语义是精确 key 归属：backend 若把别的 key 的快照从
            # AND 过滤里漏回来，必然是过滤契约被破坏，绝不静默当作命中。
            raise BridgeError(
                f'snapshot {row["snapshot_id"]} 的 key 归属与查询不符 '
                f'(返回 {row["key_sha256"][:16]}…, 查询 {identify[:16]}…)，'
                'strictly rejected')
        grouped.setdefault(row['key_sha256'], []).append(row)
    keys: list[dict] = []
    for identify, snaps in sorted(grouped.items()):
        state = 'conflict' if len(snaps) > 1 else 'discovered-unverified'
        keys.append({
            'key_sha256': identify,
            'state': state,
            # Discovery never claims acceptance; acceptance is by the research
            # acceptance process performing local restore and comparing pins.
            'pins_verification': 'NOT_ACCEPTED (requires restore + acceptance process)',
            'snapshots': snaps,
        })
    result: dict = {
        'protocol': 'v1',
        'project': project_name,
        'session': session,
        'keys': keys,
        'conflicts': [k['key_sha256'] for k in keys if k['state'] == 'conflict'],
        'discovered_at': datetime.now(timezone.utc).isoformat(),
        'note': _NO_ACCEPTANCE_NOTE,
    }
    if queried is not None:
        queried['found'] = bool(keys)
        queried['state'] = keys[0]['state'] if keys else 'not-found'
        result['queried'] = queried
    return result


def _one_snapshot(obj, project_name: str, session: str) -> dict:
    """Strictly normalize a single snapshot entry: fail closed on any illegal value."""
    if not isinstance(obj, dict):
        raise BridgeError('snapshot entry is not a JSON object, strictly rejected')
    sid = obj.get('id')
    if not isinstance(sid, str) or not FULL64.fullmatch(sid):
        raise BridgeError(
            f'snapshot id is not complete 64-hex: {sid!r} (strict parsing, not guessing)')
    tags = obj.get('tags')
    if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
        raise BridgeError(f'snapshot {sid} tags are missing or not an array of strings, strictly rejected')
    for tag in (PROJECT_TAG_PREFIX + project_name,
                SESSION_TAG_PREFIX + session,
                PROTOCOL_TAG):
        if tag not in tags:
            raise BridgeError(f'snapshot {sid} is missing required ownership tag {tag}, strictly rejected')
    identifies = [t[len(KEY_TAG_PREFIX):] for t in tags if t.startswith(KEY_TAG_PREFIX)]
    if len(identifies) != 1 or not FULL64.fullmatch(identifies[0]):
        raise BridgeError(
            f'snapshot {sid} key tag is missing or duplicate ({identifies!r}), strictly rejected')
    metas = [t[len(META_TAG_PREFIX):] for t in tags if t.startswith(META_TAG_PREFIX)]
    # 协议 v1 的 backup 必然打且只打一个 meta tag（值 = metadata 字节的
    # SHA-256；见 remote_worker._run_backup）：缺失/重复/非 full64 都说明
    # 该快照不是本协议产物或已被动过手脚，fail closed。
    if len(metas) != 1 or not FULL64.fullmatch(metas[0]):
        raise BridgeError(
            f'snapshot {sid} metadata 指纹 tag 必须恰好一个且为 full64 hex '
            f'(实际 {metas!r})，strictly rejected')
    return {
        'snapshot_id': sid,
        'time': obj.get('time'),
        'tags': list(tags),
        'key_sha256': identifies[0],
        'metadata_sha256': metas[0],
    }
