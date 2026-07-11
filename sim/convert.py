"""trace JSONL 스키마(§6) 검증·변환기.

CLAUDE.md §6: "기존 trace 로그가 §6 스키마와 다르면 변환기(sim/convert.py)를 먼저 작성한다."
현 시점 기존 trace 자산이 없으므로(2026-07-11 확인, docs/ENV.md), 이 모듈은
  1) schema v1 검증기 + 정규화 reader
  2) 범용 "token log" 포맷 → v1 변환기 (외부 로그 인입용 플러그 포인트)
를 제공한다.

schema v1 (§6) — 선택 확장 필드(A-5, docs/DECISIONS.md):
  record["events"]: [{"pos": int, "type": "file_edit", "file": str}]
  write/edit tool-call 완료 지점의 invalidation 이벤트. 온라인에서는 SegmentFSM이
  방출하며 tracer가 함께 기록한다. 없으면 빈 목록으로 처리한다.
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass, field

from core.types import Scope, Segment

SCHEMA_VERSION = 1

_REQUIRED_TOP = ("schema_version", "request_id", "ts", "scope", "model", "tokenizer_hash", "steps")
_REQUIRED_STEP = ("pos", "proposed", "accepted_len", "bonus", "topk_ids", "topk_logp_q8", "seg")
_REQUIRED_SCOPE = ("tenant", "repo", "session")


@dataclass(frozen=True)
class TraceEvent:
    pos: int  # tool-call 생성 완료 지점 (파일-위치 결속 범위의 끝)
    type: str
    file: str
    bump_pos: int = -1  # epoch bump 발화 지점 (기본: pos). write 인자 시작 경계에서
    # 발화하면 새 내용 harvest가 새 epoch에 실려 다음 편집에서 재사용된다 (A-7)

    def effective_bump_pos(self) -> int:
        return self.bump_pos if self.bump_pos >= 0 else self.pos


@dataclass
class TraceRequest:
    """정규화된 요청 1건: realized 위치 단위로 펼친 뷰."""

    request_id: str
    ts: int
    scope: Scope
    model: str
    tokenizer_hash: str
    tokens: list[int] = field(default_factory=list)  # realized 스트림
    seg: list[int] = field(default_factory=list)  # 위치별 seg
    topk_ids: list[tuple[int, ...]] = field(default_factory=list)
    topk_logp_q8: list[tuple[int, ...]] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tokens)


def validate_record(rec: dict) -> list[str]:
    errs = []
    for k in _REQUIRED_TOP:
        if k not in rec:
            errs.append(f"missing top-level key: {k}")
    if errs:
        return errs
    if rec["schema_version"] != SCHEMA_VERSION:
        errs.append(f"schema_version {rec['schema_version']} != {SCHEMA_VERSION}")
    for k in _REQUIRED_SCOPE:
        if k not in rec["scope"]:
            errs.append(f"missing scope key: {k}")
    for i, st in enumerate(rec["steps"]):
        for k in _REQUIRED_STEP:
            if k not in st:
                errs.append(f"steps[{i}] missing key: {k}")
                break
        else:
            n_real = st["accepted_len"] + 1
            if st["accepted_len"] > len(st["proposed"]):
                errs.append(f"steps[{i}] accepted_len > len(proposed)")
            for name in ("topk_ids", "topk_logp_q8", "seg"):
                if len(st[name]) not in (n_real, 0):
                    errs.append(
                        f"steps[{i}] {name} 길이 {len(st[name])} != realized {n_real}"
                    )
            for q_row in st["topk_logp_q8"]:
                if any(not (0 <= q <= 255) for q in q_row):
                    errs.append(f"steps[{i}] topk_logp_q8 out of q8 range")
                    break
    return errs


def normalize_record(rec: dict) -> TraceRequest:
    """v1 record → realized-위치 단위 TraceRequest. (검증은 호출측 책임)"""
    sc = rec["scope"]
    req = TraceRequest(
        request_id=rec["request_id"],
        ts=rec["ts"],
        scope=Scope(
            tenant=sc["tenant"],
            repo=sc["repo"],
            session=sc["session"],
            instance_id=sc.get("instance_id", ""),
        ),
        model=rec["model"],
        tokenizer_hash=rec["tokenizer_hash"],
    )
    for st in rec["steps"]:
        realized = list(st["proposed"][: st["accepted_len"]]) + [st["bonus"]]
        n = len(realized)
        seg = st["seg"] or [int(Segment.TEXT)] * n
        tki = st["topk_ids"] or [[] for _ in range(n)]
        tkq = st["topk_logp_q8"] or [[] for _ in range(n)]
        req.tokens.extend(realized)
        req.seg.extend(int(s) for s in seg)
        req.topk_ids.extend(tuple(int(t) for t in row) for row in tki)
        req.topk_logp_q8.extend(tuple(int(q) for q in row) for row in tkq)
    for ev in rec.get("events", []):
        req.events.append(
            TraceEvent(
                pos=int(ev["pos"]),
                type=ev["type"],
                file=ev["file"],
                bump_pos=int(ev.get("bump_pos", -1)),
            )
        )
    req.events.sort(key=lambda e: e.pos)
    return req


def read_traces(patterns: list[str], strict: bool = True) -> list[TraceRequest]:
    """glob 패턴들에서 v1 JSONL을 읽어 ts 순으로 정렬해 반환."""
    reqs: list[TraceRequest] = []
    paths = sorted(p for pat in patterns for p in glob.glob(pat))
    if not paths:
        raise FileNotFoundError(f"no trace files match: {patterns}")
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                errs = validate_record(rec)
                if errs:
                    msg = f"{path}:{ln}: " + "; ".join(errs)
                    if strict:
                        raise ValueError(msg)
                    continue
                reqs.append(normalize_record(rec))
    reqs.sort(key=lambda r: (r.ts, r.request_id))
    return reqs


# ---------------------------------------------------------------- converters
def convert_token_log(rec: dict) -> dict:
    """범용 token-log 포맷 → v1 (vanilla steps: proposed=[], step당 1 realized).

    입력: {"request_id","ts","scope":{...},"model","tokenizer_hash",
           "tokens":[...], "seg":[...]?, "topk_ids":[[..]]?, "topk_logp_q8":[[..]]?,
           "events":[...]? }
    """
    toks = rec["tokens"]
    n = len(toks)
    seg = rec.get("seg") or [int(Segment.TEXT)] * n
    tki = rec.get("topk_ids") or [[] for _ in range(n)]
    tkq = rec.get("topk_logp_q8") or [[] for _ in range(n)]
    steps = []
    for i, t in enumerate(toks):
        steps.append(
            {
                "pos": i,
                "proposed": [],
                "accepted_len": 0,
                "bonus": int(t),
                "topk_ids": [list(tki[i])] if tki[i] else [],
                "topk_logp_q8": [list(tkq[i])] if tkq[i] else [],
                "seg": [int(seg[i])],
                "t_us": 0,
            }
        )
    out = {
        "schema_version": SCHEMA_VERSION,
        "request_id": rec["request_id"],
        "ts": rec["ts"],
        "scope": rec["scope"],
        "model": rec.get("model", "unknown"),
        "tokenizer_hash": rec["tokenizer_hash"],
        "steps": steps,
        "final_text_sha": rec.get("final_text_sha", ""),
    }
    if rec.get("events"):
        out["events"] = rec["events"]
    return out


_CONVERTERS = {"v1": lambda r: r, "token_log": convert_token_log}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="trace 변환기: 외부 포맷 → schema v1 JSONL")
    ap.add_argument("--input", required=True, help="입력 JSONL")
    ap.add_argument("--format", choices=sorted(_CONVERTERS), required=True)
    ap.add_argument("--output", required=True, help="출력 v1 JSONL")
    args = ap.parse_args(argv)

    conv = _CONVERTERS[args.format]
    n_ok = 0
    with open(args.input, encoding="utf-8") as fi, open(args.output, "w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            rec = conv(json.loads(line))
            errs = validate_record(rec)
            if errs:
                raise ValueError("; ".join(errs))
            fo.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
            n_ok += 1
    print(f"converted {n_ok} records → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
