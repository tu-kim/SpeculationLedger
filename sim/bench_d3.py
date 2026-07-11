"""D3 μbench: rolling hash stack vs suffix automaton (CLAUDE.md §11 D3).

결정 기준(제안): store 키링에는 고정 차수 2..8 서명만 필요하므로,
rolling stack이 토큰당 갱신+스택 산출에서 SA보다 빠르고 메모리가 유계면 rolling 채택.
SA는 G1 재발률 '분석'용(정확 최장 match)으로만 유지한다.
결과는 results/d3_bench/bench.json — wall-clock 포함이므로 I4 golden 대상이 아니다.
판정 수치는 docs/DECISIONS.md D3에 옮겨 적는다.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

from core.signature import RollingSigStack, SuffixAutomaton
from core.store import LedgerStore, StoreParams
from core.types import Scope, VerifyOutcome


def _stream(n: int, vocab: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    base = [rng.randrange(16, vocab) for _ in range(n // 4)]
    out: list[int] = []
    while len(out) < n:  # 재발 구조: 과거 구간 재방출 + 노이즈
        if out and rng.random() < 0.5:
            s = rng.randrange(0, max(1, len(out) - 32))
            out.extend(out[s : s + rng.randint(4, 24)])
        else:
            s = rng.randrange(0, max(1, len(base) - 16))
            out.extend(base[s : s + rng.randint(4, 16)])
    return out[:n]


def bench(n_tokens: int = 50_000, vocab: int = 8000, seed: int = 7, lookups: int = 20_000) -> dict:
    toks = _stream(n_tokens, vocab, seed)

    t0 = time.perf_counter()
    rs = RollingSigStack()
    acc = 0
    for t in toks:
        rs.push(t)
        acc ^= rs.stack_list()[-1] if len(rs) >= 2 else 0
    t_roll = time.perf_counter() - t0

    t0 = time.perf_counter()
    sam = SuffixAutomaton()
    m_acc = 0
    for t in toks:
        m_acc += sam.extend(t)
    t_sam = time.perf_counter() - t0

    # store lookup 비용 (G4 사전 신호 — Python 참조 구현 기준치)
    store = LedgerStore(StoreParams(version=1))
    scope = Scope("t", "r", "s")
    rs2 = RollingSigStack()
    for i in range(0, min(len(toks) - 1, 20_000)):
        ev = VerifyOutcome(
            scope=scope,
            ctx_tail=tuple(toks[max(0, i - 8) : i]),
            draft_ids=(),
            accepted_len=0,
            bonus_id=toks[i],
            topk_ids=((toks[i], toks[(i + 1) % len(toks)]),),
            topk_logp_q8=((3, 40),),
            seg=(3,),
        )
        store.harvest([ev])
    store.drain()
    scope_stack = [sid for _, sid in scope.scope_stack()]
    rs2.push_many(toks[:8])
    t0 = time.perf_counter()
    hits = 0
    for i in range(8, 8 + lookups):
        p = store.lookup(rs2.stack_list(), scope_stack, 3)
        hits += 1 if p else 0
        rs2.push(toks[i % (len(toks) - 1)])
    t_lookup = time.perf_counter() - t0

    res = {
        "n_tokens": n_tokens,
        "rolling_us_per_token": round(t_roll / n_tokens * 1e6, 3),
        "sam_us_per_token": round(t_sam / n_tokens * 1e6, 3),
        "sam_states": sam.n_states(),
        "sam_approx_bytes": sam.approx_bytes(),
        "rolling_bytes": 8 * 8,  # ring 8 tokens 고정 — O(1)
        "store_lookup_us": round(t_lookup / lookups * 1e6, 3),
        "store_entries": store.stats().entries,
        "speedup_rolling_vs_sam": round(t_sam / t_roll, 2) if t_roll else 0.0,
        "checksum": acc & 0xFFFF ^ (m_acc & 0xFFFF),
    }
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=50_000)
    ap.add_argument("--out", default="results/d3_bench")
    args = ap.parse_args(argv)
    res = bench(n_tokens=args.tokens)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "bench.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, sort_keys=True, indent=1)
    print(json.dumps(res, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
