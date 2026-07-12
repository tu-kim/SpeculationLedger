"""성능 리그레션 가드 (marker: perf) — CI의 perf job이 실행한다.

CI wall-clock 플레이크를 피하는 3중 설계:
  1. 머신속도 정규화: 순수 파이썬 LCG 스핀으로 'unit'(≈인터프리터 기본 연산 시간)을
     보정하고 모든 시간 예산을 unit 배수로 표현한다 — 러너가 느리면 분모도 함께 느려진다.
  2. 비율 가드: 최적화 구현 vs 같은 프로세스의 비최적화 참조 구현 배율 하한 (기계 무관).
  3. 연산-횟수 가드: 배치 등 구조적 최적화는 시간이 아니라 결정적 카운트로 검증.
측정은 GC off + min-of-repeats(잡음에 강건한 하한 추정)로 안정화한다.

예산 산정 기준 (2026-07-12, M4/py3.12 실측 → ×2.5~3 여유):
  rolling 18u (naive 210u, 11.6×) · harvest 383u/ev · lookup 939u · cache 56×
측정치는 results/perf/perf.json으로 남겨 CI 아티팩트로 추이를 추적한다.
"""

import gc
import json
import os
import platform
import random
import sys
import time

import pytest

from core.signature import MAX_ORDER, RollingSigStack, sig_of
from core.store import LedgerStore, StoreParams
from core.types import Scope, Segment, VerifyOutcome

pytestmark = pytest.mark.perf

U64 = (1 << 64) - 1
_METRICS: dict[str, float] = {}


# ------------------------------------------------------------------ 계측 기반
def _timed_ns(fn, *args) -> int:
    gc.collect()
    gc.disable()
    try:
        t0 = time.perf_counter_ns()
        fn(*args)
        return time.perf_counter_ns() - t0
    finally:
        gc.enable()


def _best_of(n: int, fn, *args) -> int:
    return min(_timed_ns(fn, *args) for _ in range(n))


@pytest.fixture(scope="module")
def unit_ns() -> float:
    """머신속도 단위: 64-bit LCG 1 스텝의 최소 관측 시간."""

    def spin(n: int) -> int:
        x = 1
        for _ in range(n):
            x = (x * 6364136223846793005 + 1442695040888963407) & U64
        return x

    n = 200_000
    u = _best_of(3, spin, n) / n
    _METRICS["unit_ns"] = round(u, 2)
    return u


def _stream(n: int, vocab: int, seed: int) -> list[int]:
    """재발 구조(60% 과거 재방출)를 갖는 결정적 토큰 스트림 — 실제 워크로드 근사."""
    rng = random.Random(seed)
    out = [rng.randrange(16, vocab) for _ in range(16)]
    while len(out) < n:
        if rng.random() < 0.6:
            j = rng.randrange(0, len(out) - 8)
            out.extend(out[j : j + rng.randint(3, 10)])
        else:
            out.append(rng.randrange(16, vocab))
    return out[:n]


SCOPE = Scope("t", "t/r", "t/r/s")
STACK = [sid for _, sid in SCOPE.scope_stack()]


def _vanilla_events(toks: list[int]) -> list[VerifyOutcome]:
    return [
        VerifyOutcome(
            scope=SCOPE,
            ctx_tail=tuple(toks[max(0, i - 8) : i]),
            draft_ids=(),
            accepted_len=0,
            bonus_id=toks[i],
            topk_ids=((toks[i], toks[i - 1]),),
            topk_logp_q8=((3, 40),),
            seg=(int(Segment.TEXT),),
        )
        for i in range(1, len(toks))
    ]


# ------------------------------------------------------- 1. 서명 스택 (증분)
class _NaiveRing:
    """비최적화 참조: push마다 ring에서 전 차수 재계산 (최적화 이전 구현과 동형)."""

    def __init__(self):
        self.ring = [0] * MAX_ORDER
        self.n = 0

    def push(self, t):
        self.ring[self.n % MAX_ORDER] = t
        self.n += 1

    def stack_list(self):
        hi = min(self.n, MAX_ORDER)
        n = self.n
        return [
            sig_of([self.ring[(n - o + i) % MAX_ORDER] for i in range(o)])
            for o in range(2, hi + 1)
        ]


def _drive_stack(cls, toks):
    rs = cls()
    acc = 0
    for t in toks:
        rs.push(t)
        s = rs.stack_list()
        acc ^= s[-1] if s else 0
    return acc


def test_perf_rolling_sig_absolute_and_ratio(unit_ns):
    toks = _stream(20_000, 3000, 5)
    t_inc = _best_of(3, _drive_stack, RollingSigStack, toks) / len(toks)
    t_naive = _best_of(2, _drive_stack, _NaiveRing, toks[:6000]) / 6000

    units = t_inc / unit_ns
    ratio = t_naive / t_inc
    _METRICS["rolling_units_per_token"] = round(units, 1)
    _METRICS["rolling_speedup_vs_naive"] = round(ratio, 1)

    assert units <= 55, f"rolling 회귀: {units:.1f} units/tok (기준 18, 예산 55)"
    assert ratio >= 3.5, f"증분 이점 소실: naive 대비 {ratio:.1f}x (기준 11.6x, 하한 3.5x)"


# --------------------------------------------------- 2. store harvest/lookup
@pytest.fixture(scope="module")
def populated_store():
    toks = _stream(8_000, 3000, 7)
    store = LedgerStore(StoreParams())
    evs = _vanilla_events(toks)
    t = _timed_ns(lambda: (store.harvest(evs), store.drain()))
    _METRICS["harvest_ns_per_event_raw"] = round(t / len(evs), 0)
    return store, toks, t / len(evs)


def test_perf_harvest_budget(unit_ns, populated_store):
    _, _, ns_per_event = populated_store
    units = ns_per_event / unit_ns
    _METRICS["harvest_units_per_event"] = round(units, 1)
    assert units <= 1100, f"harvest 회귀: {units:.1f} units/event (기준 383, 예산 1100)"


def test_perf_lookup_budget(unit_ns, populated_store):
    store, toks, _ = populated_store

    def lookups(n):
        rs = RollingSigStack()
        rs.push_many(toks[:8])
        acc = 0
        for i in range(8, 8 + n):
            p = store.lookup(rs.stack_list(), STACK, Segment.TEXT)
            acc ^= 1 if p else 0
            rs.push(toks[i % (len(toks) - 1)])
        return acc

    t = _best_of(3, lookups, 3000) / 3000
    units = t / unit_ns
    _METRICS["lookup_units"] = round(units, 1)
    assert units <= 2800, f"lookup 회귀: {units:.1f} units (기준 939, 예산 2800)"


# ------------------------------------------------------ 3. sources 캐시 비율
def test_perf_sources_cache_ratio(populated_store):
    store, _, _ = populated_store
    e = next(iter(store._hot.values()))
    for k in range(16):
        e.cands.setdefault(10_000 + k, [1, 0, 3])
    e.invalidate()

    def cached(n):
        for _ in range(n):
            e.sources_tuple()

    def rebuild(n):
        for _ in range(n):
            e.invalidate()
            e.sources_tuple()

    t_c = _best_of(3, cached, 20_000) / 20_000
    t_r = _best_of(3, rebuild, 20_000) / 20_000
    ratio = t_r / t_c
    _METRICS["sources_cache_speedup"] = round(ratio, 1)
    assert ratio >= 8.0, f"sources 캐시 무력화: rebuild 대비 {ratio:.1f}x (기준 56x, 하한 8x)"


# ------------------------------------------- 4. faiss 배치 (연산-횟수, 무플레이크)
def test_perf_dense_add_batching_opcount():
    from sim.proposers import make_proposer

    d = make_proposer({"kind": "dense"})
    d.begin_request(SCOPE)
    toks = _stream(1_000, 3000, 9)
    for ev in _vanilla_events(toks):
        d.harvest(ev)
    idx = d._by_repo[SCOPE.repo_id()]
    n_keys = len(idx.offsets)
    # 배치 계약: 검색 없이는 256개 단위로만 flush — ntotal은 256의 배수,
    # 잔여는 버퍼에. (per-add flush로 회귀하면 ntotal == n_keys가 되어 실패)
    assert idx.index.ntotal == (n_keys // 256) * 256, (
        f"add 배치 회귀: ntotal={idx.index.ntotal}, keys={n_keys}"
    )
    assert len(idx._buf) == n_keys % 256
    _METRICS["dense_batch_flushes"] = n_keys // 256


# ------------------------------------------------------------- 5. e2e replay
def test_perf_e2e_ledger_replay_budget(unit_ns, tmp_path):
    from sim.convert import read_traces
    from sim.proposers import make_proposer
    from sim.replay import run_replay
    from sim.synth import SynthParams, generate

    out = str(tmp_path / "perf.jsonl")
    generate(
        SynthParams(seed=3, repos=1, sessions_per_repo=2, turns_per_session=4), out
    )
    reqs = read_traces([out])
    n_tokens = sum(len(r) for r in reqs)
    assert n_tokens > 400  # 워크로드 전제

    def replay_once():
        run_replay(reqs, make_proposer({"kind": "ledger", "store": {"version": 2}}), 8)

    t = _best_of(2, replay_once) / n_tokens
    units = t / unit_ns
    _METRICS["e2e_ledger_units_per_token"] = round(units, 1)
    _METRICS["e2e_tokens"] = n_tokens
    # 소형 trace라 cold-store(엔트리 신규 생성) 비중이 커서 기준치(≈2500)가 높다 —
    # 예산은 2.4× 여유. 미세 회귀는 위의 세분화 가드가, 이 가드는 통합 경로 폭주를 잡는다.
    assert units <= 6000, f"e2e replay 회귀: {units:.1f} units/tok (기준 2500, 예산 6000)"


# ---------------------------------------------------------- 아티팩트 기록
def teardown_module(module):
    """perf.json 아티팩트 — CI가 업로드해 추이를 추적한다 (판정에는 미사용)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(root, "results", "perf")
    os.makedirs(outdir, exist_ok=True)
    doc = {
        "metrics": dict(sorted(_METRICS.items())),
        "machine": platform.machine(),
        "system": platform.system(),
        "python": sys.version.split()[0],
    }
    with open(os.path.join(outdir, "perf.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
