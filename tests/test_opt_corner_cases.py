"""최적화(증분 서명·sources 캐시·q8 LUT·faiss 배치)가 도입한 상태의 코너 케이스.

각 최적화는 값-보존이 계약이다: 기준(비최적화) 산식과의 동치를 경계에서 고정한다.
"""

import math
import random

import pytest

from core.signature import MAX_ORDER, RollingSigStack, sig_of
from core.store import LedgerStore, StoreParams
from core.types import Scope, Segment, VerifyOutcome, q8_to_p
from sim.proposers import make_proposer

SCOPE = Scope("t", "t/r", "t/r/s")
STACK = [sid for _, sid in SCOPE.scope_stack()]


def _ev(ctx, bonus, topk=None, seg=Segment.TEXT, draft=(), acc=0):
    n = acc + 1
    tki = tuple(topk) if topk is not None else tuple(() for _ in range(n))
    return VerifyOutcome(
        scope=SCOPE,
        ctx_tail=tuple(ctx),
        draft_ids=tuple(draft),
        accepted_len=acc,
        bonus_id=bonus,
        topk_ids=tki,
        topk_logp_q8=tuple(tuple(3 for _ in row) for row in tki),
        seg=tuple(int(seg) for _ in range(n)),
    )


# ------------------------------------------------- 증분 RollingSigStack 경계
def test_incremental_sigs_match_reference_through_growth_and_beyond():
    """성장 구간(n<MAX_ORDER)과 정상 구간 전체에서 비증분 기준과 동치."""
    rng = random.Random(9)
    toks = [rng.randrange(0, 500) for _ in range(3 * MAX_ORDER)]
    rs = RollingSigStack()
    for i, t in enumerate(toks):
        rs.push(t)
        lst = rs.stack_list()
        hi = min(i + 1, MAX_ORDER)
        assert len(lst) == max(0, hi - 1)
        for j, sig in enumerate(lst):
            order = 2 + j
            assert sig == sig_of(toks[i + 1 - order : i + 1]), f"pos {i} order {order}"


def test_clone_is_fully_independent():
    rs = RollingSigStack()
    rs.push_many([1, 2, 3, 4, 5])
    c = rs.clone()
    base = rs.stack_list()
    c.push(99)  # clone 변이가 원본에 새면 안 된다 (chain 확장 시뮬 경로의 핵심 전제)
    assert rs.stack_list() == base
    assert len(c) == len(rs) + 1
    rs.push(42)  # 반대 방향도
    assert c.stack_list()[-1] == sig_of([1, 2, 3, 4, 5, 99][-MAX_ORDER:])


def test_stack_and_stack_list_are_consistent_views():
    rs = RollingSigStack()
    rs.push_many([7, 8, 9, 10])
    pairs = rs.stack()  # 고차수 우선
    lst = rs.stack_list()  # 저차수 우선
    assert [s for _, s in reversed(pairs)] == lst
    assert [o for o, _ in pairs] == sorted((o for o, _ in pairs), reverse=True)


def test_push_many_equals_repeated_push():
    a, b = RollingSigStack(), RollingSigStack()
    toks = [5, 1, 5, 1, 5]
    a.push_many(toks)
    for t in toks:
        b.push(t)
    assert a.stack_list() == b.stack_list()
    assert a.stack() == b.stack()


# ---------------------------------------------------- sources 캐시 무효화 경계
def _lookup(store, ctx, seg=Segment.TEXT):
    rs = RollingSigStack()
    rs.push_many(ctx)
    return store.lookup(rs.stack_list(), STACK, seg)


def test_cache_lookup_harvest_lookup_sees_update():
    """lookup(캐시 형성) → 같은 키 harvest → lookup이 갱신을 봐야 한다."""
    store = LedgerStore(StoreParams())
    ctx = [1, 2, 3, 4, 5, 6, 7, 8]
    store.harvest([_ev(ctx, bonus=42)])
    store.drain()
    p1 = _lookup(store, ctx)
    assert p1.cands[0].support == 1

    store.harvest([_ev(ctx, bonus=42)])
    store.drain()
    p2 = _lookup(store, ctx)
    assert p2.cands[0].support == 2, "sources 캐시가 stale — 무효화 누락"


def test_cache_invalidated_by_compaction():
    store = LedgerStore(StoreParams())
    ctx = [1, 2, 3, 4, 5, 6, 7, 8]
    for _ in range(4):
        store.harvest([_ev(ctx, bonus=42)])
    store.drain()
    p1 = _lookup(store, ctx)  # 캐시 형성
    assert p1.cands[0].support == 4
    store.compact()  # cands 직접 변이 경로
    p2 = _lookup(store, ctx)
    assert p2.cands[0].support == 2, "compaction 후 캐시 stale"


def test_cache_invalidated_by_rejection_and_topk_merge():
    store = LedgerStore(StoreParams())
    ctx = [1, 2, 3, 4, 5, 6, 7, 8]
    store.harvest([_ev(ctx, bonus=42)])
    store.drain()
    _ = _lookup(store, ctx)  # 캐시 형성
    # rejection + topk 병합이 모두 반영돼야 한다
    store.harvest([_ev(ctx, bonus=99, draft=(42,), acc=0, topk=[(99, 42)])])
    store.drain()
    p = _lookup(store, ctx)
    toks = {c.tok: c for c in p.cands}
    assert toks[42].support == 2  # acc1 + rej1
    assert toks[99].support == 1


def test_cache_reset_on_epoch_stale_rewrite():
    store = LedgerStore(StoreParams())
    ctx = [1, 2, 3, 4, 5, 6, 7, 8]
    store.harvest([_ev(ctx, bonus=42, seg=Segment.CODE)])
    store.drain()
    # 이 lookup은 file 결속이 필요 없는 조회 — 캐시만 형성
    assert _lookup(store, ctx, Segment.CODE) is not None
    # 참고: file_id=0 이벤트라 dom=0 — bump 무관. dom 경로는 corner 스위트가 커버.
    store.harvest([_ev(ctx, bonus=43, seg=Segment.CODE)])
    store.drain()
    p = _lookup(store, ctx, Segment.CODE)
    assert {c.tok for c in p.cands} >= {42, 43}


# --------------------------------------------------------------- q8 LUT 경계
def test_q8_lut_matches_exp_at_boundaries():
    for q in (0, 1, 127, 254, 255):
        assert q8_to_p(q) == math.exp(-q / 16.0)
    # 범위 밖은 fallback 경로 — LUT 인덱스 에러 금지
    assert q8_to_p(-1) == pytest.approx(math.exp(1 / 16.0))
    assert q8_to_p(300) == pytest.approx(math.exp(-300 / 16.0))


# ----------------------------------------------------------- faiss 배치 경계
def test_dense_search_sees_unflushed_buffer_immediately():
    """배치 임계(256) 미만의 add 직후 검색이 그 키를 반드시 봐야 한다."""
    d = make_proposer({"kind": "dense", "max_dist": 4.0})
    d.begin_request(SCOPE)
    stream = [100 + (i % 7) for i in range(24)]  # 주기 7 반복 → 재발 문맥
    for i in range(1, len(stream)):
        d.harvest(_ev(stream[max(0, i - 16) : i], bonus=stream[i]))
    idx = d._by_repo[SCOPE.repo_id()]
    assert len(idx._buf) > 0, "전제: 아직 flush 전 버퍼가 남아 있어야 함"
    assert d.stats()["keys"] == len(idx.offsets)  # stats도 버퍼 포함

    from core.signature import RollingSigStack as RS
    from sim.proposers import ProposeCtx

    rs = RS()
    rs.push_many(stream)
    tree = d.propose(
        ProposeCtx(sigs=rs, scope_stack=STACK, seg=Segment.TEXT, budget=4,
                   recent=tuple(stream), pos=len(stream))
    )
    assert len(tree) > 0, "flush-on-search 누락 — 버퍼 키가 검색에 안 보임"
    assert len(idx._buf) == 0  # 검색이 flush를 수행


def test_dense_offsets_alignment_with_batched_adds():
    """배치 add에서도 faiss id ↔ offsets 정렬이 유지돼야 한다 (id 어긋나면 엉뚱한 연속열 제안)."""
    d = make_proposer({"kind": "dense", "max_dist": 0.05})
    d.begin_request(SCOPE)
    rng = random.Random(4)
    stream = [rng.randrange(16, 4000) for _ in range(300)]
    for i in range(1, len(stream)):
        d.harvest(_ev(stream[max(0, i - 16) : i], bonus=stream[i]))
    idx = d._by_repo[SCOPE.repo_id()]
    idx._flush()
    assert idx.index.ntotal == len(idx.offsets)
    # 정확히 같은 컨텍스트로 검색하면 dist≈0으로 자기 위치가 나와야 하고,
    # 그 offset의 연속열은 실제 스트림과 일치해야 한다
    probe_end = 200
    window = tuple(stream[probe_end - 16 : probe_end + 1])
    vec = d._embed(window)
    dist, fid = idx.search(vec)
    assert dist == pytest.approx(0.0, abs=1e-6)
    off = idx.offsets[fid]
    assert idx.stream[off] == stream[probe_end]
