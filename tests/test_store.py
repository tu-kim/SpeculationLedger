"""core.LedgerStore 계약(§3.1) 단위 테스트."""

from core.signature import RollingSigStack
from core.store import LedgerStore, StoreParams
from core.types import Scope, Segment, VerifyOutcome

SCOPE = Scope("t0", "t0/r0", "t0/r0/s0")
STACK = [sid for _, sid in SCOPE.scope_stack()]


def _ev(ctx, draft=(), acc=0, bonus=0, topk=None, seg=None, file_id=0):
    n = acc + 1
    return VerifyOutcome(
        scope=SCOPE,
        ctx_tail=tuple(ctx),
        draft_ids=tuple(draft),
        accepted_len=acc,
        bonus_id=bonus,
        topk_ids=tuple(topk or [() for _ in range(n)]),
        topk_logp_q8=tuple(tuple(3 for _ in row) for row in (topk or [() for _ in range(n)])),
        seg=tuple(seg or [int(Segment.TEXT)] * n),
        file_id=file_id,
    )


def _feed_stream(store, toks, seg=Segment.TEXT, file_id=0):
    for i in range(1, len(toks)):
        store.harvest([_ev(toks[max(0, i - 8) : i], bonus=toks[i],
                           seg=[int(seg)], file_id=file_id)])
    store.drain()


def _lookup(store, ctx, seg=Segment.TEXT):
    rs = RollingSigStack()
    rs.push_many(ctx)
    return store.lookup(rs.stack_list(), STACK, seg)


def test_harvest_then_lookup_roundtrip():
    store = LedgerStore(StoreParams())
    toks = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    _feed_stream(store, toks)
    _feed_stream(store, toks)
    post = _lookup(store, toks[:5])
    assert post is not None
    assert post.argmax().tok == toks[5]


def test_rejection_recorded_and_correction_realized():
    store = LedgerStore(StoreParams())
    ctx = [1, 2, 3, 4, 5, 6, 7, 8]
    # draft [100] 기각, correction(bonus)=200
    store.harvest([_ev(ctx, draft=(100,), acc=0, bonus=200)])
    store.drain()
    post = _lookup(store, ctx)
    toks = {c.tok: c for c in post.cands}
    assert 200 in toks  # correction이 realized로 학습됨
    assert toks[200].support >= 1


def test_queue_bounded_drop_oldest():
    store = LedgerStore(StoreParams(queue_cap=4))
    evs = [_ev([1, 2, 3], bonus=i) for i in range(10)]
    store.harvest(evs)
    st = store.stats()
    assert st.queue_depth == 4
    assert st.drops == 6
    assert store.drain() == 4


def test_epoch_bump_invalidates_code_entries_lazily():
    store = LedgerStore(StoreParams())
    toks = list(range(50, 62))
    _feed_stream(store, toks, seg=Segment.CODE, file_id=77)
    post = _lookup(store, toks[:8], seg=Segment.CODE)
    assert post is not None
    before = store.stats().entries

    for sid in STACK:
        store.bump_epoch(sid, 77)
    # lazy: 엔트리는 남아 있으나 lookup에서 걸러짐
    assert store.stats().entries == before
    assert _lookup(store, toks[:8], seg=Segment.CODE) is None
    assert store.stats().stale_skips > 0

    # compaction에서 실제 폐기 (§3.1)
    store.compact()
    assert store.stats().entries < before


def test_epoch_bump_is_scoped_to_file():
    store = LedgerStore(StoreParams())
    toks = list(range(90, 102))
    _feed_stream(store, toks, seg=Segment.CODE, file_id=77)
    for sid in STACK:
        store.bump_epoch(sid, 888)  # 다른 파일
    assert _lookup(store, toks[:8], seg=Segment.CODE) is not None


def test_snapshot_load_roundtrip(tmp_path):
    store = LedgerStore(StoreParams(version=2))
    toks = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    _feed_stream(store, toks)
    _feed_stream(store, toks)
    p1 = _lookup(store, toks[:6])
    path = str(tmp_path / "snap.json.gz")
    store.snapshot(path)

    fresh = LedgerStore(StoreParams(version=2))
    fresh.load(path)
    p2 = _lookup(fresh, toks[:6])
    assert p1 is not None and p2 is not None
    assert [c.tok for c in p1.cands] == [c.tok for c in p2.cands]


def test_max_entries_fifo_eviction():
    store = LedgerStore(StoreParams(max_entries=64))
    for base in range(0, 400, 10):
        _feed_stream(store, list(range(base, base + 10)))
    st = store.stats()
    assert st.entries <= 64
    assert st.evictions > 0


def test_adaptive_k_grows_under_low_coverage():
    store = LedgerStore(StoreParams(k_init=2, k_max=16))
    ctx = [1, 2, 3, 4, 5, 6, 7, 8]
    # 같은 키에 매번 다른 topk 8종 → coverage 낮음 → k 확장
    for i in range(6):
        topk = [tuple(1000 + i * 8 + j for j in range(8))]
        store.harvest([_ev(ctx, bonus=500 + i, topk=topk)])
    store.drain()
    e = next(iter(store._hot.values()))
    assert e.k_cap > 2


def test_compaction_decays_counts():
    store = LedgerStore(StoreParams())
    ctx = [1, 2, 3, 4, 5, 6, 7, 8]
    for _ in range(4):
        store.harvest([_ev(ctx, bonus=42)])
    store.drain()
    post = _lookup(store, ctx)
    sup_before = post.cands[0].support
    store.compact()
    post2 = _lookup(store, ctx)
    assert post2.cands[0].support < sup_before
