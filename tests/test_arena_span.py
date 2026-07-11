"""V2 span arena·break·patch (core.arena + store V2 경로)."""

from core.arena import SpanArena
from core.signature import RollingSigStack
from core.store import LedgerStore, StoreParams
from core.types import Scope, Segment, VerifyOutcome

SCOPE = Scope("t", "t/r", "t/r/s")
STACK = [sid for _, sid in SCOPE.scope_stack()]
SEG = Segment.CODE
CTX = [900, 901, 902, 903, 904, 905, 906, 907]
SPAN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_arena_dedup_and_compact():
    a = SpanArena()
    off1, n1 = a.add([5, 6, 7])
    off2, _ = a.add([5, 6, 7])
    off3, _ = a.add([8, 9])
    assert off1 == off2  # content-hash dedup
    assert a.get(off3, 2) == (8, 9)
    assert a.n_tokens() == 5


def _feed_span(store: LedgerStore, times: int = 2):
    """CTX 뒤에 SPAN이 이어지는 스트림을 times회 관측시킨다."""
    for _ in range(times):
        toks = CTX + SPAN
        for i in range(len(CTX), len(toks)):
            store.harvest(
                [
                    VerifyOutcome(
                        scope=SCOPE,
                        ctx_tail=tuple(toks[max(0, i - 8) : i]),
                        draft_ids=(),
                        accepted_len=0,
                        bonus_id=toks[i],
                        topk_ids=((),),
                        topk_logp_q8=((),),
                        seg=(int(SEG),),
                    )
                ]
            )
        store.drain()
        store.flush_runs()


def test_span_registered_and_looked_up():
    store = LedgerStore(StoreParams(version=2, span_min_len=6))
    _feed_span(store, times=2)
    rs = RollingSigStack()
    rs.push_many(CTX)
    sp = store.lookup_span(rs.stack_list(), STACK, SEG)
    assert sp is not None
    assert list(sp.tokens[:6]) == SPAN[:6]
    assert sp.count >= 2


def test_break_recorded_on_span_prefix_rejection():
    store = LedgerStore(StoreParams(version=2, span_min_len=6))
    _feed_span(store, times=2)
    # span 프리픽스를 draft로 냈는데 offset 3에서 기각된 verify outcome
    store.harvest(
        [
            VerifyOutcome(
                scope=SCOPE,
                ctx_tail=tuple(CTX),
                draft_ids=tuple(SPAN[:6]),
                accepted_len=3,
                bonus_id=777,  # correction
                topk_ids=tuple((SPAN[i],) for i in range(3)) + ((777,),),
                topk_logp_q8=tuple((3,) for _ in range(4)),
                seg=tuple(int(SEG) for _ in range(4)),
            )
        ]
    )
    store.drain()
    rs = RollingSigStack()
    rs.push_many(CTX)
    sp = store.lookup_span(rs.stack_list(), STACK, SEG)
    assert sp is not None
    assert any(off == 3 and cnt >= 1 for off, cnt in sp.breaks)


def test_span_compaction_keeps_live_spans():
    store = LedgerStore(StoreParams(version=2, span_min_len=6))
    _feed_span(store, times=3)
    st0 = store.stats()
    assert st0.span_entries > 0
    store.compact()
    rs = RollingSigStack()
    rs.push_many(CTX)
    sp = store.lookup_span(rs.stack_list(), STACK, SEG)
    assert sp is not None  # count 3>>1=1 → 생존
    assert store.stats().arena_tokens <= st0.arena_tokens
