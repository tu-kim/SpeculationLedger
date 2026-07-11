"""I3 scope 격리: cross-scope lookup이 API 수준에서 불가능 (CLAUDE.md §9)."""

from core.signature import RollingSigStack
from core.store import LedgerStore, StoreParams
from core.types import Scope, Segment, VerifyOutcome

TOKS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]


def _feed(store: LedgerStore, scope: Scope):
    for i in range(1, len(TOKS)):
        store.harvest(
            [
                VerifyOutcome(
                    scope=scope,
                    ctx_tail=tuple(TOKS[max(0, i - 8) : i]),
                    draft_ids=(),
                    accepted_len=0,
                    bonus_id=TOKS[i],
                    topk_ids=((),),
                    topk_logp_q8=((),),
                    seg=(int(Segment.TEXT),),
                )
            ]
        )
    store.drain()


def _lookup(store: LedgerStore, scope: Scope):
    rs = RollingSigStack()
    rs.push_many(TOKS[:6])
    return store.lookup(rs.stack_list(), [sid for _, sid in scope.scope_stack()], Segment.TEXT)


def test_session_isolation_within_repo():
    """같은 repo의 다른 세션: repo tier 공유는 허용, session tier는 격리."""
    store = LedgerStore(StoreParams(scope_depths=(0,)))  # session tier만 활성
    s_a = Scope("t", "t/r", "t/r/sA")
    s_b = Scope("t", "t/r", "t/r/sB")
    _feed(store, s_a)
    assert _lookup(store, s_a) is not None
    assert _lookup(store, s_b) is None  # cross-session 불가


def test_repo_tier_shares_within_repo_only():
    store = LedgerStore(StoreParams(scope_depths=(1,)))  # repo tier만
    s_a = Scope("t", "t/r1", "t/r1/sA")
    s_b = Scope("t", "t/r1", "t/r1/sB")  # 같은 repo, 다른 세션
    s_c = Scope("t", "t/r2", "t/r2/sC")  # 다른 repo
    _feed(store, s_a)
    assert _lookup(store, s_b) is not None  # repo 공유 (설계 의도)
    assert _lookup(store, s_c) is None  # cross-repo 불가


def test_tenant_isolation_even_at_global_tier():
    """global tier조차 per-tenant다 (§10 tenant 격리 원칙)."""
    store = LedgerStore(StoreParams(scope_depths=(2,)))  # global tier만
    s_a = Scope("tenantA", "tenantA/r", "tenantA/r/s")
    s_b = Scope("tenantB", "tenantB/r", "tenantB/r/s")
    _feed(store, s_a)
    assert _lookup(store, s_a) is not None
    assert _lookup(store, s_b) is None  # cross-tenant 불가


def test_api_requires_scope_stack():
    """lookup은 scope_stack 없이 호출 불가(시그니처 수준) — 전 scope 순회 API 부재 확인."""
    store = LedgerStore(StoreParams())
    assert not hasattr(store, "lookup_all_scopes")
    rs = RollingSigStack()
    rs.push_many(TOKS[:6])
    # scope_stack이 비면 어떤 소스도 조회되지 않는다
    assert store.lookup(rs.stack_list(), [], Segment.TEXT) is None
