"""proposer 4종 동작 계약."""

from core.signature import RollingSigStack
from core.types import Scope, Segment, VerifyOutcome
from sim.proposers import (
    TR_K,
    TR_TREE_TEMPLATE,
    LedgerProposer,
    ProposeCtx,
    TokenRecyclingProposer,
    VanillaProposer,
    _tr_tree_wiring,
    make_proposer,
)

SCOPE = Scope("t", "t/r", "t/r/s")
STACK = [sid for _, sid in SCOPE.scope_stack()]


def _ctx(toks, budget=8, seg=Segment.TEXT, recent=None):
    rs = RollingSigStack()
    rs.push_many(toks)
    return ProposeCtx(
        sigs=rs,
        scope_stack=STACK,
        seg=seg,
        budget=budget,
        recent=tuple(recent if recent is not None else toks),
        pos=len(toks),
    )


def _vanilla_ev(ctx_tail, tok, topk=None, seg=Segment.TEXT):
    return VerifyOutcome(
        scope=SCOPE,
        ctx_tail=tuple(ctx_tail),
        draft_ids=(),
        accepted_len=0,
        bonus_id=tok,
        topk_ids=(tuple(topk) if topk else (),),
        topk_logp_q8=(tuple(3 for _ in (topk or ())),),
        seg=(int(seg),),
    )


def test_vanilla_never_drafts():
    v = VanillaProposer()
    assert len(v.propose(_ctx([1, 2, 3]))) == 0


def test_ledger_learns_and_extends_chain():
    prop = LedgerProposer()
    stream = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    for _ in range(3):
        for i in range(4, len(stream)):
            prop.harvest(_vanilla_ev(stream[max(0, i - 8) : i], stream[i]))
    tree = prop.propose(_ctx(stream[:6]))
    chain = tree.chain()
    assert chain[:4] == stream[6:10], f"학습한 연속열을 제안해야 함: {chain}"


def test_positive_mode_strips_outcome_info():
    prop = LedgerProposer(value_mode="positive")
    ev = VerifyOutcome(
        scope=SCOPE,
        ctx_tail=(1, 2, 3, 4, 5, 6, 7, 8),
        draft_ids=(100, 101),
        accepted_len=1,
        bonus_id=200,
        topk_ids=((100, 9), (200, 9)),
        topk_logp_q8=((3, 40), (3, 40)),
        seg=(int(Segment.TEXT), int(Segment.TEXT)),
    )
    prop.harvest(ev)
    # 저장된 어떤 엔트리에도 rej 카운트·topk(q8) 관측이 없어야 한다
    for e in prop.store._hot.values():
        for tok, (a, r, q) in e.cands.items():
            assert r == 0, "positive-only인데 rej 기록됨"
            assert q == -1, "positive-only인데 p̂ 기록됨"


def test_tr_template_shape_matches_official():
    assert len(TR_TREE_TEMPLATE) == 80
    depths = {}
    for path in TR_TREE_TEMPLATE:
        depths[len(path)] = depths.get(len(path), 0) + 1
    assert depths == {1: 8, 2: 21, 3: 25, 4: 15, 5: 8, 6: 3}
    wiring = _tr_tree_wiring(TR_TREE_TEMPLATE)
    assert all(p < i for i, (p, _) in enumerate(wiring) if p >= 0)


def test_tr_proposes_from_adjacency_and_cold_rows_are_zero():
    tr = TokenRecyclingProposer()
    tr.begin_request(SCOPE)
    tree = tr.propose(_ctx([5], recent=[5]))
    assert len(tree) == 80
    assert all(n.tok == 0 for n in tree.nodes), "cold row는 0 flood (원 구현 충실)"

    # M 갱신: prev → topk row
    tr.harvest(_vanilla_ev([5], 42, topk=[42, 43, 44, 45, 46, 47, 48, 49]))
    tree2 = tr.propose(_ctx([5], recent=[5]))
    roots = [n.tok for n in tree2.nodes[:TR_K]]
    assert roots == [42, 43, 44, 45, 46, 47, 48, 49]


def test_tr_matrix_is_per_tenant():
    tr = TokenRecyclingProposer()
    tr.begin_request(SCOPE)
    tr.harvest(_vanilla_ev([5], 42, topk=[42] * 8))
    other = Scope("other-tenant", "o/r", "o/r/s")
    tr.begin_request(other)
    tree = tr.propose(_ctx([5], recent=[5]))
    assert all(n.tok == 0 for n in tree.nodes), "tenant 간 M 공유 금지 (§10)"


def test_make_proposer_registry():
    for kind in ("vanilla", "ledger", "positive", "recycle", "dense"):
        p = make_proposer({"kind": kind})
        assert p.name
