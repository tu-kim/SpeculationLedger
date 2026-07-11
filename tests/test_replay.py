"""sim.replay: oracle 수락 산식과 replay 메트릭."""

from core.types import DraftTree, Scope
from sim.convert import TraceRequest
from sim.proposers import make_proposer
from sim.replay import oracle_accept, recurrence_stats, run_replay


def _tree(chain, extra_child=None):
    t = DraftTree()
    parent = -1
    for tok in chain:
        parent = t.add(tok, parent)
    if extra_child is not None:
        t.add(extra_child, parent)
    return t


def test_oracle_accept_full_chain():
    truth = [1, 2, 3, 4, 5]
    acc, path, rej = oracle_accept(_tree([2, 3, 4]), truth, 1)
    assert (acc, path, rej) == (3, [2, 3, 4], None)


def test_oracle_accept_partial_with_rejected_candidate():
    truth = [1, 2, 9, 4, 5]
    acc, path, rej = oracle_accept(_tree([2, 3, 4]), truth, 1)
    assert acc == 1
    assert path == [2]
    assert rej == 3  # 다음 자식이 기각 후보


def test_oracle_accept_tree_branch():
    truth = [1, 2, 3]
    t = DraftTree()
    r1 = t.add(9, -1)  # 틀린 가지
    t.add(8, r1)
    r2 = t.add(2, -1)  # 맞는 가지
    t.add(3, r2)
    acc, path, rej = oracle_accept(t, truth, 1)
    assert acc == 2
    assert path == [2, 3]


def test_oracle_accept_empty_tree():
    acc, path, rej = oracle_accept(DraftTree(), [1, 2], 0)
    assert (acc, path, rej) == (0, [], None)


def _mini_request(tokens, rid="r0", ts=0, session="t/r/s"):
    n = len(tokens)
    return TraceRequest(
        request_id=rid,
        ts=ts,
        scope=Scope("t", "t/r", session),
        model="m",
        tokenizer_hash="h",
        tokens=list(tokens),
        seg=[3] * n,
        topk_ids=[(t,) for t in tokens],
        topk_logp_q8=[(3,) for _ in tokens],
    )


def test_vanilla_replay_tau_is_one():
    reqs = [_mini_request([1, 2, 3, 4, 5, 6, 7, 8])]
    res = run_replay(reqs, make_proposer({"kind": "vanilla"}), budget=8)
    assert res.totals()["tau"] == 1.0
    assert res.totals()["steps"] == 8


def test_ledger_replay_gains_on_repeated_stream():
    base = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    reqs = [_mini_request(base, rid=f"r{i}", ts=i) for i in range(6)]
    res = run_replay(reqs, make_proposer({"kind": "ledger"}), budget=8)
    curve = res.learning_curve()
    assert curve[0] == 1.0  # 첫 요청은 콜드
    assert curve[-1] > 1.5  # 반복 학습 후 τ 상승
    assert res.totals()["tau"] > 1.2


def test_recurrence_stats_shape():
    base = [1, 2, 3, 4] * 10
    reqs = [_mini_request(base)]
    st = recurrence_stats(reqs)
    assert st["session"]["positions"] == len(base)
    assert st["session"]["rate_ge"]["4"] > 0.5  # 주기 4 반복 스트림
    assert "repo" in st
