"""코너 케이스 스위트: 경계·포화·오입력·퇴화 구성에서의 계약 유지.

각 테스트는 '원하는 방어적 동작'을 명세한다 — 크래시 없는 실패(graceful miss),
카운터 정합, 결정성. 발견된 버그의 수정 근거가 이 파일이다.
"""

import json
import random

import pytest

from core.backoff import BackoffParams, Source, blend
from core.signature import RollingSigStack, fold_key, sig_of
from core.store import LedgerStore, StoreParams
from core.types import (
    DraftTree,
    Scope,
    Segment,
    VerifyOutcome,
    logp_to_q8,
    q8_to_logp,
)
from sim.convert import TraceRequest, convert_token_log, normalize_record, validate_record
from sim.proposers import LedgerProposer, ProposeCtx, TokenRecyclingProposer, make_proposer
from sim.replay import oracle_accept, recurrence_stats, run_replay

SCOPE = Scope("t", "t/r", "t/r/s")
STACK = [sid for _, sid in SCOPE.scope_stack()]


def _ev(ctx, draft=(), acc=0, bonus=0, topk=None, seg=None, file_id=0, scope=SCOPE):
    n = acc + 1
    tki = tuple(topk) if topk is not None else tuple(() for _ in range(n))
    return VerifyOutcome(
        scope=scope,
        ctx_tail=tuple(ctx),
        draft_ids=tuple(draft),
        accepted_len=acc,
        bonus_id=bonus,
        topk_ids=tki,
        topk_logp_q8=tuple(tuple(3 for _ in row) for row in tki),
        seg=tuple(seg or [int(Segment.TEXT)] * n),
        file_id=file_id,
    )


# ---------------------------------------------------------------- types 경계
def test_q8_codec_handles_nonfinite_and_extremes():
    assert logp_to_q8(0.0) == 0
    assert logp_to_q8(-1e9) == 255
    assert logp_to_q8(float("-inf")) == 255  # 온라인 harvester: -inf logprob 가능
    assert logp_to_q8(float("inf")) == 0
    assert logp_to_q8(float("nan")) == 255  # 미지 확률 = 최저 신뢰로 강등
    assert q8_to_logp(0) == 0.0
    assert q8_to_logp(255) == pytest.approx(-15.9375)


def test_scope_field_namespaces_are_separated():
    # 필드 값이 같아도 tier id는 달라야 한다 (prefix 네임스페이스)
    s = Scope("x", "x", "x")
    assert len({s.session_id(), s.repo_id(), s.global_id()}) == 3


def test_scope_empty_and_unicode_strings():
    s1 = Scope("", "", "")
    s2 = Scope("한글🙂", "한글🙂/레포", "세션")
    assert s1.session_id() != s2.session_id()
    assert s1.scope_stack()[0][1] == Scope("", "", "").session_id()  # 결정적


def test_draft_tree_chain_ignores_branches():
    t = DraftTree()
    a = t.add(1, -1)
    t.add(9, -1)  # 형제 가지 — chain에는 미포함
    t.add(2, a)
    assert t.chain() == [1, 2]


# ------------------------------------------------------------- signature 경계
def test_sig_stack_below_min_order_is_empty():
    rs = RollingSigStack()
    assert rs.stack_list() == []
    rs.push(5)
    assert rs.stack_list() == []  # 1 < MIN_ORDER
    rs.push(6)
    assert len(rs.stack_list()) == 1


def test_sig_of_negative_and_huge_tokens_deterministic():
    a = sig_of([-1, 2**63, 0])
    b = sig_of([-1, 2**63, 0])
    assert a == b
    assert 0 <= a < 2**64


def test_fold_key_masks_seg_to_2bits():
    # hdr의 seg는 2-bit 필드(§3.1) — 범위 밖 seg는 마스킹되어 키 공간을 벗어나지 않는다
    sig = sig_of([1, 2, 3])
    assert fold_key(sig, 3, 111, 7) == fold_key(sig, 3, 111, 3)
    assert fold_key(sig, 3, 111, 4) == fold_key(sig, 3, 111, 0)


# --------------------------------------------------------------- backoff 경계
def test_blend_all_zero_count_sources_is_none():
    src = Source(match_len=5, scope_depth=0, cands=())
    assert blend(BackoffParams(), [src, src]) is None


def test_blend_single_candidate_probabilities_bounded():
    src = Source(match_len=8, scope_depth=0, cands=((7, 3, 1, 5),))
    post = blend(BackoffParams(), [src])
    c = post.cands[0]
    assert 0.0 <= c.p_acc <= 1.0
    assert 0.0 <= c.p_hat <= 1.0 + 1e-9


def test_blend_saturated_u16_counts_no_overflow():
    src = Source(match_len=8, scope_depth=0, cands=((7, 65535, 65535, 0),))
    post = blend(BackoffParams(), [src])
    assert post is not None
    assert 0.0 <= post.cands[0].p_acc <= 1.0


# ----------------------------------------------------------------- store 경계
def test_harvest_with_zero_queue_cap_drops_everything():
    store = LedgerStore(StoreParams(queue_cap=0))
    store.harvest([_ev([1, 2, 3], bonus=9)] * 5)  # 크래시 금지
    st = store.stats()
    assert st.queue_depth == 0
    assert st.drops == 5
    assert store.drain() == 0


def test_lookup_with_oversized_sig_stack_is_graceful():
    store = LedgerStore(StoreParams(version=2))
    huge = [sig_of([i, i + 1]) for i in range(20)]  # MAX_ORDER 초과분 포함
    assert store.lookup(huge, STACK, Segment.TEXT) is None
    assert store.lookup_span(huge, STACK, Segment.TEXT) is None  # IndexError 금지


def test_store_params_orders_out_of_range_are_ignored():
    store = LedgerStore(StoreParams(orders=(2, 9, 15)))  # 9·15는 무효 차수
    store.harvest([_ev([1, 2, 3, 4], bonus=42)])
    store.drain()  # 크래시 금지
    rs = RollingSigStack()
    rs.push_many([1, 2, 3, 4])
    post = store.lookup(rs.stack_list(), STACK, Segment.TEXT)
    assert post is not None and post.best_order == 2  # 유효 차수만 동작


def test_harvest_out_of_range_seg_does_not_crash():
    store = LedgerStore(StoreParams())
    store.harvest([_ev([1, 2, 3], bonus=9, seg=[7])])
    assert store.drain() == 1


def test_harvest_empty_ctx_tail_first_token():
    store = LedgerStore(StoreParams())
    store.harvest([_ev([], bonus=5)])
    assert store.drain() == 1
    assert store.stats().entries == 0  # 서명 불가 → 엔트리 없음, 크래시도 없음


def test_same_token_rejected_and_realized():
    store = LedgerStore(StoreParams())
    ctx = [1, 2, 3, 4, 5, 6, 7, 8]
    store.harvest([_ev(ctx, draft=(42,), acc=0, bonus=42)])  # 42 기각 후 42가 correction
    store.drain()
    rs = RollingSigStack()
    rs.push_many(ctx)
    post = store.lookup(rs.stack_list(), STACK, Segment.TEXT)
    c = post.cands[0]
    assert c.tok == 42
    assert c.support == 2  # acc 1 + rej 1


def test_malformed_accepted_len_beyond_draft_is_clamped():
    store = LedgerStore(StoreParams())
    ev = _ev([1, 2, 3], draft=(9, 8), acc=0, bonus=7)
    ev = VerifyOutcome(**{**ev.__dict__, "accepted_len": 5})  # 오염된 이벤트
    store.harvest([ev])
    assert store.drain() == 1  # 크래시 금지


def test_epoch_u16_wraparound_pins_known_limitation():
    store = LedgerStore(StoreParams())
    sid = STACK[0]
    for _ in range(65536):
        store.bump_epoch(sid, 7)
    assert store._epoch_now(sid, 7) == 0  # u16 랩 — 65536회 주기로 재앨리어싱
    # 실사용에서는 compaction이 stale 엔트리를 그 전에 폐기한다 (§3.1 lazy)


def test_max_entries_one_thrashes_but_stays_bounded():
    store = LedgerStore(StoreParams(max_entries=1))
    for base in range(0, 100, 10):
        toks = list(range(base, base + 10))
        for i in range(2, len(toks)):
            store.harvest([_ev(toks[:i], bonus=toks[i])])
        store.drain()
    st = store.stats()
    assert st.entries <= 1
    assert st.evictions > 0


def test_aggressive_compaction_every_event():
    store = LedgerStore(StoreParams(version=2, compact_every=1))
    toks = list(range(30, 60))
    for _ in range(3):
        for i in range(2, len(toks)):
            store.harvest([_ev(toks[:i], bonus=toks[i], seg=[int(Segment.CODE)])])
        store.drain()
        store.flush_runs()
    st = store.stats()
    assert st.compactions > 0
    assert st.entries >= 0  # 살아남는 것이 목적이 아니라 크래시·불변식 확인


def test_snapshot_empty_store_roundtrip(tmp_path):
    store = LedgerStore(StoreParams(version=2))
    p = str(tmp_path / "empty.json.gz")
    store.snapshot(p)
    fresh = LedgerStore(StoreParams(version=2))
    fresh.load(p)
    assert fresh.stats().entries == 0


def test_snapshot_load_then_continue_equals_uninterrupted(tmp_path):
    toks = list(range(200, 230))

    def feed(store, lo, hi):
        for i in range(max(2, lo), hi):
            store.harvest([_ev(toks[max(0, i - 8) : i], bonus=toks[i])])
        store.drain()

    cont = LedgerStore(StoreParams(version=2))
    feed(cont, 2, 29)

    part = LedgerStore(StoreParams(version=2))
    feed(part, 2, 15)
    p = str(tmp_path / "mid.json.gz")
    part.snapshot(p)
    resumed = LedgerStore(StoreParams(version=2))
    resumed.load(p)
    feed(resumed, 15, 29)

    rs = RollingSigStack()
    rs.push_many(toks[:10])
    a = cont.lookup(rs.stack_list(), STACK, Segment.TEXT)
    b = resumed.lookup(rs.stack_list(), STACK, Segment.TEXT)
    assert a is not None and b is not None
    assert [(c.tok, c.support) for c in a.cands] == [(c.tok, c.support) for c in b.cands]


def test_u16_counter_saturation():
    store = LedgerStore(StoreParams())
    ctx = [1, 2, 3, 4, 5, 6, 7, 8]
    e = store._get_or_create(fold_key(sig_of(ctx[-2:]), 2, STACK[0], 3), 3, 0, STACK[0])
    e.cands[42] = [65535, 65535, 3]
    e.update_realized(42, 16)
    e.update_rejected(42, 16)
    assert e.cands[42][0] == 65535 and e.cands[42][1] == 65535  # 포화, 오버플로 금지


# ------------------------------------------------------------- convert 경계
def _rec(**over):
    rec = {
        "schema_version": 1,
        "request_id": "r",
        "ts": 0,
        "scope": {"tenant": "t", "repo": "r", "session": "s"},
        "model": "m",
        "tokenizer_hash": "h",
        "steps": [
            {
                "pos": 0,
                "proposed": [],
                "accepted_len": 0,
                "bonus": 1,
                "topk_ids": [[1]],
                "topk_logp_q8": [[0]],
                "seg": [0],
                "t_us": 0,
            }
        ],
        "final_text_sha": "",
    }
    rec.update(over)
    return rec


def test_validator_rejects_negative_accepted_len():
    rec = _rec()
    rec["steps"][0]["accepted_len"] = -1
    assert any("accepted_len" in e for e in validate_record(rec))


def test_validator_rejects_out_of_range_seg():
    rec = _rec()
    rec["steps"][0]["seg"] = [9]
    assert any("seg" in e for e in validate_record(rec))


def test_empty_steps_record_is_valid_and_normalizes_empty():
    rec = _rec(steps=[])
    assert validate_record(rec) == []
    req = normalize_record(rec)
    assert len(req) == 0


def test_token_log_empty_tokens():
    rec = convert_token_log(
        {"request_id": "r", "ts": 0, "scope": {"tenant": "t", "repo": "r", "session": "s"},
         "tokenizer_hash": "h", "tokens": []}
    )
    assert validate_record(rec) == []


# -------------------------------------------------------------- replay 경계
def _mini(tokens, rid="r0", ts=0, events=None, seg=None):
    n = len(tokens)
    return TraceRequest(
        request_id=rid,
        ts=ts,
        scope=SCOPE,
        model="m",
        tokenizer_hash="h",
        tokens=list(tokens),
        seg=list(seg or [3] * n),
        topk_ids=[(t,) for t in tokens],
        topk_logp_q8=[(3,) for _ in tokens],
        events=events or [],
    )


def test_replay_empty_request():
    res = run_replay([_mini([])], make_proposer({"kind": "ledger"}), budget=8)
    assert res.totals()["steps"] == 0
    assert res.totals()["tau"] == 0.0


def test_replay_single_token_request():
    res = run_replay([_mini([42])], make_proposer({"kind": "vanilla"}), budget=8)
    assert res.totals() == {**res.totals(), "steps": 1, "tokens": 1}


def test_replay_event_beyond_length_does_not_crash():
    from sim.convert import TraceEvent

    req = _mini([1, 2, 3], events=[TraceEvent(pos=99, type="file_edit", file="a.py")])
    res = run_replay([req], make_proposer({"kind": "ledger"}), budget=8)
    assert res.totals()["steps"] >= 1


def test_oracle_accept_truth_exhausted_mid_tree():
    truth = [1, 2]
    t = DraftTree()
    p1 = t.add(2, -1)
    t.add(3, p1)  # truth가 끝났는데 트리는 더 깊음
    acc, path, rej = oracle_accept(t, truth, 1)
    assert acc == 1 and path == [2]
    assert rej == 3  # 경계 밖 비교는 기각 후보로만


def test_recurrence_on_empty_and_constant_streams():
    st = recurrence_stats([_mini([])])
    assert st["session"]["positions"] == 0
    st2 = recurrence_stats([_mini([5] * 40)])
    assert st2["session"]["rate_ge"]["8"] > 0.7  # 단일 토큰 반복 = 최대 재발


# ------------------------------------------------------------ proposer 경계
def test_ledger_budget_zero_and_never_exceeded():
    prop = LedgerProposer()
    stream = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    for _ in range(3):
        for i in range(4, len(stream)):
            prop.harvest(_ev(stream[max(0, i - 8) : i], bonus=stream[i]))

    def ctx(budget):
        rs = RollingSigStack()
        rs.push_many(stream[:6])
        return ProposeCtx(sigs=rs, scope_stack=STACK, seg=Segment.TEXT,
                          budget=budget, recent=tuple(stream[:6]), pos=6)

    assert len(prop.propose(ctx(0))) == 0
    for b in (1, 2, 3, 8):
        assert len(prop.propose(ctx(b))) <= b, "splice 포함 budget 초과 금지"


def test_ledger_budget_respected_with_forced_splice():
    """top이 rej-dominant인 상황(빈도 1위 && rej>acc)을 구성해 splice 경로에서도
    budget이 엄수되는지 — 특히 budget=1에서 leaf+patch 2노드 초과가 없는지."""
    prop = LedgerProposer()
    ctx_toks = [1, 2, 3, 4, 5, 6, 7, 8]
    # 도달 가능한 rej-dominant 케이스: target이 X를 계속 지지(topk에 X 상존)하는데
    # 검증은 계속 기각(교정은 Y/Z로 분산) — freq·p̂ 모두 X 우위 유지 + rej>acc.
    # 이때 patch(p̂ argmax)==X이므로 §3.2에 따라 leaf 하나만 남기고 중단해야 한다.
    for _ in range(6):
        prop.harvest(_ev(ctx_toks, bonus=100, topk=[(100,)]))
    for i in range(8):
        y = 300 + (i % 2)
        prop.harvest(_ev(ctx_toks, draft=(100,), acc=0, bonus=y, topk=[(y, 100)]))
    rs = RollingSigStack()
    rs.push_many(ctx_toks)
    post = prop.store.lookup(rs.stack_list(), STACK, Segment.TEXT)
    top = post.cands[0]
    assert top.tok == 100 and top.p_acc < 0.45, f"전제: top이 rej-dominant여야 함, got {top}"
    for b in (1, 2, 3, 8):
        tree = prop.propose(
            ProposeCtx(sigs=rs.clone(), scope_stack=STACK, seg=Segment.TEXT,
                       budget=b, recent=tuple(ctx_toks), pos=8)
        )
        assert len(tree) <= b, f"budget {b} 초과: {len(tree)}"
    tree = prop.propose(
        ProposeCtx(sigs=rs.clone(), scope_stack=STACK, seg=Segment.TEXT,
                   budget=8, recent=tuple(ctx_toks), pos=8)
    )
    assert len(tree) == 1 and tree.nodes[0].tok == 100, "patch==top → leaf만 남기고 budget 0"


def test_ledger_splice_structure_on_conflicting_evidence():
    """white-box: 빈도는 X 우위·p̂는 교정 우위인 '증거 충돌' 엔트리를 직접 주입 —
    splice가 leaf(X)+patch(Y) 형제를 만들고 patch를 통해 체인을 잇는지 검증.

    (end-to-end 동역학에서는 freq×p̂ 랭킹이 self-heal해 이 창이 매우 좁다 —
    단일 안정 교정에서는 도달 불가: acc(correction)=rej(X)>acc(X)가 되어 빈도가
    먼저 역전된다. 그래서 상태를 직접 구성한다.)"""
    from core.signature import fold_key, sig_of

    prop = LedgerProposer()
    ctx_toks = [1, 2, 3, 4, 5, 6, 7, 8]
    key = fold_key(sig_of(ctx_toks), 8, STACK[0], int(Segment.TEXT))
    e = prop.store._get_or_create(key, int(Segment.TEXT), 0, STACK[0])
    e.cands[100] = [5, 7, 18]  # freq 우위, rej>acc, p̂ 중간
    e.cands[300] = [1, 0, 3]  # p̂ 최고 (correction)
    e.cands[301] = [1, 0, 40]

    rs = RollingSigStack()
    rs.push_many(ctx_toks)
    post = prop.store.lookup(rs.stack_list(), STACK, Segment.TEXT)
    top = post.cands[0]
    assert top.tok == 100 and top.p_acc < 0.45, f"전제 불성립: {post.cands[:3]}"

    tree = prop.propose(
        ProposeCtx(sigs=rs.clone(), scope_stack=STACK, seg=Segment.TEXT,
                   budget=8, recent=tuple(ctx_toks), pos=8)
    )
    parents = [n.parent for n in tree.nodes]
    assert parents[:2] == [-1, -1], f"leaf(X)+patch 형제 구조여야 함: {tree.nodes}"
    assert tree.nodes[0].tok == 100 and tree.nodes[1].tok == 300
    assert prop.n_patches == 1
    # budget=1이면 leaf 없이 patch만
    tree1 = prop.propose(
        ProposeCtx(sigs=rs.clone(), scope_stack=STACK, seg=Segment.TEXT,
                   budget=1, recent=tuple(ctx_toks), pos=8)
    )
    assert len(tree1) == 1 and tree1.nodes[0].tok == 300


def test_tr_short_topk_row_padded():
    tr = TokenRecyclingProposer()
    tr.begin_request(SCOPE)
    tr.harvest(_ev([5], bonus=42, topk=[(42, 43)]))
    row = tr._m[5]
    assert len(row) == 8 and row[:2] == [42, 43] and row[2:] == [0] * 6


def test_dense_proposer_tiny_context_and_empty_index():
    d = make_proposer({"kind": "dense"})
    d.begin_request(SCOPE)
    rs = RollingSigStack()
    tree = d.propose(ProposeCtx(sigs=rs, scope_stack=STACK, seg=Segment.TEXT,
                                budget=8, recent=(7,), pos=0))
    assert len(tree) == 0  # window<2 → 제안 없음, 크래시 없음


def test_make_proposer_unknown_kind_raises():
    with pytest.raises(ValueError):
        make_proposer({"kind": "nope"})


@pytest.mark.slow
def test_dense_ivfpq_upgrade_path():
    d = make_proposer({"kind": "dense", "ivfpq": True})
    d.begin_request(SCOPE)
    rng = random.Random(3)
    toks = [rng.randrange(16, 500) for _ in range(4300)]
    for i in range(1, len(toks)):
        d.harvest(_ev(toks[max(0, i - 16) : i], bonus=toks[i]))
    idx = d._by_repo[SCOPE.repo_id()]
    assert idx.trained_ivf, "IVF-PQ 승급이 일어나야 함"
    rs = RollingSigStack()
    rs.push_many(toks[:20])
    d.propose(ProposeCtx(sigs=rs, scope_stack=STACK, seg=Segment.TEXT,
                         budget=8, recent=tuple(toks[:20]), pos=20))  # 크래시 금지


# ------------------------------------------------------------------- 퍼징
def test_fuzz_random_traces_through_all_proposers():
    rng = random.Random(20260711)
    for trial in range(12):
        n_req = rng.randint(1, 4)
        reqs = []
        for r in range(n_req):
            n = rng.randint(0, 40)
            toks = [rng.randrange(16, 300) for _ in range(n)]
            seg = [rng.randrange(0, 4) for _ in range(n)]
            events = []
            if n > 4 and rng.random() < 0.5:
                from sim.convert import TraceEvent

                events.append(TraceEvent(pos=rng.randrange(0, n), type="file_edit",
                                         file=f"f{rng.randrange(3)}.py",
                                         bump_pos=rng.randrange(0, n)))
            reqs.append(_mini(toks, rid=f"t{trial}r{r}", ts=r, events=events, seg=seg))
        for kind in ("ledger", "positive", "recycle", "vanilla"):
            spec = {"kind": kind}
            if kind in ("ledger", "positive"):
                spec["store"] = {"version": rng.choice([1, 2]),
                                 "max_entries": rng.choice([0, 8]),
                                 "queue_cap": rng.choice([1, 4096]),
                                 "compact_every": rng.choice([0, 3])}
            prop = make_proposer(spec)
            res = run_replay(reqs, prop, budget=rng.choice([1, 4, 8]))
            tot = res.totals()
            assert tot["tokens"] == sum(len(r) for r in reqs), (trial, kind)
            st = prop.stats()
            if "hit_rate" in st:
                assert 0.0 <= st["hit_rate"] <= 1.0
            if "bytes" in st:
                assert st["bytes"] >= 0


# ------------------------------------------------------------- gates 경계
def test_gates_zero_record_trace(tmp_path):
    import yaml

    from sim.gates import run

    trace = tmp_path / "empty.jsonl"
    trace.write_text("")
    cfg = {
        "exp_id": "empty_case",
        "gates": ["G1", "G2", "G-R1"],
        "budget": 4,
        "trace": {"provenance": "synthetic", "paths": [str(trace)]},
        "proposers": [
            {"role": "a", "kind": "ledger"},
            {"role": "b", "kind": "positive"},
        ],
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    doc = run(str(cfg_path), str(tmp_path / "out"), plots=False)
    assert doc["trace"]["tokens"] == 0
    assert doc["gates"]["G1"]["pass"] is False  # 증거 없음 = FAIL, 크래시 아님
    gj = json.loads((tmp_path / "out" / "empty_case" / "gates.json").read_text())
    assert gj["exp_id"] == "empty_case"
