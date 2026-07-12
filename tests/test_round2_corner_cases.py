"""2차 최적화(LUT-λ·totals 캐시·인라인 fold) + 미커버 영역(CLI·플롯·스크립트) 코너.

값-보존 최적화의 계약: 기준(직접 계산) 산식과의 동치를 경계에서 고정한다.
"""

import json
import math
import subprocess
import sys

import pytest

from core.backoff import BackoffParams, Source, lam
from core.signature import MAX_ORDER, fold_key, sig_of
from core.store import LedgerStore, StoreParams
from core.types import Scope, Segment, VerifyOutcome

ROOT = __file__.rsplit("/tests/", 1)[0]
SCOPE = Scope("t", "t/r", "t/r/s")
STACK = [sid for _, sid in SCOPE.scope_stack()]


def _ev(ctx, bonus, topk=None, seg=Segment.TEXT, draft=(), acc=0):
    n = acc + 1
    tki = tuple(topk) if topk is not None else tuple(() for _ in range(n))
    return VerifyOutcome(
        scope=SCOPE, ctx_tail=tuple(ctx), draft_ids=tuple(draft), accepted_len=acc,
        bonus_id=bonus, topk_ids=tki,
        topk_logp_q8=tuple(tuple(3 for _ in row) for row in tki),
        seg=tuple(int(seg) for _ in range(n)),
    )


# ------------------------------------------------ λ LUT · Source totals 동치
def test_lam_lut_equals_direct_pow_across_params():
    """_decay_pows LUT는 직접 pow와 완전 동일해야 한다 (임의 파라미터 포함)."""
    for od, sd in ((0.55, 0.35), (0.7, 0.5), (0.3, 0.9), (1.0, 1.0)):
        p = BackoffParams(order_decay=od, scope_decay=sd)
        for m in range(2, MAX_ORDER + 1):
            for d in (0, 1, 2):
                for c in (1, 5, 100):
                    direct = (od ** (MAX_ORDER - m)) * (sd**d) * (c / (c + p.count_prior))
                    assert lam(p, m, d, c) == direct, (od, sd, m, d, c)


def test_source_precomputed_totals_equal_fallback():
    cands = ((7, 3, 2, 10), (9, 1, 0, 20), (11, 0, 4, 30))
    fallback = Source(8, 0, cands)
    pre = Source(8, 0, cands, total=10, total_acc=4)
    assert fallback.total_count() == pre.total_count() == 10
    assert fallback.total_acc_count() == pre.total_acc_count() == 4


def test_sources_view_totals_track_all_mutation_paths():
    """캐시된 (Σacc+rej, Σacc)가 모든 변이 경로 후 재계산치와 일치해야 한다."""
    store = LedgerStore(StoreParams())
    ctx = [1, 2, 3, 4, 5, 6, 7, 8]
    store.harvest([_ev(ctx, bonus=42, topk=[(42, 43)])])
    store.harvest([_ev(ctx, bonus=44, draft=(42,), acc=0, topk=[(44, 42)])])
    store.drain()
    for e in store._hot.values():
        cands, total, total_acc = e.sources_view()
        assert total == sum(a + r for _, a, r, _ in cands)
        assert total_acc == sum(a for _, a, _, _ in cands)
    store.compact()
    for e in store._hot.values():
        cands, total, total_acc = e.sources_view()
        assert total == sum(a + r for _, a, r, _ in cands)
        assert total_acc == sum(a for _, a, _, _ in cands)


def test_inlined_fold_matches_spec_fold_key():
    """store 핫루프의 인라인 전개가 스펙 함수 fold_key와 키 단위로 동일해야 한다 —
    harvest로 만든 엔트리 키가 fold_key로 계산한 키에서 정확히 조회되는지로 검증."""
    store = LedgerStore(StoreParams())
    ctx = [10, 11, 12, 13, 14, 15, 16, 17]
    store.harvest([_ev(ctx, bonus=99, seg=Segment.CODE)])
    store.drain()
    hits = 0
    for order in range(2, MAX_ORDER + 1):
        sig = sig_of(ctx[-order:])
        for sid in STACK:
            key = fold_key(sig, order, sid, int(Segment.CODE))
            if key in store._hot:
                hits += 1
    assert hits == 7 * 3, f"인라인 fold ↔ fold_key 불일치: {hits}/21"


# ---------------------------------------------------------- CLI 진입점 관통
def test_convert_cli_roundtrip(tmp_path):
    from sim.convert import main as convert_main

    src = tmp_path / "raw.jsonl"
    rec = {"request_id": "r", "ts": 0,
           "scope": {"tenant": "t", "repo": "r", "session": "s"},
           "tokenizer_hash": "h", "tokens": [1, 2, 3], "seg": [0, 1, 2]}
    src.write_text(json.dumps(rec) + "\n")
    out = tmp_path / "v1.jsonl"
    assert convert_main(["--input", str(src), "--format", "token_log",
                         "--output", str(out)]) == 0
    from sim.convert import read_traces

    reqs = read_traces([str(out)])
    assert reqs[0].tokens == [1, 2, 3]


def test_synth_cli_writes_valid_trace(tmp_path):
    from sim.convert import read_traces
    from sim.synth import main as synth_main

    out = tmp_path / "s.jsonl"
    assert synth_main(["--out", str(out), "--seed", "42"]) == 0
    reqs = read_traces([str(out)])  # strict 검증 통과 = 생성기 스키마 정합
    assert len(reqs) > 0
    assert (tmp_path / "s.jsonl.meta.json").exists()


def test_gates_cli_end_to_end(tmp_path):
    import yaml

    from sim.gates import main as gates_main

    cfg = {
        "exp_id": "cli_e2e", "gates": ["G1"], "seed": 1, "budget": 4,
        "trace": {"provenance": "synthetic",
                  "paths": [str(tmp_path / "t.jsonl")],
                  "synth": {"seed": 1, "repos": 1, "sessions_per_repo": 1,
                            "turns_per_session": 2}},
        "proposers": [],
    }
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    assert gates_main(["--config", str(cfg_path), "--out", str(tmp_path / "res"),
                       "--no-plots"]) == 0
    doc = json.loads((tmp_path / "res" / "cli_e2e" / "gates.json").read_text())
    assert doc["exp_id"] == "cli_e2e"
    assert (tmp_path / "res" / "cli_e2e" / "c.yaml").exists()  # config 사본 (§8)


# ----------------------------------------------------------- 플롯 엣지 데이터
def test_plots_handle_edge_shapes(tmp_path):
    from analysis.plots import plot_exp

    # proposer 0개 + sweep 0개 → 산출물 없음, 크래시 없음
    assert plot_exp({"exp_id": "empty", "proposers": {}, "size_sweep": []},
                    str(tmp_path)) == []
    # 요청 1개짜리 learning curve + 단일 sweep 점
    doc = {
        "exp_id": "one",
        "proposers": {"a": {"totals": {"per_seg_tau": {"tool": 2.0}},
                            "learning_curve": [1.5]}},
        "size_sweep": [{"bytes": 100, "hit_rate": 0.5, "max_entries": 10}],
    }
    written = plot_exp(doc, str(tmp_path))
    assert len(written) == 3


# ------------------------------------------------------- span 스냅샷 연속성
def test_span_proposal_survives_snapshot_reload(tmp_path):
    """V2 span·break가 snapshot→load 후에도 proposer 경로에서 그대로 동작해야 한다."""
    from core.signature import RollingSigStack

    ctx = [900, 901, 902, 903, 904, 905, 906, 907]
    span = [1, 2, 3, 4, 5, 6, 7, 8]
    store = LedgerStore(StoreParams(version=2, span_min_len=6))
    for _ in range(2):
        toks = ctx + span
        for i in range(len(ctx), len(toks)):
            store.harvest([_ev(toks[max(0, i - 8):i], bonus=toks[i], seg=Segment.CODE)])
        store.drain()
        store.flush_runs()
    path = str(tmp_path / "v2.json.gz")
    store.snapshot(path)

    fresh = LedgerStore(StoreParams(version=2, span_min_len=6))
    fresh.load(path)
    rs = RollingSigStack()
    rs.push_many(ctx)
    sp = fresh.lookup_span(rs.stack_list(), STACK, Segment.CODE)
    assert sp is not None and list(sp.tokens[:6]) == span[:6] and sp.count >= 2


# ----------------------------------------------------------------- 스크립트
def test_shell_script_syntax():
    r = subprocess.run(["bash", "-n", f"{ROOT}/bench/opencode/run_instance.sh"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_require_gpu_blocks_with_exit_2():
    r = subprocess.run([sys.executable, f"{ROOT}/scripts/require_gpu.py", "online-smoke"],
                       capture_output=True, text=True)
    assert r.returncode == 2
    assert "BLOCKED" in r.stdout


def test_normalize_golden_idempotent(tmp_path):
    sys.path.insert(0, f"{ROOT}/scripts")
    from normalize_golden import normalize

    doc = b'{"a":1,"git_hash":"abc123","z":2}'
    once = normalize(doc)
    assert b'"git_hash":"GOLDEN"' in once
    assert normalize(once) == once  # 멱등


# ------------------------------------------------------------- q8 LUT 재검증
def test_q8_lut_full_range_equals_exp():
    from core.types import q8_to_p

    for q in range(256):
        assert q8_to_p(q) == math.exp(-q / 16.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
