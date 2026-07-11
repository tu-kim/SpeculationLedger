"""I4 결정성: 동일 trace + config → byte-동일 gates.json (golden) (CLAUDE.md §9).

golden 재생성: `make golden` (diff 리뷰 후 커밋).
"""

import json
import pathlib

import pytest

from sim.gates import run

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = pathlib.Path(__file__).parent / "golden" / "synth_smoke_gates.json"


@pytest.fixture()
def smoke_doc(tmp_path):
    # trace를 tmp에 생성하도록 config를 복사·수정하지 않고, 결정적 synth 경로를 그대로
    # 사용하되 결과만 tmp로 보낸다. trace 파일 자체도 seed 결정적이다.
    trace_dir = ROOT / "traces" / "synth"
    smoke = trace_dir / "smoke.jsonl"
    if smoke.exists():
        smoke.unlink()  # 항상 재생성 → 생성기까지 포함한 결정성 검증
    out = run(str(ROOT / "configs" / "synth_smoke.yaml"), str(tmp_path), plots=False)
    return out, tmp_path


def test_gates_json_byte_identical_across_runs(smoke_doc, tmp_path):
    _, out1 = smoke_doc
    b1 = (out1 / "synth_smoke" / "gates.json").read_bytes()

    out2 = tmp_path / "second"
    run(str(ROOT / "configs" / "synth_smoke.yaml"), str(out2), plots=False)
    b2 = (out2 / "synth_smoke" / "gates.json").read_bytes()
    assert b1 == b2, "동일 config+trace인데 gates.json이 byte-불일치 (I4 위반)"


def test_gates_json_matches_committed_golden(smoke_doc):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from normalize_golden import normalize

    _, out1 = smoke_doc
    # git_hash는 커밋마다 정당하게 변한다 — 계산 결정성만 golden으로 고정 (§8 기록은 results/가 짐)
    got = normalize((out1 / "synth_smoke" / "gates.json").read_bytes())
    if not GOLDEN.exists():
        pytest.skip("golden 미생성 — `make golden` 후 커밋 필요")
    want = normalize(GOLDEN.read_bytes())
    if got != want:
        # 어디가 다른지 요약해서 실패 메시지에 제공
        g, w = json.loads(got), json.loads(want)
        diff_keys = [k for k in set(g) | set(w) if g.get(k) != w.get(k)]
        raise AssertionError(
            f"golden 불일치 (다른 top-level 키: {sorted(diff_keys)}) — "
            "의도된 변경이면 `make golden` 후 diff 리뷰·커밋"
        )


# traces/synth/*는 gitignore된 재생성 가능 산출물 — 정리는 `make clean`이 담당한다.
