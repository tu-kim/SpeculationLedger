"""I5 sim 순수성: core/·sim/에서 torch/cuda import 금지 (CLAUDE.md §9).

1차 방어선은 ruff banned-api(pyproject.toml), 이 테스트는 AST 백스톱이다.
"""

import ast
import pathlib

BANNED = {"torch", "cupy", "tensorflow", "jax", "triton", "vllm"}
ROOT = pathlib.Path(__file__).resolve().parent.parent


def _imports_of(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def test_core_and_sim_are_pure():
    offenders = []
    for pkg in ("core", "sim"):
        for py in sorted((ROOT / pkg).rglob("*.py")):
            bad = _imports_of(py) & BANNED
            if bad:
                offenders.append((str(py.relative_to(ROOT)), sorted(bad)))
    assert not offenders, f"I5 위반: {offenders}"


def test_core_does_not_import_sim_or_vllm_plugin():
    """core는 최하층이다 — 상위 패키지 의존 금지 (sim/online 공유 보장)."""
    offenders = []
    for py in sorted((ROOT / "core").rglob("*.py")):
        bad = _imports_of(py) & {"sim", "vllm_plugin", "bench", "analysis"}
        if bad:
            offenders.append((str(py.relative_to(ROOT)), sorted(bad)))
    assert not offenders, f"계층 위반: {offenders}"
