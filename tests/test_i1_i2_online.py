"""I1(losslessness)·I2(non-blocking TPOT)는 온라인 불변식이다 — Phase 1 CI에서 활성화.

CLAUDE.md §9:
  I1: strict mode greedy 출력이 vanilla와 100% 일치 (소형 모델, 100 prompts, e2e)
  I2: harvest 큐 포화 주입 시 TPOT 증가 ≤ 2%
이 호스트(GPU 없음)에서는 실행 불가. Phase 0 gate 통과 후 vLLM 환경에서 구현한다.
큐 포화의 '자료구조 수준' 성질(드롭·비블로킹)은 tests/test_store.py가 커버한다.
"""

import pytest


@pytest.mark.online
@pytest.mark.skip(reason="Phase 1: GPU + vLLM 서빙 스택 필요 (docs/ENV.md) — CLAUDE.md §0.4")
def test_i1_losslessness_e2e():
    raise NotImplementedError


@pytest.mark.online
@pytest.mark.skip(reason="Phase 1: GPU + vLLM 서빙 스택 필요 (docs/ENV.md) — CLAUDE.md §0.4")
def test_i2_nonblocking_tpot():
    raise NotImplementedError
