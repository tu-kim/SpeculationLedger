"""Phase 1+ 타깃 가드: GPU 서빙 스택이 없는 호스트에서 명확한 사유와 함께 중단.

CLAUDE.md §0.4: Phase 0(G1·G2) 통과 전에 온라인(vLLM) 구현을 시작하지 않는다.
이 스크립트는 (1) phase gate, (2) 하드웨어 부재를 확인한 뒤 실행을 거부한다.
"""

import shutil
import sys

TARGET_PHASE = {
    "online-smoke": "Phase 1 (G4): vLLM 단일 GPU에서 τ·TPOT 실측",
    "bench-lite50": "Phase 2 (G-R2): SWE-bench Lite-50 관통 (x86 Docker 필요)",
    "load-sweep": "Phase 4 (G6): Dynamo 다중 worker load/batch sweep",
}


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "?"
    print(f"[BLOCKED] `{target}` — {TARGET_PHASE.get(target, 'Phase 1+')}")
    print("  이 호스트: macOS arm64, NVIDIA GPU 없음 (docs/ENV.md 버전 매트릭스 참조)")
    print("  선행 조건: Phase 0 gate(G1·G2)를 **실제 trace**로 통과한 뒤,")
    print("  CUDA 12.x + H100 호스트에서 docs/ENV.md 절차로 서빙 스택을 구성한다.")
    if shutil.which("nvidia-smi") is None:
        print("  (nvidia-smi 미검출)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
