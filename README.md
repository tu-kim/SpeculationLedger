# SpecLedger Testbed

Speculation Ledger — **검증 산출물(verify outcome)을 host memory에 영속화하는 speculative
decoding** — 연구 테스트베드. **계약 문서는 [CLAUDE.md](CLAUDE.md)** 이며 구조·모듈 계약·
불변식·phase 계획의 진실이다. 연구 로드맵 원본(`research/speculation-ledger-roadmap.md` v2)은
현재 저장소에 없다 — gate 판정식은 [docs/DECISIONS.md](docs/DECISIONS.md) A-1의 잠정 정의로
운용 중이다.

## 상태 (Phase 0 완료 — GPU 불필요 구간)

| 구성요소 | 상태 |
|---|---|
| `core/` — LedgerStore(이중 back-off·posterior·epoch·V2 span arena), 서명, 공용 타입 | ✅ 구현·검증 완료 |
| `sim/` — trace 변환기·합성 생성기·proposer 4종·oracle replay·gate 러너·D3 μbench | ✅ 구현·검증 완료 |
| `vllm_plugin/` — 온라인 통합 | ⛔ **gate-blocked** (§0.4: 실 trace로 G1·G2 통과 전 구현 금지) |
| `bench/` — OpenCode provider 템플릿(D1 반영), SWE-bench Lite-50 고정 목록, loadgen 계획 | ✅ Phase 2+ 준비물 |
| `docs/` — ENV(버전 매트릭스)·HOOKS(vLLM 04d553f hook 실측)·DECISIONS(D1·D2·D3 해결)·PATCHES | ✅ Resolve-then-Record |
| CI — `.github/workflows/ci.yml` (checks + perf 2-job) | ✅ `make ci`로 로컬 미러 |

Gate 현황 (**합성 trace — 하네스 검증용, 연구 판정 아님**; gates.json의 `trace_provenance` 참조):
G1 PASS(0.32) · G2 FAIL(τ=1.83/2.0) · G3 PASS(0.82) · G-R1 ≈0 — outcome annotation의
순가치 판정은 실제 trace 확보가 선행되어야 한다는 로드맵 전제를 재확인 (DECISIONS A-3).

## 빠른 시작

```bash
make setup     # uv sync (Python 3.12 핀)
make test      # 불변식(I3/I4/I5) + 단위 + 코너 + 차분 스위트 (perf 제외)
make perf      # 성능 리그레션 가드 6종 → results/perf/perf.json
make ci        # lint + test + perf (.github/workflows/ci.yml 로컬 미러)
make sim-all   # G1/G2/G3 gate 러너 (합성 trace 자동 생성) → results/<exp>/gates.json
make bench-d3  # 서명 μbench (D3 판정 근거)
make golden    # I4 golden 재생성 (diff 리뷰 후 커밋)
```

Phase 1+ 타깃(`online-smoke`, `bench-lite50`, `load-sweep`)은 GPU 호스트 확보 전까지
`scripts/require_gpu.py`가 차단한다 — 선행 조건과 절차는 [docs/ENV.md](docs/ENV.md).

## 검증 체계 (4층)

1. **불변식·단위** — I3(scope/tenant 격리), I4(gates.json byte-결정성: golden +
   PYTHONHASHSEED 교차 프로세스 실측), I5(core/sim torch-금지 AST+ruff), 모듈 단위 테스트.
2. **코너 케이스** — 경계·포화·오입력·퇴화 구성 60여 종 + 시드 퍼징. 이 과정에서
   크래시 버그 7건·의미론 결함 6건을 사전 검출·수정했다 (커밋 로그 참조).
3. **차분(differential) 참조** — 해싱 없는 정확-컨텍스트 키잉의 NaiveStore와 무작위
   재발 워크로드에서 posterior **완전 일치**를 검증. 민감도 자가 계측(비-None 대조
   ≥15%) 내장, 변이 3종(tier 누락/off-by-one/seg 뒤틀림) 탐지력 실증.
4. **성능 리그레션** — 머신속도 보정 단위(unit) 예산 + 최적화-vs-참조 비율 하한 +
   연산-횟수 가드. CI perf job이 실행하고 추이 아티팩트를 보존한다.

## 성능 (M4, Python 3.12 참조 구현 — G4 μs 계약은 네이티브 포팅 몫)

값-보존 최적화 2라운드(전후 gates.json byte-동일 검증) 누적:

| 지표 | 최적화 전 | 현재 | 비고 |
|---|---|---|---|
| 서명 스택 push+stack | 11.2 μs/tok | **0.99 μs/tok** | 증분 점화식 `sig_o(t)=sig_{o-1}(t-1)·M+mix(tok)` — 네이티브 포팅 형태 |
| store lookup (21-probe+blend) | 45 μs | **~34 μs** (589 units) | λ pow-LUT·totals 캐시·fold 인라인 |
| sources_tuple 재구축 | 매 lookup | 캐시 (48×) | 변이 시점만 무효화 |
| g2 end-to-end (5 proposers+sweep) | 3.34 s | **2.85 s** | |

## 리포 구조 (CLAUDE.md §2)

```
core/          ledger 자료구조·수학 (torch 금지, sim/online 공유 단일 진실)
sim/           오프라인 replay 하네스 (Phase 0): convert·synth·proposers·replay·gates
vllm_plugin/   Phase 1 통합 설계 문서만 (gate-blocked)
bench/         opencode/(D1 헤더 템플릿) · swebench/(Lite-50) · loadgen/(Phase 4)
analysis/      learning curve·per-seg τ·hit-rate 플롯
configs/       실험 yaml (config 하나 = 실험 하나, §8)
docs/          ENV.md · HOOKS.md · DECISIONS.md · PATCHES.md
tests/         불변식·단위·코너·차분·perf (116 tests)
third_party/   참조 리포 6종 (gitignore — 핀 커밋은 docs/ENV.md §2)
```

## 다음 단계

1. **실제 trace 확보** — GPU 호스트에서 Phase 1 tracer, 또는 vLLM macOS arm64 CPU
   경로(소스 빌드, ENV.md §3)로 소형 모델 채집 → G1/G2/G-R1 실판정.
2. 로드맵 v2 원본을 `research/`에 추가하고 DECISIONS A-1 잠정 정의와 대조.
3. Phase 0 gate 통과 시 `vllm_plugin/` 구현 개시 (HOOKS.md의 hook 후보 실측 확정).
