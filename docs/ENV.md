# ENV.md — 버전 매트릭스와 실행 커맨드 (Resolve-then-Record)

> CLAUDE.md §0.2·§5 규약에 따라 **실측 확정된** 버전·플래그만 기록한다.
> 마지막 실측: 2026-07-11, macOS 26.3 arm64 (Apple M4) — **GPU 없음 호스트**.

## 0. 호스트 상태 요약

| 항목 | 값 | 함의 |
|---|---|---|
| OS / arch | macOS 26.3 (Darwin 25.3), arm64 (M4) | CUDA 불가, x86 Docker 이미지 네이티브 불가 |
| NVIDIA GPU | 없음 (`nvidia-smi` 미검출) | **Phase 0(sim)만 이 호스트에서 수행** — CLAUDE.md §0.4와 정합 |
| 디스크 여유 | ~134 GiB | SWE-bench 전체 평가(≥120GB) 불가는 아니나 arch가 먼저 막음 |

Phase 1+ 는 Ubuntu 22.04+ / CUDA 12.x / H100 호스트 확보 후 §3의 순서로 진행한다.

## 1. Phase 0 (이 리포에서 실측 확정, 이 호스트에서 동작)

| 패키지 | 버전(핀) | 비고 |
|---|---|---|
| uv | 0.11.28 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Python | 3.12.13 | `.python-version`으로 핀, uv-managed |
| numpy | 2.5.1 | sim 전용 (core는 stdlib만) |
| faiss-cpu | 1.14.3 | arm64 wheel 확인, `faiss.omp_set_num_threads(1)`로 결정성(I4) |
| pyarrow | 25.0.0 | metrics.parquet |
| pyyaml | 6.0.3 | configs |
| pytest / ruff | 9.1.1 / 0.15.21 | `make test` / `make lint` |
| matplotlib | 3.11.0 | analysis 플롯 (dev group) |

실행: `make setup` → `make test` → `make sim-g1 sim-g2 sim-g3` → `make bench-d3`.

## 2. third_party 핀 (shallow clone, 2026-07-11; gitignore됨 — 재클론 시 이 해시로 checkout)

| repo | commit | 버전 문자열 | 출처 |
|---|---|---|---|
| vllm-project/vllm | `04d553f390fd37e09ab111936ef1592881299957` | ~0.11.2.dev (setuptools_scm, 태그 없음) | main @2026-07-10 |
| snowflakedb/ArcticInference | `9d4643881608178f0f58a33d1b61919ca246997f` | 0.2.1.dev0 | pyproject.toml:17 |
| ai-dynamo/dynamo | `15709131c2717090e7a05dee97f2036a7d797710` | 1.3.0 | pyproject.toml:6 |
| SWE-bench/SWE-bench | `f7bbbb2ccdf479001d6467c9e34af59e44a840f9` | 4.1.0 | swebench/__init__.py:1 |
| sst/opencode | `9976269ab1accfc9f9dc98a4a688c516934de422` | 1.17.18 | packages/opencode/package.json:3 |
| Luowaterbi/TokenRecycling | `1b4c05cc642d111f12c61d4a99950d737121d9fe` | (버전 태그 없음) | §7 baseline 재현 사양 출처 |

## 3. Phase 1+ 스택 호환 매트릭스 (소스 실측; 스모크는 GPU 호스트에서 재확인)

**핵심 발견: arctic-inference와 dynamo는 서로 다른 vLLM을 핀한다 — 단일 venv 공존 불가.**

| 조합 | 핀 | 근거 (파일:라인) | 판정 |
|---|---|---|---|
| arctic-inference ↔ vllm | **vllm==0.18.0** (`[vllm]` extra), 런타임 버전 강제 체크 | arctic pyproject.toml:52-54; plugin.py:29-34 (`ARCTIC_INFERENCE_SKIP_VERSION_CHECK=1`로 우회 가능) | X (락 실재) |
| dynamo ↔ vllm | **vllm[flashinfer,runai,otel]==0.24.0**, nixl 1.1.0, etcd+NATS 필요(단일 노드 포함) | dynamo pyproject.toml:59-71; docs/backends/vllm/README.md:105-111 | X (별도 env) |
| vllm main(핀 커밋) 자체 | `suffix` method가 **인트리** (`SpeculativeMethod` literal에 "suffix", v1/spec_decode/suffix_decoding.py 존재) | vllm/config/speculative.py:64-74 | O |

**우회책(제안, GPU 호스트에서 검증 후 확정):** baseline별 독립 venv. 최신 vLLM 하나로
vanilla/ngram/**suffix(인트리)**/EAGLE/custom_class(SpecLedger)를 모두 구동하고,
arctic-inference(0.18.0 락)는 SuffixDecoding 원저자 구현 교차검증용 별도 venv로만 사용.
Dynamo(Phase 4)는 vllm==0.24.0 env — SpecLedger 플러그인이 0.24.0의 custom_class와
호환되는지 그 시점에 실측한다.

### 지뢰 목록 점검 (CLAUDE.md §5)

| 항목 | O/X | 실측 내용 · 우회책 |
|---|---|---|
| spec decode × attention backend | X(제약 있음) | vLLM 핀 커밋: HPC attn ≤3 spec tokens (hpc_attn.py:133), GDN은 decode/spec-decode 혼합 배치 불가 (gdn_attn.py:409), MAX_SPEC_LEN=128 (rejection_sampler.py:34). **v1 경로 tree drafting 미지원** → 온라인 proposer는 chain 제안 |
| tokenizer 정합 | O(규약 확정) | trace는 target tokenizer 기준 token id + `tokenizer_hash` 기록 (§6). 합성 trace는 toy hash 사용 — 실 trace와 혼합 금지 (gates.json의 tokenizer_hash로 검출) |
| Dynamo ↔ vLLM 버전 | X(0.24.0 핀) | 위 표 참조. 단일 노드도 etcd+NATS 필요 (`docker compose -f dev/docker-compose.yml up -d`) |
| SWE-bench 이미지 arch | X(이 호스트) | 프리빌드 이미지는 x86 전용. arm64는 `--namespace ''`로 로컬 빌드(공식 experimental, README.md:74-76) — 평가는 x86 호스트 권장 |
| OpenCode custom header | **O** | D1 해결 — docs/DECISIONS.md D1, bench/opencode/ 템플릿 참조 |
| 기존 testbed 자산 | X(없음) | 2026-07-11 확인: 이 저장소 이전 자산(trace 로그·도커 이미지) 부재 → Phase 0은 합성 trace로 하네스 검증 (DECISIONS A-3) |
| vLLM macOS arm64 CPU | O(참고) | 핀 커밋 기준 Apple Silicon CPU 지원(소스 빌드, FP32/FP16, docs/getting_started/installation/cpu.apple.inc.md). CPU에서 spec decode 지원 명시 (v1/worker/cpu_model_runner.py:31). **소형 모델 실 trace 채집의 잠재 경로** — 단 성능/안정성 미검증 |

## 4. Phase 1+ 실행 커맨드 (소스에서 확정한 형태 — GPU 호스트에서 스모크 후 이 문서에 O/X 갱신)

```bash
# vLLM 단독 ngram smoke (§5.2)
vllm serve Qwen/Qwen3-0.6B \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 4,
                          "prompt_lookup_max": 8, "prompt_lookup_min": 2}'

# suffix decoding — 인트리 (vllm 핀 커밋, method="suffix")
vllm serve <model> --speculative-config '{"method": "suffix", "num_speculative_tokens": 8}'

# suffix decoding — arctic-inference 별도 venv (anchor ① 교차검증, vllm==0.18.0)
ARCTIC_INFERENCE_ENABLED=1 vllm serve <model> \
  --speculative-config '{"method": "suffix", "enable_suffix_decoding": true}'

# SpecLedger 플러그인 (Phase 1, custom_class 경로 — fork 불필요, D2 참조)
vllm serve <model> \
  --speculative-config '{"model": "vllm_plugin.proposer.LedgerProposer",
                          "num_speculative_tokens": 8}'

# Dynamo 단일 노드 (Phase 4; 사전: docker compose -f dev/docker-compose.yml up -d)
python -m dynamo.frontend                 # :8000 /v1/chat/completions
DYN_SYSTEM_PORT=8081 python -m dynamo.vllm --model <model> ...

# SWE-bench Lite 평가 (x86 호스트)
python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Lite \
  --predictions_path <preds.jsonl> --max_workers 4 --run_id <id>
```

EAGLE-3 baseline: 공개 head `Tengyunw/qwen3_8b_eagle3`의 존재·호환은 **미실측**(네트워크
확인 필요 항목) — Phase 1 진입 시 HF에서 확인 후 여기 기록.
