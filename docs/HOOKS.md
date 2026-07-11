# HOOKS.md — vLLM 통합 지점 (D2, Resolve-then-Record)

> 대상 커밋: **vllm-project/vllm `04d553f390fd37e09ab111936ef1592881299957`** (main, 2026-07-10, ~0.11.2.dev)
> 상태: **소스 실측 완료, 런타임 실측(GPU) 대기.** Phase 1 진입 시 이 커밋(또는 그 시점 재핀)으로
> 스모크를 돌려 각 항목을 O/X 갱신한다. 아래 file:line은 모두 해당 커밋 기준.

## 1. Proposer 통합 — fork 불필요 (CLAUDE.md §10 충족)

- vLLM v1에 커스텀 speculative method 1급 경로가 존재한다:
  `speculative_config={"model": "vllm_plugin.proposer.LedgerProposer", "num_speculative_tokens": N}`
  → method 자동 판정 `custom_class` (`vllm/config/speculative.py:640-647`)
  → `create_custom_proposer()`가 importlib로 로드 (`vllm/v1/spec_decode/custom_class_proposer.py:12`)
  → 인스턴스는 callable `propose`만 있으면 됨 (`custom_class_proposer.py:57-65`).
- 러너 호출 시그니처 (`vllm/v1/worker/gpu_model_runner.py:4952-4959`):
  `propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None)`
  — ngram proposer의 시그니처와 다름에 주의.
- 참조 골격(ngram): `NgramProposer` (`vllm/v1/spec_decode/ngram_proposer.py:12`),
  `propose(...)` (`:135`), 등록 dispatch (`gpu_model_runner.py:574-644`).
- 일반 플러그인 엔트리포인트: `vllm.general_plugins` group
  (`vllm/plugins/__init__.py:36-42`) — arctic-inference가 쓰는 방식. SpecLedger는
  custom_class 경로가 우선.

**설계 제약(실측):** v1 경로는 **tree drafting 미지원** (v1 트리 관련 심볼 부재,
rejection sampler 출력이 `[batch, max_spec_len+1]` 선형). → 온라인 LedgerProposer
프로파일 A는 **chain 제안**으로 구현한다. sim의 DraftTree는 유지하되(TR 재현·연구용)
온라인 어댑터는 최우수 chain을 평탄화한다. per-step 상한 `MAX_SPEC_LEN=128`
(`vllm/v1/sample/rejection_sampler.py:34`).

## 2. OutcomeHarvester hook 후보 (§3.3의 (draft_ids, accepted_len, bonus, topk) 획득 지점)

| 후보 | 위치 | 가용 텐서 | 평가 |
|---|---|---|---|
| **A. RejectionSampler.forward 서브클래스** | `vllm/v1/sample/rejection_sampler.py:88` | `metadata.draft_token_ids`(:170), `logits[target_logits_indices]`(:148), bonus(:143), `output_token_ids`(:169), top-k logprobs `_get_logprobs_tensors`(:184-199) | 가장 깊음 — 모든 텐서 동시 가시. fused top-k gather 넣기 최적 |
| **B. `_bookkeeping_sync`** | `gpu_model_runner.py:3636` | `sampler_output.sampled_token_ids`(:3671), `logprobs_tensors`(:3672), per-seq accepted = `len(valid_sampled_token_ids[i])`(:3702-3709) | CPU측·서브클래스 불필요. draft_ids는 그 step의 `spec_decode_metadata`와 상관 필요 |
| **C. `_sample` 래핑 (권장 시작점)** | `gpu_model_runner.py:3605` (호출부 :4501) | `logits` 전체, `spec_decode_metadata`, 반환 `sampler_output` 모두 한 프레임 | 코어 무패치로 최소 침습. I2(GPU 크리티컬 패스 0) 달성 위해 top-k gather는 A로 내려보낼 수 있음 |
| 참고: per-seq accepted 카운트 버퍼 | `gpu_model_runner.py:798`(할당), `:1549`(계산) | `num_accepted_tokens` | 메트릭 교차검증용 |

- top-k logprob: spec decode 중에도 `SamplingParams.logprobs=k`로 per-token top-k 산출됨
  (`rejection_sampler.py:184-216`, target·bonus 위치 모두). → **계약 §3.3의 k=8 수확이
  기존 경로로 가능** — 다만 요청 전역 logprobs 활성화의 오버헤드는 실측 필요.
  (오버헤드 크면 후보 A에서 fused gather로 대체.)
- 확정 절차: Phase 1 스모크에서 C로 시작 → TPOT/I2 측정 → 필요 시 A로 최적화.
  확정 시 이 표를 갱신하고 커밋 해시를 다시 기록한다.

## 3. SegmentFSM 참고

- 서버가 tool-call 텍스트를 직접 생성하므로 클라이언트 훅 불필요 (§3.4).
- OpenCode는 요청에 `X-Session-Id`/`x-session-affinity` 헤더를 자동 첨부
  (opencode `packages/opencode/src/session/llm/request.ts:196-201`) — scope의 session 축은
  서버측에서 이 헤더로 획득 가능 (D1).

## 4. CPU 경로 (참고)

- 핀 커밋 기준 CPU(v1)에서 spec decode 지원: `vllm/v1/worker/cpu_model_runner.py:31`
  (Triton 커널의 C++ 대체 monkey-patch, `:62-102`). macOS arm64는 소스 빌드 필요
  (FP32/FP16). GPU 확보 전 소형 모델 실 trace 채집 경로로 검토 가치 있음 — 미검증.
