# CLAUDE.md — SpecLedger Testbed

Speculation Ledger(검증 산출물을 host memory에 영속화하는 speculative decoding) 연구를 위한 실험 테스트베드.
스택: OpenCode(coding agent) × SWE-bench(workload) × Dynamo frontend + vLLM backend(serving) × SpecLedger(본 연구 구현).
연구 계획의 원본은 `research/speculation-ledger-roadmap.md`(v2)이며, 이 문서의 모든 gate 번호(G1–G6, G-R1–R3)는 그 문서를 가리킨다.

## 0. 이 문서를 읽는 Claude Code에게 — 작업 규약

1. 이 문서는 계약이다. 구현이 계약과 충돌하면 구현을 고친다. 계약을 바꿔야 하면 먼저 `docs/DECISIONS.md`에 기록하고 사용자 승인을 받는다.
2. **Resolve-then-Record.** 버전에 민감한 정보(정확한 CLI 플래그, 설정 키, vLLM 내부 hook 위치)는 이 문서에 하드코딩하지 않는다. 설치 시점에 `--help`, 공식 문서, 소스 코드로 실측 확정한 뒤 `docs/ENV.md`(버전 매트릭스와 실행 커맨드), `docs/HOOKS.md`(vLLM 커밋 해시 + hook 위치)에 커밋한다. 이유: vLLM, Dynamo, Arctic Inference의 플래그와 내부 경로는 릴리스마다 변한다.
3. 모든 실험은 `configs/*.yaml` + git commit hash + seed로 재현 가능해야 한다. 수기 실행 결과는 인정하지 않는다.
4. Phase 순서를 지킨다. 특히 **Phase 0(G1·G2) 통과 전에 온라인(vLLM) 구현을 시작하지 않는다.** 이 연구는 전제가 기각되면 싸게 멈추는 것이 설계 목표다.
5. 커밋/PR 메시지에 관련 gate를 명시한다. 예: `[G2] oracle replay: outcome annotation ablation`.

## 1. 시스템 개요

```
[Workload]   SWE-bench instance × OpenCode agent
                 │  OpenAI-compatible HTTP (/v1/chat/completions)
[Serving]    Dynamo frontend ──▶ vLLM worker(s)
                 │                        │
[SD]         SpecLedger vLLM plugin: LedgerProposer · OutcomeHarvester · SegmentFSM · Tracer
                 │                        │
[Store]      core.LedgerStore (host DRAM, write-behind) ⇄ NVMe tier
                 │
[Data]       traces/*.jsonl ──▶ sim/ (offline replay: G1·G2·G3·G-R1)
```

설계 대원칙: **sim과 online은 `core/`를 공유한다.** 한 구현, 두 하네스. 시뮬레이터에서 검증한 τ가 온라인으로 이전됨을 보장하는 유일한 방법이다.

## 2. 리포 구조

```
specledger/
  core/            # ledger 자료구조·수학 (torch/cuda import 금지)
    store.py       #   LedgerStore: 이중 back-off, posterior, epoch
    arena.py       #   V2 span arena + patch + break histogram
    backoff.py     #   λ(match_len, scope_depth, count) 보간
    signature.py   #   rolling hash / suffix automaton (결정 D3)
    types.py       #   VerifyOutcome, Entry, Scope, Segment 공용 타입
  vllm_plugin/     # 온라인 통합 (Phase 1+)
    proposer.py    #   LedgerProposer (vLLM v1 proposer interface)
    harvester.py   #   verify 직후 outcome 수확 → pinned buffer → queue
    segment_fsm.py #   think/tool/code 태깅 + write-tool invalidation 이벤트
    tracer.py      #   서버측 trace JSONL 기록
  sim/             # 오프라인 (Phase 0, GPU 불필요)
    replay.py      #   trace → oracle τ, 재발률, hit-rate
    proposers.py   #   (a)(a′)(b)(c) 4종
    dense_key.py   #   (a′): PCA-64 + faiss IVF-PQ (SENSE 공개값으로 초기화)
    gates.py       #   G1/G2/G3/G-R1~R3 판정 → results/<exp>/gates.json
  bench/
    opencode/      #   provider 설정 템플릿, per-instance 실행 래퍼
    swebench/      #   instance 준비·평가 harness 래퍼, Lite-50 고정 목록
    loadgen/       #   Poisson arrival, batch sweep (Phase 4)
  analysis/        # learning curve, per-segment τ, goodput 플롯
  configs/         # 실험 yaml
  docs/            # ENV.md, HOOKS.md, DECISIONS.md, PATCHES.md
  tests/           # 불변식 I1–I5
```

## 3. 모듈 계약

### 3.1 core.LedgerStore

```python
lookup(sig_stack: list[u64], scope_stack: list[ScopeId], seg: Segment) -> Posterior | None
harvest(events: list[VerifyOutcome]) -> None      # 절대 블로킹 금지
bump_epoch(scope: ScopeId, file_id: u32) -> None  # O(1)
snapshot(path) / load(path)
stats() -> LedgerStats                            # hit_rate, entries, bytes, queue_depth, drops
```

- 성능 계약(G4): lookup p50 ≤ 5 μs, p99 ≤ 50 μs. harvest는 bounded queue + drop-oldest(드롭 카운터 노출).
- HotEntry 레이아웃(48–64 B): `key u64(suffix hash ⊕ scope ⊕ seg) · epoch u16 · hdr u8(k:4b | seg:2b | flags) · cand[k]{tok u32, logp̂ u8, acc u16, rej u16}`. k ∈ 1..16, 저장된 top-k coverage ≥ 0.9까지 적응 확장.
- 이중 back-off: 차수 2..8 × scope(session→repo→global)를 단일 보간 λ(match 길이, scope 깊이, count)로 통합.
- correction 필드는 없다. greedy의 correction = p̂ argmax, T>0의 correction 분포 = p̂ 자체.
- V2: `SpanEntry{arena_off u40, len u16}` + `breaks[{off u16, patch → HotEntry*}]`. arena는 append-only, content-hash dedup, 주기적 compaction(count decay 겸행).
- epoch invalidation은 lazy: 읽기 시 비교, 실제 폐기는 compaction에서.

### 3.2 vllm_plugin.LedgerProposer

- vLLM v1 proposer 인터페이스 구현. ngram proposer를 참조 골격으로 삼는다.
- `propose(ctx, budget) -> DraftTree`. 규칙: posterior 기대 accept 길이로 확장, reject-dominant edge는 budget 0 + patch splice, V2 span은 통째 제안 후 break offset에서 budget pre-split.
- **strict mode에서 acceptance rule을 절대 건드리지 않는다**(불변식 I1). relaxed 로직은 Phase 3 전까지 코드에 존재하지 않는다.

### 3.3 vllm_plugin.OutcomeHarvester

- 계약: verify step 직후 seq별 `(draft_ids, accepted_len, bonus_id, topk_ids[pos], topk_logp[pos] 8-bit 양자화, seg[pos])` 획득. k=8 기본.
- 구현: GPU에서 top-k gather를 fused로 수행, pinned buffer로 async D2H → lock-free queue → `store.harvest`. GPU 크리티컬 패스 개입 0이 목표(I2).
- hook 위치는 vLLM 버전 종속이다. 후보: v1 spec-decode 경로의 rejection/typical sampler 출력 지점. 실측 후 `docs/HOOKS.md`에 커밋 해시와 함께 기록(D2).

### 3.4 vllm_plugin.SegmentFSM

- 토큰 스트림에서 seg ∈ {think, tool, code, text}를 온라인 태깅: reasoning 태그/채널, tool-call JSON 괄호 FSM, 코드펜스·edit-tool 인자 구간.
- write/edit tool-call **생성 완료 시점**에 `InvalidationEvent(file_path)` 방출 → `bump_epoch`. 서버가 tool-call 텍스트를 직접 생성하므로 클라이언트 훅이 필요 없다.

### 3.5 sim.Replay

- 입력: §6 스키마의 trace JSONL. 출력: proposer 4종의 oracle τ, 재발률 CDF, hit-rate–size 곡선, gates.json.
- proposer와 store는 core를 그대로 사용한다. torch import 금지(I5).

## 4. Phase 계획 (roadmap gate 매핑)

| Phase | 범위 | 통과 gate | Definition of Done |
|---|---|---|---|
| 0 | sim/ 완성: proposer (a)(a′)(b)(c), 재발률·τ·hit-rate | G1, G2, G3, G-R1 | 기존 trace로 gates.json 자동 생성. GPU 미사용 |
| 1 | vLLM plugin: Harvester + Proposer(프로파일 A) + Tracer, baseline 활성화 | G4, I1 CI | 단일 GPU에서 τ·TPOT 실측. vanilla/ngram/arctic/eagle3를 동일 스크립트로 구동 |
| 2 | V2 arena/patch/break + epoch invalidation, OpenCode E2E | G-R2 | SWE-bench Lite-50 관통, per-segment τ 분해표 |
| 3 | segment 정책 + relaxed-think(기본 off, think 한정 구조 가드) | G5 | pass@1 비교표 |
| 4 | Dynamo 다중 worker, load/batch sweep, budget controller | G6 | goodput–batch 곡선, learning curve 플롯 |

Phase 1까지는 Dynamo 없이 vLLM OpenAI 서버 직결로 변수를 줄이는 것을 허용한다. 동형 API이므로 OpenCode는 baseURL만 교체하면 된다(D5).

## 5. 환경 구축 절차 (Resolve-then-Record)

스택: Ubuntu 22.04+, CUDA 12.x, H100, Python 3.11+, uv, Docker(SWE-bench 평가용, 디스크 ≥ 100 GB, x86).

순서:
1. 버전 매트릭스 확정 → `docs/ENV.md`: vllm(최신 안정 버전 핀), ai-dynamo, arctic-inference, faiss-cpu, opencode, swebench. **arctic-inference ↔ vllm 호환부터 확인한다** — 플러그인이 특정 vLLM 버전에 락되는 이력이 있다.
2. vLLM 단독 smoke: 소형 모델(Qwen3-0.6B급)로 ngram spec decode 활성화, spec 메트릭 출력 확인.
3. arctic-inference 설치 후 suffix decoding smoke (anchor ① baseline).
4. EAGLE-3 baseline smoke: Qwen3-8B + 공개 head(예: `Tengyunw/qwen3_8b_eagle3` — 존재와 호환을 실측 확인).
5. Dynamo frontend + vLLM backend smoke: `/v1/chat/completions` 관통, 스트리밍 확인.
6. OpenCode provider 설정: baseURL → 서빙 엔드포인트, 모델명 매핑. scope 전달 방식은 D1 확정 후 반영.
7. SWE-bench 1 instance 관통: 준비 → OpenCode 실행 → patch 산출 → 공식 harness 평가 → resolved 판정.

지뢰 목록(각 항목을 확인하고 `docs/ENV.md`에 O/X와 우회책 기록):
- spec decode × attention backend 조합 제약.
- tokenizer 정합: trace의 token id는 target tokenizer 기준으로 고정하고 `tokenizer_hash`를 기록한다.
- Dynamo ↔ vLLM 버전 매트릭스.
- SWE-bench 이미지 아키텍처(x86)와 디스크 사용량.
- OpenCode의 custom header 지원 여부(D1과 직결).
- 기존 testbed 자산(도커 이미지, 인증서 설정, 기존 agent trace 로그)이 있으면 재사용하고 출처를 ENV.md에 남긴다.

## 6. 데이터 스키마

trace JSONL — 서버측 tracer가 기록, `schema_version` 필수:

```json
{"schema_version": 1, "request_id": "...", "ts": 0,
 "scope": {"tenant": "...", "repo": "...", "session": "...", "instance_id": "swe-..."},
 "model": "...", "tokenizer_hash": "...",
 "steps": [{"pos": 0, "proposed": [0], "accepted_len": 0, "bonus": 0,
            "topk_ids": [[0]], "topk_logp_q8": [[0]], "seg": [0], "t_us": 0}],
 "final_text_sha": "..."}
```

metrics(parquet): per-request τ, TPOT, TTFT, wall-clock, seg별 τ / system: hit_rate, entries, bytes, queue_depth, drops, cpu_us_per_token / task: resolved(bool).

기존 trace 로그가 §6 스키마와 다르면 **변환기(`sim/convert.py`)를 먼저 작성**한다 — Phase 0의 첫 작업이다.

## 7. Baseline 매트릭스

| Baseline | 방법 | Phase | 비고 |
|---|---|---|---|
| vanilla | spec off | 1 | 절대 기준선 |
| ngram/PLD | vLLM 내장 | 1 | model-free 하한 |
| SuffixDecoding | arctic-inference | 1 | **anchor ①, 필수** |
| EAGLE-3 | 공개 head (Qwen3-8B) | 1 | model-based 대표 |
| Token Recycling | sim 재구현 | 0 | 공개 코드 확인 후 온라인 여부 결정 |
| dense-key positive-only | sim (a′) | 0 | key/value 효과 분리 통제군 |
| Aurora | 코드 공개 확인 후 | 4 | 불가 시 iso-GPU(우리 +0 vs 학습 +1) 조건 명시한 정성 비교 |

## 8. 실험 실행 규약

- config 하나 = 실험 하나. 예: `configs/g2_oracle.yaml`(trace 경로, proposer, back-off 차수, k, scope 계층, seed).
- 결과는 `results/<exp_id>/`: config 사본, git hash, gates.json, metrics.parquet, 플롯.
- Makefile 타깃: `make sim-g1 sim-g2 online-smoke bench-lite50 load-sweep`.

## 9. 불변식과 CI (머지 조건)

- I1 Losslessness: strict mode의 greedy 출력이 vanilla와 100% 일치(소형 모델, 100 prompts, e2e). 실패 시 머지 금지.
- I2 Non-blocking: harvest 큐 포화를 주입해도 TPOT 증가 ≤ 2%.
- I3 Scope 격리: cross-scope lookup이 API 수준에서 불가능(unit).
- I4 결정성: 동일 trace + config → byte-동일 gates.json(golden).
- I5 sim 순수성: `sim/`, `core/`에서 torch/cuda import 금지(lint).

## 10. 하지 말 것

- vLLM core fork 금지. plugin/entrypoint로만 통합한다. 불가피한 최소 패치는 diff와 대상 커밋을 `docs/PATCHES.md`에 남긴다.
- relaxed acceptance를 tool/code segment에 적용하는 코드 경로를 만들지 않는다. Phase 3 이후에도 구조적 가드로 막는다.
- sim과 online의 정책 로직 분기 금지 — core만이 진실이다.
- full logits, 원시 hidden state, KV cache를 저장하지 않는다(top-k와 64-d int8 사영만).
- trace에는 tenant 코드가 담긴다. 리포 외부 유출 금지, per-tenant 디렉토리로 격리한다.

## 11. 열린 결정 (확정 시 docs/DECISIONS.md에 기록)

- D1 scope 전달: OpenCode custom header 지원 여부 확인 → 불가 시 per-instance API key 매핑 또는 프롬프트 prefix hash.
- D2 harvester hook 위치: vLLM v1 소스 실측 후 확정.
- D3 서명 구현: rolling hash vs suffix automaton — Phase 0 μbench로 결정.
- D4 K2(dense root key) 온라인 채택: G-R3 통과 시에만.
- D5 Dynamo 도입 시점: Phase 1은 직결 허용, Phase 4에서 라우팅/다중 worker. PD disaggregation은 선택 확장.

## 12. 참고 (구현 관련)

SuffixDecoding arXiv 2411.04975 / arctic-inference · Token Recycling arXiv 2408.08696 · OSD arXiv 2310.07177 · Aurora arXiv 2602.06932 · EfficientEdit arXiv 2506.02780 · DReSD arXiv 2502.15572 · NEST arXiv 2405.19325 · SENSE arXiv 2606.00021 · EAGLE-3 head 예: Tengyunw/qwen3_8b_eagle3 · SWE-bench 공식 harness · OpenCode · NVIDIA Dynamo. 연구 로드맵: `research/speculation-ledger-roadmap.md` (v2).
