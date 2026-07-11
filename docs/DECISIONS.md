# DECISIONS.md — 결정 기록

> CLAUDE.md §0.1: 계약을 바꿔야 하면 먼저 여기 기록하고 사용자 승인을 받는다.
> 상태: RESOLVED(확정) / PROPOSED(가정 — **사용자 승인 대기**) / OPEN(미결).

## D1. scope 전달 방식 — RESOLVED (2026-07-11, 소스 실측)

**결정: custom HTTP header.** OpenCode(1.17.18, `9976269`)는 3중으로 지원한다:
1. provider 정적 헤더: `provider.<id>.options.headers` (options는 StructWithRest —
   임의 키 허용, `packages/core/src/v1/config/provider.ts:76-120`), 값에 `{env:VAR}` 치환
   가능 (`config/variable.ts:33-38`).
2. 모델별 정적 헤더: `provider.<id>.models.<m>.headers` (`provider.ts:62`).
3. per-request 동적: plugin `chat.headers` 훅 — sessionID·model·agent 접근
   (`packages/plugin/src/index.ts:257-260`, 주입 `session/llm/request.ts:134-146,203`).

추가로 비-opencode provider에는 `X-Session-Id`/`x-session-affinity`가 **자동 첨부**된다
(`request.ts:196-201`). → 채택 구성: tenant/repo는 인스턴스 래퍼 스크립트의 env →
`options.headers`, session은 자동 헤더 사용. per-instance 배치 실행:
`opencode run --format json -m <provider>/<model> [--session <id>] "<prompt>"`.
템플릿: `bench/opencode/`.

## D2. harvester hook 위치 — RESOLVED(후보 확정) / 런타임 확정은 Phase 1

vLLM `04d553f` 소스 실측으로 후보 3곳 특정 — `docs/HOOKS.md` §2.
시작점은 `GPUModelRunner._sample` 래핑(후보 C), I2 최적화 필요 시
`RejectionSampler.forward` 서브클래스(후보 A). custom_class proposer 경로로 fork 불필요.
**파생 결정: v1은 tree drafting 미지원 → 온라인 프로파일 A는 chain 제안** (HOOKS.md §1).

## D3. 서명 구현: rolling hash vs suffix automaton — RESOLVED (μbench 2026-07-11)

`make bench-d3` (60k tokens, M4, Python 3.12; results/d3_bench/bench.json):

| 지표 | rolling stack | suffix automaton |
|---|---|---|
| μs/token (Python 참조 구현) | 11.49 | 0.66 |
| 메모리 | **64 B 상수** (ring 8 tokens) | 2.27 MB / 60k tokens (101k states, 무한 성장) |
| 기능 | 고정 차수 2..8 서명 (store 키와 1:1) | 정확 최장 match + count |

**결정: 온라인 store 키링은 rolling hash.** 근거: (i) store는 고정 차수 2..8 fold key만
필요 — SAM의 가변 최장 match는 잉여, (ii) 메모리 O(1) vs 세션당 무한 성장(호스트 메모리
영속화 목표와 상충), (iii) Python μs 수치는 인터프리터 오버헤드 지배 — 네이티브에서는
차수당 mul+add 몇 개(rolling)라 역전됨. SAM은 **sim측 G1 재발 분석 전용**으로 유지
(`sim/replay.py:recurrence_stats`). 참고: Python 참조 store lookup 44.9μs — G4(p50≤5μs)는
네이티브 포팅 몫으로 확인됨.

## D4. K2(dense root key) 온라인 채택 — OPEN

G-R3 통과 시에만 (CLAUDE.md §11). sim 통제군 (a′)는 구현 완료 (`sim/dense_key.py`).

## D5. Dynamo 도입 시점 — OPEN (계획 유지)

Phase 1은 vLLM OpenAI 서버 직결. 실측 발견: dynamo 1.3.0은 vllm==0.24.0 핀 + 단일
노드도 etcd/NATS 필요 (ENV.md §3) — Phase 4 진입 시 SpecLedger 플러그인의 0.24.0 호환
재실측 필요.

---

## 가정 기록 (roadmap 원본 부재로 인한 잠정 확정 — 사용자 승인 대기)

### A-1. gate 운영 정의·임계값 — PROPOSED

`research/speculation-ledger-roadmap.md`(v2)가 저장소에 없어(2026-07-11 확인) G1/G2/G3/
G-R1의 판정식·임계값을 CLAUDE.md 본문에서 도출해 잠정 확정했다. 산식은
`sim/gates.py` docstring, 기본 임계값은 configs와 gates.json에 `thresholds` 블록으로
기록된다(`assumed: true` 마킹). **roadmap 원본 입수 시 이 정의를 대조·수정할 것.**
- G1: within-session P(suffix match_len ≥ 4) ≥ 0.30
- G2: oracle τ(a) ≥ 2.0
- G3: ∃cap: bytes ≤ 64MiB ∧ hit_rate ≥ 0.8 × unbounded
- G-R1: (τ_a − τ_b)/τ_b ≥ 0.10 (전체 또는 support ≥ 300 steps인 seg)
제안: G-R1은 oracle τ와 함께 draft 효율(accepted/drafted)을 병기해 판정하는 것을 검토
— oracle 무비용 drafting에서는 무차별 심층 제안이 τ를 부풀린다(§8.1 참고).

### A-2. proposer (a)(a′)(b)(c) 매핑 — PROPOSED

- (a) ledger: suffix-key + outcome annotation(acc/rej/p̂/correction) — 본 연구
- (a′) dense: 64-d dense key + positive-only — §7 명시 준수
- (b) positive: suffix-key + positive-only — (a)와 단일 변수 차이(outcome ablation)
- (c) recycle: Token Recycling 충실 재현 (`1b4c05c` 실측: M=(vocab,8) zeros init·
  overwrite·전역 영속(→tenant 격리로 완화), 정적 트리 2.2.2 80노드, accept=1+best_reward)
sim 한계(공통 적용): trace에는 realized 경로의 top-k만 있어 TR의 "기각 가지 위치 M 갱신"
은 재현 불가 — TR 과소평가 방향. (a)의 p̂도 동일 제약.

### A-3. 합성 trace — PROPOSED (하네스 검증 전용)

실 trace 자산 부재로 `sim/synth.py`(str_replace형 old/new 편집, 일관 rename, 파일
locality, 템플릿화된 think)로 하네스를 검증한다. gates.json에
`trace_provenance: synthetic`이 박히며 **연구 가설의 판정 근거가 아니다.** 실 trace는
Phase 1 tracer 또는 vLLM CPU 경로(ENV.md §3 참고)로 확보한다.
2026-07-11 합성 결과(g2_oracle): τ(a)=1.826 ≈ τ(b)=1.825, G-R1≈0 — 합성 워크로드에선
annotation 순가치가 중립. **G-R1 판정은 실 trace 필수**라는 로드맵 전제를 재확인.

### A-4. VerifyOutcome 확장 필드 — PROPOSED

계약 §3.3 튜플에 `ctx_tail`(서명 재계산용 직전 토큰 ≤8개)과 `file_id`(epoch domain)를
추가했다. 온라인 harvester는 두 값을 자연히 보유(시퀀스 버퍼, SegmentFSM)하므로 계약
비용 없음. HotEntry에는 `dom u32`(file_id) 필드가 추가된다 — 48-64B 레이아웃은 k≤5
기준 유지(11+4+9k).

### A-5. trace 스키마 v1 선택 확장 `events` — PROPOSED

`{"pos": 완료 지점, "bump_pos": 무효화 발화 지점, "type": "file_edit", "file": ...}`.
`pos`는 CODE 위치↔파일 결속 범위의 끝, `bump_pos`는 epoch bump 시점(기본 pos).

### A-6. (a′) dense key 사영 — PROPOSED

SENSE 공개 PCA 가중치를 오프라인에서 미확보 — 결정적 feature-hash 사영(64-d, int8
격자)을 프록시로 사용. 교체 지점은 `sim/dense_key.py:_embed` 단일 함수. 기본 인덱스는
IndexFlatL2(정확 탐색 = dense key 상한 성능 — (a)에 보수적), `ivfpq: true`로 압축 실측.

### A-7. invalidation 발화 시점 정련 — PROPOSED

§3.4 "생성 완료 시점"을 "write-내용 인자 시작 경계"로 앞당긴다(FSM이 edit_file+path를
파싱한 직후). 효과: 새 내용 harvest가 새 epoch에 실려 다음 편집에서 재사용됨. 안전성:
조기 무효화는 recall만 낮출 뿐 losslessness(I1)와 무관. 근거 실험: 완료 시점 발화 시
new-block 엔트리가 자기 bump에 무효화되어 code τ 재사용이 소실됨(2026-07-11 sim 실측).

### A-8. blend 랭킹의 annotation 신뢰 게이트 — PROPOSED (core 수학의 실질 설계 결정)

저차수 키는 다수 문맥이 겹쳐(앨리어싱) rej·p̂가 노이즈 — annotation은
`match_len ≥ p_hat_min_order(4)` 소스에서만 블렌드하고 λ-질량 게이트
`w/(w+anno_prior_w)`를 곱해 반영, 랭킹 기저는 acc-빈도(=positive-only와 동일).
상대 점수 10% 미만 차이는 support→tok로 결정(luck 비대칭 제거). 근거: 게이트 없이는
(a)가 다봉 slot 위치에서 rotation-rejection 함정에 빠져 (b)에 계통적으로 밀림
(2026-07-11 sim 실측, tool seg τ 2.86→3.50 회복).
