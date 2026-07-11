# vllm_plugin — Phase 1 통합 설계 (구현은 Phase 0 gate 통과 후)

> **gate-blocked**: CLAUDE.md §0.4에 따라 G1·G2를 실제 trace로 통과하기 전에는 이
> 디렉토리에 실행 코드를 넣지 않는다. 아래는 D2 실측(docs/HOOKS.md)을 반영한 구현 계획.

## 파일 계획 (§2)

- `proposer.py` — `LedgerProposer(vllm_config)`: custom_class 경로
  (`speculative_config.model = "vllm_plugin.proposer.LedgerProposer"`).
  러너 호출 시그니처(04d553f): `propose(sampled_token_ids, num_tokens_no_spec,
  token_ids_cpu, slot_mappings=None)`. **chain 제안** (v1 tree 미지원 — HOOKS.md §1).
  내부는 `core.LedgerStore.lookup` 그대로 사용 — sim과 정책 분기 금지 (§10).
- `harvester.py` — `GPUModelRunner._sample` 래핑(후보 C)으로 시작, I2 미달 시
  `RejectionSampler.forward` 서브클래스(후보 A)에서 fused top-k gather + pinned buffer
  async D2H → lock-free queue → `core.LedgerStore.harvest`.
- `segment_fsm.py` — think/tool/code/text 온라인 태깅, edit-tool 인자 파싱 →
  `InvalidationEvent(file_path)` (발화 시점: write-내용 인자 시작 경계, DECISIONS A-7).
  scope는 OpenCode 자동 `X-Session-Id` 헤더 + provider 정적 헤더에서 획득 (D1).
- `tracer.py` — §6 스키마 JSONL 서버측 기록 (`sim/convert.py`의 validate_record가
  스키마 진실).

## 불변식 배선

- I1: strict mode에서 acceptance rule 불변 — proposer는 draft만 내고 sampler는 무수정.
- I2: harvest 경로는 bounded queue + drop-oldest (core.LedgerStore.harvest가 이미 계약).
- CI: tests/test_i1_i2_online.py의 skip 마커를 Phase 1에서 실테스트로 교체.
