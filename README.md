# SpecLedger Testbed

Speculation Ledger(검증 산출물을 host memory에 영속화하는 speculative decoding) 연구
테스트베드. **계약 문서는 [CLAUDE.md](CLAUDE.md)** — 구조·불변식·phase 계획의 진실이다.

## 현재 상태 (Phase 0 — GPU 불필요 구간)

- `core/` — LedgerStore(이중 back-off·posterior·epoch·V2 span arena), 서명, 공용 타입
- `sim/` — trace 변환기·합성 생성기·proposer 4종((a)(a′)(b)(c))·oracle replay·gate 러너
- `tests/` — 불변식 I3(scope 격리)·I4(결정성 golden)·I5(순수성) + 단위 55개
- `docs/` — ENV(버전 매트릭스)·HOOKS(vLLM 통합 지점 실측)·DECISIONS(D1·D2·D3 해결)·PATCHES
- `vllm_plugin/` — **gate-blocked** (Phase 0을 실 trace로 통과 전 온라인 구현 금지, §0.4)

## 빠른 시작

```bash
make setup     # uv sync
make test      # 불변식 + 단위 테스트 (perf 제외)
make perf      # 성능 리그레션 가드 (보정 단위 예산) → results/perf/perf.json
make ci        # lint + test + perf (.github/workflows/ci.yml 로컬 미러)
make sim-all   # G1/G2/G3 gate 러너 (합성 trace 자동 생성) → results/<exp>/gates.json
make bench-d3  # 서명 μbench (D3)
```

주의: 현재 gate 결과는 **합성 trace**(하네스 검증용)다 — `gates.json`의
`trace_provenance` 필드 참조. 연구 판정은 실제 trace 확보 후 (docs/DECISIONS.md A-3).
