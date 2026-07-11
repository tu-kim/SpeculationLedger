# bench/loadgen — Poisson arrival · batch sweep (Phase 4, G6)

Phase 4 진입 시 구현한다. 계획(CLAUDE.md §4):
- Poisson arrival로 OpenAI 호환 엔드포인트에 다중 세션 부하 → goodput–batch 곡선
- budget controller sweep (draft budget × batch size)
- Dynamo 다중 worker 라우팅 (D5)

Phase 0 산출물 중 재사용: metrics.parquet 스키마, analysis/plots.py의 곡선 플롯.
