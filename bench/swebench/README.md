# bench/swebench — SWE-bench 준비·평가 래퍼 (Phase 2)

- 평가 호스트: **x86_64 Linux** 권장 (프리빌드 이미지가 x86 전용; arm64는
  `--namespace ''` 로컬 빌드, experimental — docs/ENV.md §3).
- 데이터셋: `SWE-bench/SWE-bench_Lite` (구 `princeton-nlp/SWE-bench_Lite` 별칭 동작).
- Lite-50 고정 목록: `lite50.txt` — `make_lite50.py`가 결정적으로 생성(사전순 상위 50).
  목록 파일이 곧 계약이다 — 재생성으로 바뀌면 안 되며, 바뀌면 커밋 diff로 드러난다.

평가 커맨드(ENV.md §4에서 실측 고정):

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Lite \
  --predictions_path <preds.jsonl> \
  --instance_ids $(tr '\n' ' ' < lite50.txt) \
  --max_workers 4 --run_id <exp_id>
```
