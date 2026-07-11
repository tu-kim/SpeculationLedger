# SpecLedger 실험 타깃 (CLAUDE.md §8)
UV ?= uv run
CONFIG_DIR := configs

.PHONY: setup lint test golden sim-g1 sim-g2 sim-g3 sim-all bench-d3 \
        online-smoke bench-lite50 load-sweep clean

setup:
	uv sync --group dev

lint:
	$(UV) ruff check core sim analysis tests bench

test:
	$(UV) pytest -q

# --- Phase 0 (GPU 불필요) ---------------------------------------------------
sim-g1:
	$(UV) python -m sim.gates --config $(CONFIG_DIR)/g1_recurrence.yaml

sim-g2:
	$(UV) python -m sim.gates --config $(CONFIG_DIR)/g2_oracle.yaml

sim-g3:
	$(UV) python -m sim.gates --config $(CONFIG_DIR)/g3_hitrate.yaml

sim-all: sim-g1 sim-g2 sim-g3

bench-d3:
	$(UV) python -m sim.bench_d3 --out results/d3_bench

# I4 golden 재생성 (diff 리뷰 후 커밋할 것). git_hash는 정규화된다.
golden:
	$(UV) python -m sim.gates --config $(CONFIG_DIR)/synth_smoke.yaml --out results --no-plots
	cp results/synth_smoke/gates.json tests/golden/synth_smoke_gates.json
	$(UV) python scripts/normalize_golden.py tests/golden/synth_smoke_gates.json

# --- Phase 1+ (GPU 필요 — 이 호스트에서는 차단됨, docs/ENV.md 참고) -----------
online-smoke:
	@$(UV) python scripts/require_gpu.py online-smoke

bench-lite50:
	@$(UV) python scripts/require_gpu.py bench-lite50

load-sweep:
	@$(UV) python scripts/require_gpu.py load-sweep

clean:
	rm -rf results/* traces/synth .pytest_cache .ruff_cache
