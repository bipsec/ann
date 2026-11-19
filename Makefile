# ANN index-selection study — stage + analysis orchestration.
#
# PY defaults to the project venv's python.
# This host's faiss-cpu 1.15.1 is an AVX2 build (get_compile_options() == "OPTIMIZE DD AVX2"),
# so measured latency numbers from here are AVX2-representative; see logs/decisions.md.

PY ?= ./.venv/bin/python
CORES ?= 0-15

.PHONY: test hardware acceptance budget paper stage-reports stage-c-plan stage-a stage-b stage-c stage-c-smoke consolidate-filters stage-d \
	analysis pilot clean-raw all

hardware:
	$(PY) -m annstudy.hardware

acceptance:
	$(PY) -m annstudy.acceptance --checks all

budget:
	$(PY) -m annstudy.budget --cores 16

paper:
	$(PY) -m annstudy.paper

stage-reports:
	$(PY) -m annstudy.stage_reports

test:
	$(PY) -m pytest -q

pilot:
	$(PY) -m annstudy.coordinate --config configs/pilot.yaml --phase all --max-build-variants 1 --cores $(CORES)

stage-a:
	$(PY) -m annstudy.coordinate --config configs/stage_a.yaml --phase all --cores $(CORES)

# 8-bit scalar-quantized HNSW at two cells — tests whether the sub-raw memory regime (section 5.3)
# is genuinely empty or just absent from the original grid.
stage-sq:
	$(PY) -m annstudy.coordinate --config configs/stage_sq.yaml --phase all --cores $(CORES)

stage-b:
	$(PY) -m annstudy.coordinate --config configs/stage_b.yaml --phase all --keep-indexes --cores $(CORES)

# Stage C reuses the Stage B indexes retained for it. Latency must stay single-threaded, so
# parallelism comes from sharding across pinned cores. Driven cell by cell: each shard holds the cell's corpus AND index (13.1 GB at N=1M/D=1536),
# so the shard count is chosen per cell to fit memory rather than filling all 16 cores.
stage-c:
	./scripts/stage_c.sh

stage-c-plan:
	$(PY) -m annstudy.filter_runner --plan

stage-c-smoke:
	$(PY) -m annstudy.filter_runner --config configs/stage_c_smoke.yaml \
		--results-dir /tmp/stage_c_smoke --max-tuning-queries 150 --max-test-queries 300 \
		--gt-threads 16

consolidate-filters:
	$(PY) -m annstudy.filter_runner --consolidate

stage-d:
	$(PY) -m annstudy.update_runner --config configs/stage_d.yaml

analysis:
	$(PY) -m annstudy.analysis.run_all

all: stage-a stage-b stage-c stage-d analysis

clean-raw:
	@echo "Refusing to auto-delete results/raw (append-only). Remove manually if intended."
