PY := .venv/bin/python

.PHONY: test lint inference learning mnist report all

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check src tests experiments

# inference-side experiments (E1-E4): fast, no torch needed
inference:
	$(PY) -m experiments.e1_overlap
	$(PY) -m experiments.e2_depth
	$(PY) -m experiments.e3_surrogate
	$(PY) -m experiments.e4_witnesses

# learning-side experiments (E6 fact-table + pixels, E7): minutes on CPU
learning:
	$(PY) -m experiments.e6_facttable
	$(PY) -m experiments.e6_pixels
	$(PY) -m experiments.e7_depth_learning

# MNIST-scale (downloads MNIST to .data/ on first run)
mnist:
	$(PY) -m experiments.e5_mnist
	$(PY) -m experiments.e5b_noise_ablation

# CLUTRR-style systematic generalization
clutrr:
	$(PY) -m experiments.e8_clutrr

# scorecard + consolidated RESULTS.md (needs inference + learning outputs)
report:
	$(PY) -m experiments.scorecard
	$(PY) -m experiments.arena
	$(PY) -m experiments.report

# everything, in dependency order
all: test inference learning mnist clutrr report
