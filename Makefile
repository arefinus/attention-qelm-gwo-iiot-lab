PY ?= python
PKG = attention_qelm_gwo_iiot_lab

.PHONY: setup smoke demo test evaluate train search audit figures fixture

setup:
	pip install -e . && pip install pytest

smoke:
	$(PY) -m $(PKG) smoke

demo:
	$(PY) -m $(PKG) demo

test:
	$(PY) -m pytest -q

evaluate:
	$(PY) -m $(PKG) evaluate --predictions examples/output/predictions.csv --config configs/demo_multiclass.yaml

train:
	$(PY) -m $(PKG) train --config configs/demo_multiclass.yaml --out examples/output/train_multiclass

search:
	$(PY) -m $(PKG) search --config configs/demo_multiclass.yaml --gwo configs/gwo_demo.yaml --out examples/output/search

audit:
	$(PY) -m $(PKG) audit --out examples/output/audit

figures:
	$(PY) tools/render_figures.py

fixture:
	$(PY) tools/make_fixture.py --out examples/fixtures/v1 --seed 42
