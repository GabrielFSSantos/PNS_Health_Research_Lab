.PHONY: install install-gpu lint typecheck test test-gpu run-v2-gpu format

VENV ?= .venv
VENV_GPU ?= .venv-gpu
PY := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PY_GPU := $(if $(wildcard $(VENV_GPU)/bin/python),$(VENV_GPU)/bin/python,$(PY))
UV := $(if $(shell command -v uv 2>/dev/null),uv,)

install:
	@if [ -n "$(UV)" ]; then uv sync --all-extras; else $(PY) -m venv $(VENV) && $(VENV)/bin/pip install -e ".[dev]"; fi

install-gpu:
	@PY312=$$(command -v python3.12 2>/dev/null || true); \
	if [ -n "$$PY312" ]; then \
	  $$PY312 -m venv $(VENV_GPU); \
	else \
	  $(PY) -m venv $(VENV_GPU); \
	fi
	$(VENV_GPU)/bin/pip install -U pip
	$(VENV_GPU)/bin/pip install -e ".[gpu,dev]" || echo "Se falhar, instale Python 3.12 e cuml manualmente (ver docs/gpu_setup.md)"

lint:
	$(PY) -m ruff check src tests

format:
	$(PY) -m ruff format src tests

typecheck:
	$(PY) -m mypy src/pns_health_research

test:
	$(PY) -m pytest tests/ -m "not gpu"

test-gpu:
	$(PY_GPU) -m pytest tests/ -m gpu

run-v2-gpu:
	$(VENV_GPU)/bin/pns-research experiment --config configs/experiments/research_v2.yaml
