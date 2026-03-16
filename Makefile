PYTHON_BOOTSTRAP ?= python

# Detect whether we are inside a venv (Windows) or a system install (Docker/Linux).
ifeq ($(wildcard .venv/Scripts/python.exe),)
  VENV_PYTHON := python
  VENV_PIP := pip
else
  VENV_PYTHON := .venv/Scripts/python.exe
  VENV_PIP := .venv/Scripts/pip.exe
endif

.PHONY: install lint fix typecheck test demo-preprocessing-pipeline demo-ml-track demo-dl-track clean

install:
	@if [ ! -f "$(VENV_PYTHON)" ]; then \
		$(PYTHON_BOOTSTRAP) -m venv .venv; \
	fi
	"$(VENV_PYTHON)" -m pip install --upgrade pip
	"$(VENV_PIP)" install -e ".[dev]"

lint:
	"$(VENV_PYTHON)" -m ruff check .
	"$(VENV_PYTHON)" -m ruff format --check .

fix:
	"$(VENV_PYTHON)" -m ruff check --fix .
	"$(VENV_PYTHON)" -m ruff format .

typecheck:
	"$(VENV_PYTHON)" -m mypy

test:
	"$(VENV_PYTHON)" -m pytest

demo-preprocessing-pipeline:
	"$(VENV_PYTHON)" -m brainrisk demo-preprocessing --output-dir artifacts --n-subjects 50

demo-ml-track:
	"$(VENV_PYTHON)" -m brainrisk demo-ml --output-dir artifacts --n-subjects 50

demo-dl-track:
	@echo "Placeholder"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist htmlcov .coverage artifacts
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	mkdir -pv .pytest_cache
