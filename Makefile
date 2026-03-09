PYTHON_BOOTSTRAP ?= python
VENV_PYTHON := .venv/Scripts/python.exe
VENV_PIP := .venv/Scripts/pip.exe

.PHONY: install lint typecheck test demo-preprocessing-pipeline demo-ml-track demo-dl-track clean

install:
	@if [ ! -f "$(VENV_PYTHON)" ]; then \
		$(PYTHON_BOOTSTRAP) -m venv .venv; \
	fi
	"$(VENV_PYTHON)" -m pip install --upgrade pip
	"$(VENV_PIP)" install -e ".[dev]"

lint:
	"$(VENV_PYTHON)" -m ruff check .
	"$(VENV_PYTHON)" -m ruff format --check .

typecheck:
	"$(VENV_PYTHON)" -m mypy

test:
	"$(VENV_PYTHON)" -m pytest

demo-preprocessing-pipeline:
	@echo "Day 1 placeholder: implemented on Day 3."

demo-ml-track:
	@echo "Day 1 placeholder: implemented on Day 5."

demo-dl-track:
	@echo "Day 1 placeholder: implemented on Day 8."

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist htmlcov .coverage
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
