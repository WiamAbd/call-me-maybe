# ===============================
# Project Makefile
# ===============================

PYTHON = python
UV = uv

SRC = src

INPUT = data/input/function_calling_tests.json
FUNCTIONS = data/input/functions_definition.json
OUTPUT = data/output/function_calling_results.json


# ===============================
# Phony targets
# ===============================
.PHONY: install run debug clean lint lint-strict test format


# ===============================
# Install dependencies
# ===============================
install:
	@if command -v $(UV) >/dev/null 2>&1; then \
		UV_CACHE_DIR=.uv_cache $(UV) sync; \
	else \
		echo "uv not found."; \
		echo "Installing manually with pip..."; \
		pip install numpy pydantic flake8 mypy; \
	fi


# ===============================
# Run project
# ===============================
run:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) run $(PYTHON) -m $(SRC) \
			--functions_definition $(FUNCTIONS) \
			--input $(INPUT) \
			--output $(OUTPUT); \
	else \
		$(PYTHON) -m $(SRC) \
			--functions_definition $(FUNCTIONS) \
			--input $(INPUT) \
			--output $(OUTPUT); \
	fi


# ===============================
# Debug mode
# ===============================
debug:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) run $(PYTHON) -m pdb -m $(SRC) \
			--functions_definition $(FUNCTIONS) \
			--input $(INPUT) \
			--output $(OUTPUT); \
	else \
		$(PYTHON) -m pdb -m $(SRC) \
			--functions_definition $(FUNCTIONS) \
			--input $(INPUT) \
			--output $(OUTPUT); \
	fi


# ===============================
# Clean cache files
# ===============================
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete


# ===============================
# Lint
# ===============================
lint:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) run flake8 .; \
		$(UV) run mypy . \
			--warn-return-any \
			--warn-unused-ignores \
			--ignore-missing-imports \
			--disallow-untyped-defs \
			--check-untyped-defs; \
	else \
		flake8 .; \
		mypy . \
			--warn-return-any \
			--warn-unused-ignores \
			--ignore-missing-imports \
			--disallow-untyped-defs \
			--check-untyped-defs; \
	fi


# ===============================
# Strict lint
# ===============================
lint-strict:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) run flake8 .; \
		$(UV) run mypy . --strict; \
	else \
		flake8 .; \
		mypy . --strict; \
	fi


# ===============================
# Optional tests
# ===============================
test:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) run pytest; \
	else \
		pytest; \
	fi


# ===============================
# Optional formatting
# ===============================
format:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) run black .; \
	else \
		black .; \
	fi