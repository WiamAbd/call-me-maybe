# ===============================
# Project Makefile
# ===============================

PYTHON = python
UV = uv
SRC = src

# ===============================
# Phony targets
# ===============================
.PHONY: install run debug clean lint lint-strict

# ===============================
# Install dependencies
# ===============================
install:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) sync; \
	else \
		echo "uv not found, skipping install"; \
	fi

# ===============================
# Run project
# ===============================
run:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) run $(PYTHON) -m $(SRC); \
	else \
		$(PYTHON) -m $(SRC); \
	fi

# ===============================
# Debug mode
# ===============================
debug:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) run $(PYTHON) -m pdb -m $(SRC); \
	else \
		$(PYTHON) -m pdb -m $(SRC); \
	fi

# ===============================
# Clean cache
# ===============================
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# ===============================
# Lint (flake8 + mypy)
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