# ===============================
# Project Makefile
# ===============================

PYTHON = python
UV = uv
SRC = src

# ===============================
# Install dependencies
# ===============================
install:
	$(UV) sync

# ===============================
# Run project
# ===============================
run:
	$(UV) run python -m $(SRC)

# ===============================
# Debug mode (with pdb)
# ===============================
debug:
	$(UV) run python -m pdb -m $(SRC)

# ===============================
# Clean cache files
# ===============================
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# ===============================
# Lint (flake8 + mypy)
# ===============================
lint:
	$(UV) run flake8 .
	$(UV) run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

# ===============================
# Strict lint (optional)
# ===============================
lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict