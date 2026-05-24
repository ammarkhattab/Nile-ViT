# =============================================================================
# Nile-ViT Makefile
# =============================================================================
# Works on Linux, macOS, and Windows (PowerShell with `make` installed,
# or run the underlying `uv` commands directly).
# =============================================================================

UV ?= uv

.PHONY: help
help:  ## Show this help message
	@echo "Nile-ViT targets:"
	@echo "  install         Install all deps (incl. dev) via uv sync --extra dev"
	@echo "  sync            Re-resolve and install from uv.lock"
	@echo "  lock            Update uv.lock without installing"
	@echo "  test            Run pytest"
	@echo "  test-fast       Run pytest, skip slow / gpu / network tests"
	@echo "  lint            ruff check + mypy"
	@echo "  fmt             ruff format"
	@echo "  fix             ruff format + ruff check --fix"
	@echo "  smoke           Run the CI smoke test path"
	@echo "  pre-commit      Install pre-commit hooks"
	@echo "  clean           Remove build artifacts and caches"
	@echo "  clean-data      Remove all cached data (DANGER: re-downloads needed)"

.PHONY: install
install:
	$(UV) sync --extra dev

.PHONY: sync
sync:
	$(UV) sync --extra dev --frozen

.PHONY: lock
lock:
	$(UV) lock

.PHONY: test
test:
	$(UV) run pytest

.PHONY: test-fast
test-fast:
	$(UV) run pytest -m "not slow and not gpu and not network"

.PHONY: lint
lint:
	$(UV) run ruff check .
	$(UV) run mypy nilevit

.PHONY: fmt
fmt:
	$(UV) run ruff format .

.PHONY: fix
fix:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

.PHONY: smoke
smoke: test-fast
	@echo "Smoke test passed."

.PHONY: pre-commit
pre-commit:
	$(UV) run pre-commit install
	$(UV) run pre-commit install --hook-type commit-msg

.PHONY: clean
clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

.PHONY: clean-data
clean-data:
	@echo "This will delete data/raw, data/interim, data/processed."
	@read -p "Proceed? [y/N] " ans && [ "$$ans" = "y" ] || exit 1
	rm -rf data/raw data/interim data/processed
