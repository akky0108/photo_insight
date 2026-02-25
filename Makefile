# =========================
# Variables
# =========================
PYTHON ?= python
RUFF ?= $(PYTHON) -m ruff
PYTEST ?= $(PYTHON) -m pytest

BASE_YML := environment_base.yml
PIP_JSON := pip_list.json
FINAL_YML := environment_combined.yml
REQUIREMENTS := requirements.txt
CI_YML := environment_ci.yml
EXCLUDE_CI := .github/exclude_for_ci.txt

MERGE_SCRIPT := src/photo_eval_env_manager/merge_envs.py

# Optional extra args for merge script
EXTRA_ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help merge dry-run only-pip audit clean check-ci-env lint fmt fmt-check test ci

help:
	@echo "Available commands:"
	@echo ""
	@echo "  make merge           Merge Conda + pip environments (outputs .yml and .txt)"
	@echo "  make dry-run         Check merge result without writing files"
	@echo "  make only-pip        Extract pip-only dependencies to requirements.txt"
	@echo "  make audit           Audit security issues in requirements.txt using pip-audit"
	@echo "  make clean           Delete generated environment files"
	@echo "  make check-ci-env    [CI] Run merge and check for unexpected changes"
	@echo "  make fmt             Format code with ruff"
	@echo "  make fmt-check       Check formatting (no changes)"
	@echo "  make lint            Run ruff lint on src/ and tests/"
	@echo "  make test            Run pytest on tests/"
	@echo "  make ci              Run fmt-check + lint + test (CI-equivalent)"

merge:
	@if [ ! -f $(PIP_JSON) ]; then \
		echo "Error: $(PIP_JSON) not found. Run 'pip list --format=json > $(PIP_JSON)' first."; \
		exit 1; \
	fi
	$(PYTHON) $(MERGE_SCRIPT) --base $(BASE_YML) --pip-json $(PIP_JSON) \
		--final $(FINAL_YML) --requirements $(REQUIREMENTS) \
		--ci $(CI_YML) --exclude-for-ci $(EXCLUDE_CI) $(EXTRA_ARGS)

dry-run:
	$(PYTHON) $(MERGE_SCRIPT) --base $(BASE_YML) --pip-json $(PIP_JSON) \
		--final $(FINAL_YML) --requirements $(REQUIREMENTS) \
		--ci $(CI_YML) --exclude-for-ci $(EXCLUDE_CI) --dry-run

only-pip:
	$(PYTHON) $(MERGE_SCRIPT) --pip-json $(PIP_JSON) \
		--requirements $(REQUIREMENTS) --only-pip

audit:
	$(PYTHON) $(MERGE_SCRIPT) --pip-json $(PIP_JSON) \
		--requirements $(REQUIREMENTS) --only-pip --audit
	pip-audit -r $(REQUIREMENTS) || true

check-ci-env:
	$(MAKE) merge
	git diff --exit-code $(FINAL_YML) $(REQUIREMENTS) $(CI_YML)

# =========================
# Code quality
# =========================
fmt:
	$(RUFF) format src tests

fmt-check:
	$(RUFF) format --check src tests

lint:
	$(RUFF) check src tests

test:
	$(PYTEST) -q tests

test-light:
	$(PYTEST) -q tests/unit -m "not heavy"

test-heavy:
	$(PYTEST) -q -m "heavy" --run-heavy

test-integration:
	$(PYTEST) -q tests/integration -m "integration" --run-heavy

ci: fmt-check lint test
ci-light: fmt-check lint test-light

# ---- Docker helpers ----
DOCKER_COMPOSE ?= docker compose
SERVICE ?= app

docker-build:
	$(DOCKER_COMPOSE) build $(SERVICE)

docker-shell:
	$(DOCKER_COMPOSE) run --rm $(SERVICE) bash

docker-ci:
	$(DOCKER_COMPOSE) run --rm $(SERVICE) make ci

docker-ci-light:
	$(DOCKER_COMPOSE) run --rm $(SERVICE) make ci-light

docker-test:
	$(DOCKER_COMPOSE) run --rm $(SERVICE) make test

docker-lint:
	$(DOCKER_COMPOSE) run --rm $(SERVICE) make lint

docker-fmt-check:
	$(DOCKER_COMPOSE) run --rm $(SERVICE) make fmt-check

# GPU版（必要なときだけ）
docker-ci-gpu:
	$(DOCKER_COMPOSE) run --rm app-gpu make ci