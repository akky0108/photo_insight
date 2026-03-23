# =========================
# Variables
# =========================
PYTHON ?= python
RUFF ?= $(PYTHON) -m ruff
PYTEST ?= $(PYTHON) -m pytest

BASE_YML := environment_base.yml
PIP_JSON := pip_list.json
FINAL_YML := environment_combined.yml

# ★ ここが事故の元。生成物は generated/ に逃がす
REQUIREMENTS := generated/requirements.txt

CI_YML := environment_ci.yml
EXCLUDE_CI := .github/exclude_for_ci.txt

MERGE_SCRIPT := src/photo_eval_env_manager/merge_envs.py

# Optional extra args for merge script
EXTRA_ARGS ?=

# =========================
# GitHub issue sync
# =========================
SYNC_ISSUES_SCRIPT := scripts/github/sync_issues.py
SYNC_ISSUES_YML ?= .github/issues/epics.yaml
SYNC_ISSUES_ENV ?= .env
GITHUB_REPO ?= akky0108/photo_insight
DOCKER_SYNC_SERVICE ?= app-ci

.DEFAULT_GOAL := help

.PHONY: help merge dry-run only-pip audit clean check-ci-env lint fmt fmt-check test \
        test-light test-heavy test-integration ci ci-light ci-full \
        docker-build docker-shell docker-ci docker-ci-light docker-test \
        docker-integration docker-lint docker-fmt-check docker-ci-gpu \
        sync-issues sync-issues-dry docker-sync-issues docker-sync-issues-dry

help:
	@echo "Available commands:"
	@echo ""
	@echo "  make merge                   Merge Conda + pip environments (outputs .yml and .txt)"
	@echo "  make dry-run                 Check merge result without writing files"
	@echo "  make only-pip                Extract pip-only dependencies to $(REQUIREMENTS)"
	@echo "  make audit                   Audit security issues in $(REQUIREMENTS) using pip-audit"
	@echo "  make clean                   Delete generated environment files"
	@echo "  make check-ci-env            [CI] Run merge and check for unexpected changes"
	@echo "  make fmt                     Format code with ruff"
	@echo "  make fmt-check               Check formatting (no changes)"
	@echo "  make lint                    Run ruff lint on src/ and tests/"
	@echo "  make test                    Run pytest on tests/"
	@echo "  make test-light              Run light unit tests"
	@echo "  make test-heavy              Run heavy tests"
	@echo "  make test-integration        Run integration tests"
	@echo "  make ci                      Run fmt-check + lint + test (CI-equivalent)"
	@echo "  make ci-light                Run fmt-check + lint + test-light"
	@echo "  make ci-full                 Run fmt-check + lint + test + test-integration"
	@echo "  make sync-issues-dry         Run GitHub issue sync in dry-run mode"
	@echo "  make sync-issues             Sync issues.yml to GitHub Issues"
	@echo "  make docker-sync-issues-dry  Run issue sync in Docker (dry-run)"
	@echo "  make docker-sync-issues      Run issue sync in Docker"

merge:
	@if [ ! -f $(PIP_JSON) ]; then \
		echo "Error: $(PIP_JSON) not found. Run 'pip list --format=json > $(PIP_JSON)' first."; \
		exit 1; \
	fi
	@mkdir -p $(dir $(REQUIREMENTS))
	$(PYTHON) $(MERGE_SCRIPT) --base $(BASE_YML) --pip-json $(PIP_JSON) \
		--final $(FINAL_YML) --requirements $(REQUIREMENTS) \
		--ci $(CI_YML) --exclude-for-ci $(EXCLUDE_CI) $(EXTRA_ARGS)

dry-run:
	$(PYTHON) $(MERGE_SCRIPT) --base $(BASE_YML) --pip-json $(PIP_JSON) \
		--final $(FINAL_YML) --requirements $(REQUIREMENTS) \
		--ci $(CI_YML) --exclude-for-ci $(EXCLUDE_CI) --dry-run

only-pip:
	@mkdir -p $(dir $(REQUIREMENTS))
	$(PYTHON) $(MERGE_SCRIPT) --pip-json $(PIP_JSON) \
		--requirements $(REQUIREMENTS) --only-pip

audit:
	@mkdir -p $(dir $(REQUIREMENTS))
	$(PYTHON) $(MERGE_SCRIPT) --pip-json $(PIP_JSON) \
		--requirements $(REQUIREMENTS) --only-pip --audit
	pip-audit -r $(REQUIREMENTS) || true

clean:
	rm -rf generated

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
	$(PYTEST) -q tests -m "not integration and not heavy"

test-light:
	$(PYTEST) -q tests/unit -m "not heavy"

test-heavy:
	$(PYTEST) -q -m "heavy" --run-heavy

test-integration:
	$(PYTEST) -q tests/integration -m "not heavy"

ci: fmt-check lint test

ci-light: fmt-check lint test-light

ci-full: fmt-check lint test test-integration

# =========================
# GitHub issue sync
# =========================
sync-issues-dry:
	$(PYTHON) $(SYNC_ISSUES_SCRIPT) \
		--repo $(GITHUB_REPO) \
		--issues-yml $(SYNC_ISSUES_YML) \
		--env-file $(SYNC_ISSUES_ENV) \
		--dry-run

sync-issues:
	$(PYTHON) $(SYNC_ISSUES_SCRIPT) \
		--repo $(GITHUB_REPO) \
		--issues-yml $(SYNC_ISSUES_YML) \
		--env-file $(SYNC_ISSUES_ENV)

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

docker-integration:
	$(DOCKER_COMPOSE) run --rm $(SERVICE) make test-integration

docker-lint:
	$(DOCKER_COMPOSE) run --rm $(SERVICE) make lint

docker-fmt-check:
	$(DOCKER_COMPOSE) run --rm $(SERVICE) make fmt-check

docker-sync-issues-dry:
	$(DOCKER_COMPOSE) run --rm $(DOCKER_SYNC_SERVICE) \
		$(PYTHON) $(SYNC_ISSUES_SCRIPT) \
		--repo $(GITHUB_REPO) \
		--issues-yml $(SYNC_ISSUES_YML) \
		--env-file $(SYNC_ISSUES_ENV) \
		--dry-run

docker-sync-issues:
	$(DOCKER_COMPOSE) run --rm $(DOCKER_SYNC_SERVICE) \
		$(PYTHON) $(SYNC_ISSUES_SCRIPT) \
		--repo $(GITHUB_REPO) \
		--issues-yml $(SYNC_ISSUES_YML) \
		--env-file $(SYNC_ISSUES_ENV)

# GPU版（必要なときだけ）
docker-ci-gpu:
	$(DOCKER_COMPOSE) run --rm app-gpu make ci