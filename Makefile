# common
PYTHON ?= python
RUFF ?= $(PYTHON) -m ruff
PYTEST ?= $(PYTHON) -m pytest

BASE_YML := environment_base.yml
PIP_JSON := pip_list.json
FINAL_YML := environment_combined.yml
REQUIREMENTS := generated/requirements.txt
CI_YML := environment_ci.yml
EXCLUDE_CI := .github/exclude_for_ci.txt
MERGE_SCRIPT := src/photo_eval_env_manager/merge_envs.py
EXTRA_ARGS ?=

SYNC_ISSUES_SCRIPT := scripts/github/sync_issues.py
SYNC_ISSUES_YML ?= .github/issues/epics.yaml
SYNC_ISSUES_ENV ?= .env
GITHUB_REPO ?= akky0108/photo_insight
DOCKER_SYNC_SERVICE ?= app-ci

BASE_BRANCH ?= develop
REMOTE_NAME ?= origin
DELETE_REMOTE ?= true
TARGET_BRANCH ?=

DOCKER_COMPOSE ?= docker compose
SERVICE ?= app

ISSUE ?=
TYPE ?= fix
BRANCH ?=
TITLE ?=
BODY ?=
LABELS ?=
ASSIGNEE ?= @me

PIPELINE ?=
CONFIG ?= config/config.yaml
DATE ?=
RUN_ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help merge dry-run only-pip audit clean check-ci-env lint fmt fmt-check test \
        test-light test-heavy test-integration ci ci-light ci-full \
        run run-dry \
        docker-build docker-shell docker-ci docker-ci-light docker-test \
        docker-integration docker-lint docker-fmt-check docker-ci-gpu \
        sync-issues sync-issues-dry docker-sync-issues docker-sync-issues-dry \
        issue-new issue-start pr-create pr-draft branch-cleanup branch-cleanup-current \
        release-pr release-pr-draft \
        deploy-check deploy-prod \
        cur st lg

# help
help:
	@echo "make merge / fmt / lint / test / ci / run / docker / gh-*"

# env
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

# quality
fmt:
	$(RUFF) format src tests

fmt-check:
	$(RUFF) format --check src tests

lint:
	$(RUFF) check src tests

# test
test:
	$(PYTEST) -q tests -m "not integration and not heavy"

test-light:
	$(PYTEST) -q tests/unit -m "not heavy"

test-heavy:
	$(PYTEST) -q -m "heavy" --run-heavy

test-integration:
	$(PYTEST) -q tests/integration -m "not heavy"

# ci
ci: fmt-check lint test

ci-light: fmt-check lint test-light

ci-full: fmt-check lint test test-integration

# run
run:
	@if [ -z "$(PIPELINE)" ]; then \
		echo "Usage: make run PIPELINE=... DATE=YYYY-MM-DD"; \
		exit 1; \
	fi
	@if [ -z "$(DATE)" ]; then \
		echo "Usage: make run PIPELINE=... DATE=YYYY-MM-DD"; \
		exit 1; \
	fi
	$(PYTHON) -m photo_insight.cli.run_batch \
		--pipeline "$(PIPELINE)" \
		--config "$(CONFIG)" \
		--date "$(DATE)" \
		$(RUN_ARGS)

run-dry:
	$(MAKE) run RUN_ARGS="--dry-run"

# docker
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

docker-ci-gpu:
	$(DOCKER_COMPOSE) run --rm app-gpu make ci

# github
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

issue-new:
	./scripts/github/issue-new.sh

issue-start:
	./scripts/github/start_issue.sh $(ISSUE) $(TYPE)

pr-create:
	./scripts/github/create_pr.sh

pr-draft:
	./scripts/github/create_pr.sh --draft

branch-cleanup:
	./scripts/github/cleanup_branch.sh "$(TARGET_BRANCH)"

branch-cleanup-current:
	./scripts/github/cleanup_branch.sh

# release
release-pr:
	./scripts/github/create_release_pr.sh

release-pr-draft:
	./scripts/github/create_release_pr.sh --draft

# deploy
deploy-check:
	@echo "[INFO] deploy check"

deploy-prod:
	docker compose -f compose.prod.yaml --env-file config/.env run --rm photo_insight \
		$(PYTHON) -m photo_insight.cli.run_batch \
		--pipeline "$(PIPELINE)" \
		--config /work/config/config.yaml \
		--date "$(DATE)"

# debug
cur:
	git branch --show-current

st:
	git status -sb

lg:
	git log --oneline --graph --decorate -10