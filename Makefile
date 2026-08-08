# AWEAI Makefile — model factory automation targets

PYTHON ?= python3
PIP ?= pip

.PHONY: help install install-all test autotest serve clean lint smoke train-demo \
        export-demo check-hf-free android-build docs-build

help: ## Show all targets
        @grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install AWEAI (core, no HF)
        $(PIP) install -e .

install-all: ## Install everything (torch, sklearn, UI, onnx)
        $(PIP) install -e ".[all]"

test: ## Run the unit test suite
        $(PYTHON) -m pytest -q

autotest: ## Run the full system autotest (deps, imports, smoke-train all model types, RAG, actions, i18n, UI, CLI)
        $(PYTHON) -m aweai autotest

serve: ## Launch the browser UI (port 8888, auto +1)
        $(PYTHON) -m aweai serve

cli: ## Show CLI help
        $(PYTHON) -m aweai --help

hardware: ## Show detected hardware
        $(PYTHON) -m aweai hardware

types: ## List model types
        $(PYTHON) -m aweai types

lint: ## Compile-check all python files
        $(PYTHON) -m compileall -q aweai tests examples scripts

clean: ## Remove build artifacts
        rm -rf build dist *.egg-info .pytest_cache __pycache__ aweai/__pycache__

smoke-train: ## Smoke-train every model type quickly
        $(PYTHON) -m aweai autotest --quick --no-ui

train-demo: ## Train a demo MLP on synthetic XOR
        $(PYTHON) examples/train_demo.py

export-demo: ## Export all zoo models to every format
        $(PYTHON) scripts/export_all.py

check-hf-free: ## Ensure no Hugging Face references remain
        @grep -rIl -E "transformers|datasets|huggingface|sentence-transformers|peft" --include="*.py" --include="*.toml" --include="*.txt" . || echo "✓ No Hugging Face references"

android-build: ## Build the Android APK (buildozer)
        bash scripts/build_apk.sh

docs-build: ## Build docs index
        @echo "Docs live in docs/ — see docs/README.md"
