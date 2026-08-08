.PHONY: install install-all install-ui install-ml install-rag dev test serve chat train finetune doctor build-apk lint clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e .

install-all:
	$(PYTHON) -m pip install -e ".[all,dev]"

install-ui:
	$(PYTHON) -m pip install -e ".[ui]"

install-ml:
	$(PYTHON) -m pip install -e ".[ml]"

install-rag:
	$(PYTHON) -m pip install -e ".[rag]"

dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

serve:
	aweai serve

chat:
	aweai chat

train:
	aweai train --data $(DATA)

finetune:
	aweai finetune --base $(BASE) --data $(DATA)

doctor:
	aweai doctor

build-apk:
	bash scripts/build_apk.sh

lint:
	$(PYTHON) -m compileall -q aweai examples tests && $(PYTHON) -m pytest

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache build dist *.egg-info
