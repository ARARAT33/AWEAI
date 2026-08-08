.PHONY: install dev test lint build clean serve chat

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .

install-all:
	$(PYTHON) -m pip install -e ".[all,dev]"

serve:
	aweai serve

chat:
	aweai chat

test:
	$(PYTHON) -m pytest

test-verbose:
	$(PYTHON) -m pytest -v

lint:
	$(PYTHON) -m compileall -q aweai && echo "compile OK"
	$(PYTHON) -m flake8 aweai tests --max-line-length=110 --extend-ignore=E203,W503 2>/dev/null || echo "flake8 not installed (optional)"

build:
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
