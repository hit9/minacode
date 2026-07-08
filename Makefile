PYTHON ?= python3
VERSION := $(shell $(PYTHON) -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
DIST_FILES := dist/nanocode_cli-$(VERSION)*

.PHONY: lint test clean-dist build publish-check publish

lint:
	$(PYTHON) -m ruff check nanocode.py
	$(PYTHON) -m ruff format --check nanocode.py

test:
	$(PYTHON) -m py_compile nanocode.py
	$(PYTHON) -m pytest

clean-dist:
	rm -rf dist

build: clean-dist
	uv build

publish-check: build
	uv publish --dry-run $(DIST_FILES)

publish: build
	uv publish $(DIST_FILES)
