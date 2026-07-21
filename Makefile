PYTHON ?= python3
VERSION := $(shell $(PYTHON) -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
DIST_FILES := dist/minacode-$(VERSION)*

.PHONY: lint test clean-dist build publish-check publish

lint:
	$(PYTHON) -m ruff check minacode
	$(PYTHON) -m ruff format --check minacode

test:
	$(PYTHON) -m compileall -q minacode
	$(PYTHON) -m pytest

clean-dist:
	rm -rf dist

build: clean-dist
	uv build

publish-check: build
	uv publish --dry-run $(DIST_FILES)

publish: build
	uv publish $(DIST_FILES)
