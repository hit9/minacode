PYTHON ?= python3
VERSION := $(shell $(PYTHON) -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
DIST_FILES := dist/nanocode_cli-$(VERSION)*

.PHONY: test clean-dist build publish-check publish

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
