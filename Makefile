PYTHON ?= python3
VERSION := $(shell $(PYTHON) -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
DIST_FILES := dist/wizolt-$(VERSION)*

.PHONY: lint test clean-dist build publish-check publish

lint:
	$(PYTHON) -m ruff check wizolt
	$(PYTHON) -m ruff format --check wizolt

test:
	$(PYTHON) -m compileall -q wizolt
	$(PYTHON) -m pytest

# setuptools stages into build/ and never prunes it, so a module deleted from the source tree
# keeps shipping in every later wheel. Clear both trees before building.
clean-dist:
	rm -rf dist build

build: clean-dist
	uv build

publish-check: build
	uv publish --dry-run $(DIST_FILES)

publish: build
	uv publish $(DIST_FILES)
