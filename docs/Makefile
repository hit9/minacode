# Minimal makefile for Sphinx documentation.
SPHINXBUILD ?= sphinx-build
SOURCEDIR   = .
BUILDDIR    = _build

.PHONY: help html clean serve

help:
	@echo "make html   - build the HTML docs into $(BUILDDIR)/html"
	@echo "make serve  - build, then serve on http://localhost:8000"
	@echo "make clean  - remove the build directory"

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)"

serve: html
	@cd "$(BUILDDIR)/html" && python3 -m http.server
