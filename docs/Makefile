# Minimal makefile for Sphinx documentation.
SPHINXBUILD ?= sphinx-build
SOURCEDIR   = .
BUILDDIR    = _build

.PHONY: help html gettext locale-zh html-zh clean serve

help:
	@echo "make html   - build the HTML docs into $(BUILDDIR)/html"
	@echo "make gettext - extract translatable messages"
	@echo "make locale-zh - update the Simplified Chinese catalogs"
	@echo "make html-zh - build Simplified Chinese HTML into $(BUILDDIR)/html-zh_CN"
	@echo "make serve  - build, then serve on http://localhost:8000"
	@echo "make clean  - remove the build directory"

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

gettext:
	$(SPHINXBUILD) -b gettext "$(SOURCEDIR)" "$(BUILDDIR)/gettext" $(SPHINXOPTS)

locale-zh: gettext
	sphinx-intl update -d locale -p "$(BUILDDIR)/gettext" -l zh_CN

html-zh:
	$(SPHINXBUILD) -b html -D language=zh_CN "$(SOURCEDIR)" "$(BUILDDIR)/html-zh_CN" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)"

serve: html
	@cd "$(BUILDDIR)/html" && python3 -m http.server
