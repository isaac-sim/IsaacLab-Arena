# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS       ?=
SPHINXBUILD      ?= sphinx-build
SOURCEDIR        = .
BUILDDIR         = _build


# Put it first so that "make" without argument is like "make help".
.PHONY: help Makefile
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


.PHONY: multi-docs
multi-docs:
	@sphinx-multiversion --verbose "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@cp _redirect/index.html $(BUILDDIR)/index.html


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)/current" $(SPHINXOPTS) $(O)
