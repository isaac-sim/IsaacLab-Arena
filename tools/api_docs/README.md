# Generated API reference

Reference documentation generated from the docstrings of the first-party packages (`isaaclab_arena*`).

- Generator: [sphinx-autoapi](https://sphinx-autoapi.readthedocs.io/), which parses the sources statically.
  Nothing is imported, so neither Isaac Sim nor the Docker container is needed.
- Source links: `source_links.py` appends `Source: path:first-last` (linked to GitHub) to every module, class,
  function, method and attribute.
- Docstring coverage check: `python3 tools/api_docs/find_docstring_gaps.py [--summary] [paths...]` lists public
  modules, classes and functions that lack a docstring or an `Args:` section.

## Build

```bash
python3.12 -m venv /tmp/venv_apidocs
/tmp/venv_apidocs/bin/pip install -r tools/api_docs/requirements.txt
tools/api_docs/build.sh /tmp/venv_apidocs/bin/sphinx-build /tmp/api_docs_html /tmp/api_docs_text
```

`build.sh` produces an HTML tree (for people) and a plain-text tree (for grepping and for LLM lookups). Set
`API_DOCS_SOURCE_URL` to point the source links at a different branch or commit.
