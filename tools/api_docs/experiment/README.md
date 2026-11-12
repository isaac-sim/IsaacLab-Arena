# Experiment: generated API docs vs. source lookups

Question: how much faster / cheaper is it for an LLM agent to answer development lookup questions with a
tool-generated API reference than by searching the source?

## Design

- 20 questions (`questions.md`), ground truth in `answers.md`.
- Each question is asked to a **fresh agent with an empty context**, once per arm, so 40 runs.
  - **Arm A (docs)**: told where the generated API reference (plain-text build) is and that it may also open source
    files via the `Source:` paths.
  - **Arm B (source only)**: told only the repository path; the docs are not on its path and it is told to stay in
    the repository.
- Both arms use the same snapshot of the repository (docstrings included), checked out detached at the commit that
  contains the docstring work, **without** `tools/api_docs/` (so no config or experiment material is visible).
  The docs are built outside that checkout.
- Same model for every run. Runs are executed in batches of parallel agents, so wall times carry some noise.
- Measured per run (reported by the harness): total tokens, tool calls, wall time. Correctness is graded
  against `answers.md` as correct / partial / wrong.

## Steps to reproduce

```bash
# 1. Snapshot the repo at the docstring commit, without the tooling directory.
mkdir /tmp/arena_snapshot
git archive 2c5139b53f | tar -x -C /tmp/arena_snapshot
rm -rf /tmp/arena_snapshot/tools/api_docs

# 2. Build the docs (text build is what arm A reads).
python3.12 -m venv /tmp/venv_apidocs && /tmp/venv_apidocs/bin/pip install -r tools/api_docs/requirements.txt
tools/api_docs/build.sh /tmp/venv_apidocs/bin/sphinx-build /tmp/api_docs_html /tmp/api_docs_text
```

## Agent prompt (arm A and arm B)

Common part, followed by one arm-specific paragraph and the question:

> You are answering a lookup question about a Python codebase. Answer from what you find in the files; do not guess.
> Be efficient: stop as soon as you can answer with evidence. Final message format, nothing else:
> `ANSWER:` the answer (concise, all requested facts), then `EVIDENCE:` the `path:line` locations you relied on.

- Arm A: "The repository is at `/tmp/arena_snapshot`. A generated API reference (text files, one per module; every entry
  ends with a `Source: path:lines` line giving the repo-relative source location) is at `/tmp/api_docs_text`. Start
  from the API reference; open source files only if you need to."
- Arm B: "The repository is at `/tmp/arena_snapshot`. Only read files inside that directory; do not look elsewhere on
  the filesystem."

## Results

See `results.md` (summary, per-question table and caveats) and `results.csv` (raw per-run numbers). The agents ran on Sonnet; the repository snapshot was the docstring commit `2c5139b53f`.
