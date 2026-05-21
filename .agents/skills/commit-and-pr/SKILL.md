---
name: commit-and-pr
description: Creates Arena-conformant commits and pull requests. Enforces DCO sign-off (git commit -s), no AI attribution lines (no Co-Authored-By, no Generated-with-Claude footers), <username>/<feature-desc> branch naming, separate new commits rather than --amend when iterating on PR feedback, and main as the default base branch. Use when the user asks to commit, stage changes, push the current branch, or open a pull request.
disable-model-invocation: true
allowed-tools: Bash(git *) Bash(gh pr create *) Bash(gh pr view *)
---

# Commit and Pull Request

Creates commits and PRs that follow Arena's contributor guidelines.

## Commit

1. Run the `pre-commit-check` skill (or `pre-commit run --all-files`) until the working tree is clean. Do not commit through hook failures.
2. Stage **specific files** rather than using `git add -A` — this avoids accidentally committing dotfiles, credentials, or large untracked artifacts.
3. Sign off the commit (DCO is required by Arena):

   ```bash
   git commit -s -m "<subject>"
   ```

4. **Subject line:** imperative mood, ~50 characters, no trailing period.
   - Good: `Fix attribute access on wrapped env`
   - Bad: `Fixed the attribute access on the wrapped env.` (past tense, trailing period)
5. **Body** (when needed): blank line after the subject, wrap at 72 chars, explain *what* and *why* — not *how* (the diff shows that).
6. **No AI attribution lines.** Do not include `Co-Authored-By: Claude...`, `Generated with Claude Code`, or similar footers. The DCO sign-off is the only required trailer.

Use a heredoc for multi-paragraph messages so quoting stays clean:

```bash
git commit -s -m "$(cat <<'EOF'
<subject>

<body paragraph 1>

<body paragraph 2>
EOF
)"
```

## Branch

Branch naming: `<username>/<feature-desc>` — e.g. `cvolk/feature-video-recording`, `cvolk/refactor-no-unwrap`, `mimil/create-arena-external-workflow-skill`.

Do not use `feature/...`, `fix/...`, or `docs/...` prefixes.

## Pull request

1. Push the branch (set upstream on first push):

   ```bash
   git push -u origin <branch-name>
   ```

2. Open the PR against `main` (the active branch):

   ```bash
   gh pr create --base main \
     --title "<short imperative subject>" \
     --body "<summary + test plan>"
   ```

3. PR title follows the same rules as a commit subject: imperative, ~70 chars max, no trailing period.

## Iterating on review feedback

When addressing review comments, **add new commits**. Do not `--amend` and force-push, because reviewers need to see each round of changes as a discrete commit.

```bash
# good
git commit -s -m "Address review: rename foo to bar"
git push

# avoid
git commit --amend
git push --force
```

The exception: if a single commit is *purely* a typo fix in your own as-yet-unmerged work and no one has reviewed it, `--amend` is fine.

## Verify

Before declaring the work done:

```bash
gh pr view --json title,baseRefName,statusCheckRollup
```

Confirm:
- `baseRefName` is `main`.
- Title has no trailing period and is in imperative mood.
- CI checks are running or green. Investigate any failures before requesting review.
