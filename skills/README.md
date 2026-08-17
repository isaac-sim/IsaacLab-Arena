# Isaac Lab-Arena Agent Skills

This directory contains the canonical sources for Arena's repository-owned skills. The audience folders organize ownership and validation; they do not change when a skill is available.

## Catalog

Developer skills:

- `developer/commit-and-pr/`: create Arena-conformant commits and pull requests.
- `developer/dev-container/`: bootstrap and maintain the contributor Docker environment.
- `developer/run-tests/`: run Arena's three-phase pytest suite.

User skills:

- `user/setup-arena/`: install and verify Arena with native uv or Docker.

## Discovery

Codex scans `.agents/skills/<name>/`. Each entry there is a flat symlink to one canonical skill directory under `skills/`.

Claude Code scans `.claude/skills/<name>/`. The repository's `.claude/skills` symlink points to `.agents/skills`, so both tools use the same aliases.

When adding, removing, or renaming a skill:

1. Update the canonical directory under `skills/`.
2. Update its flat `.agents/skills/<name>` alias.
3. Update this catalog.
