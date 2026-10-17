---
name: pre-release
description: Prepares Isaac Lab-Arena release documentation on main before a release branch is created. Use when finalizing release notes, versioning the next-release migration guide, or preparing main for an Arena release branch.
---

# Pre-release

Run this workflow on a branch from the latest `main`. Complete and merge its PR before creating
`release/<version>`.

Each release has a moving `release/<major>.<minor>.<patch>` maintenance branch, an immutable
`v<major>.<minor>.<patch>` tag, a GitHub Release, a section in
`docs/pages/references/release_notes.rst`, and a version-specific migration guide when users must
act. The release branch provides versioned documentation; the tag records the exact published
commit.

## 1. Establish the release

1. Set `VERSION`, `PREVIOUS_VERSION`, and `RELEASE_BRANCH=release/$VERSION`.
2. Fetch `main` and tags. Confirm the release version and compatibility pins.
3. Confirm that `RELEASE_BRANCH` and `v$VERSION` do not already exist unless this is an explicitly
   approved historical-release workflow.

## 2. Finalize release documentation

1. Add or finalize the `v$VERSION` section in
   `docs/pages/references/release_notes.rst`. Include user-facing features, fixes, limitations, and
   breaking changes.
2. If `docs/pages/references/migration_to_next_release.rst` exists, rename it to
   `migration_v<previous_version>_to_v<version>.rst`, replacing dots with underscores in both
   versions. For example, use `migration_v0_3_1_to_v0_4_0.rst` for a v0.3.1-to-v0.4.0 migration.
   Update its title, release-note link, and Sphinx navigation.
3. If the release requires user action but no generic guide exists, create the versioned migration
   guide. Include the old form, replacement, and compatibility window.
4. Update package, installation, compatibility, and landing-page version references where needed.
5. Keep `docs/pages/references/release_notes.rst` as the canonical release history; do not add a
   separate changelog.

## 3. Validate and merge to main

Run host-side documentation and repository checks:

```bash
make -C docs html
pre-commit run --all-files
```

Open a PR to `main`. After it merges, record the merge commit and create
`release/$VERSION` from that exact commit. Stop here; tagging and GitHub Release publication belong
to the `publish-release` workflow.

Before handing off, confirm:

- Release notes and any migration guide are complete and linked.
- Package and documentation versions are correct.
- Documentation builds without warnings.
- The release branch is created from the reviewed pre-release merge commit.
