---
name: publish-release
description: Publishes an Isaac Lab-Arena release from an existing release branch, including released documentation, an immutable tag, and a GitHub Release. Use after the pre-release changes are merged and the Arena release branch exists.
---

# Publish release

Run this workflow only after `release/<version>` has been created from the reviewed pre-release
commit. The release branch is a moving maintenance line; the tag is the immutable record of what
was published. Never move or reuse a published tag. Publish a patch release for later fixes.

## 1. Prepare the release branch

1. Set `VERSION`, `TAG=v$VERSION`, and `RELEASE_BRANCH=release/$VERSION`.
2. Fetch branches and tags. Confirm that `TAG` does not exist locally or remotely:

   ```bash
   git fetch origin --tags
   test -z "$(git tag --list "$TAG")"
   test -z "$(git ls-remote --tags origin "refs/tags/$TAG")"
   ```

3. Start a short-lived maintainer branch from `RELEASE_BRANCH`.
4. Remove the development-version warning at the top of `docs/index.rst`. Do not remove the warning
   from `main`; it remains accurate there.
5. Build the documentation and run pre-commit:

   ```bash
   make -C docs html
   pre-commit run --all-files
   ```

6. Open a PR back to `RELEASE_BRANCH`. After it merges, use that merge commit as `RELEASE_COMMIT`.
   Confirm it belongs to the release branch and that the versioned documentation is published:

   ```bash
   git merge-base --is-ancestor "$RELEASE_COMMIT" "origin/$RELEASE_BRANCH"
   ```

For a historical release without an authoritative tag, confirm the shipped commit from a build,
announcement, or maintainer record. Do not infer it from the current release-branch tip.

## 2. Prepare GitHub notes

The matching section in `docs/pages/references/release_notes.rst` is canonical. Copy only that
section to `/tmp/release-notes-$VERSION.rst`, then convert it:

```bash
pandoc --from=rst --to=gfm \
  "/tmp/release-notes-$VERSION.rst" \
  --output "/tmp/release-notes-$VERSION.md"
```

Rewrite relative links to `https://isaac-sim.github.io/IsaacLab-Arena/release/$VERSION/`. Check the
rendered Markdown, issue links, migration-guide link, and versioned documentation links.

## 3. Tag and publish

```bash
git tag "$TAG" "$RELEASE_COMMIT"
git push origin "$TAG"

gh release create "$TAG" \
  --repo isaac-sim/IsaacLab-Arena \
  --verify-tag \
  --title "Isaac Lab-Arena $TAG" \
  --notes-file "/tmp/release-notes-$VERSION.md" \
  --draft

gh release view "$TAG" --repo isaac-sim/IsaacLab-Arena --web
gh release edit "$TAG" --repo isaac-sim/IsaacLab-Arena --draft=false
```

Verify that the tag resolves to `RELEASE_COMMIT` and that the GitHub Release and versioned
documentation are public.

## 4. Update main's stable-documentation pointer

Create a separate branch from the latest `main`. Update the development warning in
`docs/index.rst` so its tested-version text and URL point to `release/$VERSION`, build the docs, and
open a separate PR to `main`. Do not include this main-only pointer update in the release tag.

## 5. Verify the release

Confirm:

- The release branch no longer shows the development-version warning.
- The immutable tag resolves to `RELEASE_COMMIT`.
- The GitHub Release and versioned documentation are public.
- Release-note and migration-guide links target the released documentation.
- A separate PR updates `main` to point at the new stable documentation.

## Automation follow-up

A future script or manually triggered workflow should accept a version and target commit, reject an
existing tag, extract and convert the matching RST section, rewrite documentation links, create and
push the immutable tag, and create a draft GitHub Release. Keep final publication as an explicit
human approval.
