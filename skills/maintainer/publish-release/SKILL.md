---
name: publish-release
description: Publishes an Isaac Lab-Arena release from an existing release branch, including released documentation, an immutable tag, and a GitHub Release. Use after the pre-release changes are merged and the Arena release branch exists.
---

# Publish release

Run this workflow only after `release/<version>` has been created from the reviewed pre-release
commit. The release branch is a moving maintenance line; the tag is the immutable record of what
was published. Never move or reuse a published tag. Publish a patch release for later fixes.

## 1. Prepare the release and published documentation

1. Set `VERSION`, `TAG=v$VERSION`, and `RELEASE_BRANCH=release/$VERSION`.
2. Fetch the release branch, `main`, and tags. Confirm that `TAG` does not exist locally or
   remotely:

   ```bash
   git fetch origin main "$RELEASE_BRANCH" --tags
   test -z "$(git tag --list "$TAG")"
   test -z "$(git ls-remote --tags origin "refs/tags/$TAG")"
   ```

3. Start a short-lived documentation branch from `RELEASE_BRANCH`.
4. For a normal release, the branch already contains the release notes and migration guide merged
   by the pre-release workflow. For a historical release prepared before that workflow existed,
   copy the reviewed release-note section, matching migration guide, and toctree entry from `main`
   onto this branch. Include only guides applicable at `VERSION`; remove links to later migrations.
5. Remove the development-version warning at the top of `docs/index.rst`.
6. Build the documentation and run pre-commit:

   ```bash
   # Activate the documentation environment described in docs/README.md first.
   make -C docs SPHINXOPTS=-W html
   pre-commit run --all-files
   ```

7. Open the documentation PR against `RELEASE_BRANCH`.
8. In parallel, start a separate branch from the latest `main`. Update the development warning in
   `docs/index.rst` so its tested-version text and URL point to `release/$VERSION`, run the same
   checks, and open a PR to `main`.
9. Merge the release-branch PR first, then merge the `main` pointer PR. The push to `main` triggers
   the GitHub Pages workflow, which builds the updated release branch. Wait for that deployment and
   verify the versioned documentation before creating the tag or GitHub Release.
10. Fetch the merged branches. Record the release documentation merge as `RELEASE_DOCS_COMMIT`.
    For a normal release, use it as `RELEASE_COMMIT`. For a historical release, use the
    independently confirmed shipped commit as `RELEASE_COMMIT`; do not infer that commit from the
    current branch tip. Confirm that both commits belong to the release branch:

   ```bash
   git fetch origin main "$RELEASE_BRANCH"
   git merge-base --is-ancestor "$RELEASE_DOCS_COMMIT" "origin/$RELEASE_BRANCH"
   git merge-base --is-ancestor "$RELEASE_COMMIT" "origin/$RELEASE_BRANCH"
   ```

## 2. Prepare GitHub notes

The matching section in the synchronized release-branch
`docs/pages/references/release_notes.rst` is canonical. Copy only that section to
`/tmp/release-notes-$VERSION.rst`, then convert it:

```bash
pandoc --from=rst --to=gfm \
  "/tmp/release-notes-$VERSION.rst" \
  --output "/tmp/release-notes-$VERSION.md"
```

Pandoc renders Sphinx `:doc:` roles as inline code, not links. Replace every converted `:doc:`
span with a Markdown link under
`https://isaac-sim.github.io/IsaacLab-Arena/release/$VERSION/pages/`, including the matching
migration-guide link. Check the rendered Markdown, issue links, migration-guide link, and
versioned documentation links.

## 3. Tag and create the draft release

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
```

Verify that the tag resolves to `RELEASE_COMMIT` and that the GitHub Release remains a draft.

## 4. Review and publish the release

Stop after opening the draft release. Require the user to review the rendered title, notes, links,
tag, and target commit, then explicitly confirm that the release should be made public. Do not
publish based on an earlier general instruction to run this workflow.

Only after receiving that confirmation, publish the release:

```bash
gh release edit "$TAG" --repo isaac-sim/IsaacLab-Arena --draft=false
```

Verify that the GitHub Release and versioned documentation are public.

## 5. Verify the release

Confirm:

- The release branch no longer shows the development-version warning.
- The immutable tag resolves to `RELEASE_COMMIT`.
- The GitHub Release and versioned documentation are public.
- Release-note and migration-guide links target the released documentation.
- `main` points at the new stable documentation.
