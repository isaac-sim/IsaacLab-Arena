#!/bin/bash
# Helpers that name the git commit and branch a Docker image is built from.

# Print the HEAD SHA of the checkout containing `$1` (default: cwd), suffixed
# with "-dirty" when the tree has uncommitted changes, or "unknown" outside git.
git_stamp_sha() {
    local dir=${1:-.} sha
    sha=$(git -C "$dir" rev-parse HEAD 2>/dev/null) || { echo unknown; return; }
    if [ -n "$(git -C "$dir" status --porcelain 2>/dev/null)" ]; then
        sha="$sha-dirty"
    fi
    echo "$sha"
}

# Print the current branch name of the checkout containing `$1` (default: cwd),
# or "unknown" outside git or on a detached HEAD.
git_stamp_branch() {
    local branch
    branch=$(git -C "${1:-.}" rev-parse --abbrev-ref HEAD 2>/dev/null) || { echo unknown; return; }
    if [ "$branch" = HEAD ]; then
        branch=unknown
    fi
    echo "$branch"
}
