---
name: draft-pr-review
description: Turns review findings from a local or agent-assisted code review into a pending GitHub pull request review that the reviewer edits and submits. Use when findings should go onto a PR as inline review comments, or when creating, changing, recreating or checking a pending (unsubmitted) review on the reviewer's behalf. Do not use for submitting, approving, or posting comments that are published immediately.
argument-hint: "<pr-number> <findings-file>"
---

# Draft a Pending PR Review

Deliver review findings as a **pending** GitHub review: inline comments on the right diff lines plus a summary. Only the reviewer can see a pending review. They edit and submit it themselves; the agent never submits. Every comment starts with a short line saying it was AI-drafted.

Creating, changing or deleting a pending review writes to GitHub under the reviewer's account. Do it only after the reviewer's explicit go-ahead for that PR. All commands run on the host, where `gh` is authenticated as the reviewer.

## The payload

Build this JSON, save it locally (e.g. `/tmp/<repo>-pr<N>-review/pending_review.json`), validate it, then send it.

```json
{
  "commit_id": "<PR head SHA>",
  "body": "_AI-drafted (<agent name>), reviewed by me._\n\n<short verdict>. <whether anything was run, e.g. 'Everything below comes from reading the code at `abc1234`; nothing was run.'>\n\n<index of the inline comments, most important first>\n\n<points that have no line in the diff>",
  "comments": [
    {"path": "pkg/builder.py", "line": 339, "side": "RIGHT",
     "body": "_AI-drafted (<agent name>), reviewed by me._\n\n**Blocking: <title>**\n\n<evidence, failure scenario, suggested fix>"},
    {"path": "pkg/builder.py", "start_line": 304, "line": 331, "start_side": "RIGHT", "side": "RIGHT",
     "body": "_AI-drafted (<agent name>), reviewed by me._\n\n**Design: <title>**\n\n..."},
    {"path": "tests/test_replay.py", "line": 253, "side": "LEFT",
     "body": "_AI-drafted (<agent name>), reviewed by me._\n\n**Question: <about deleted code>**\n\n..."}
  ]
}
```

- No `event` key: that is what makes the review PENDING. Any `event` value publishes it.
- The attribution line is the first line of the body and of every comment.
- `side: RIGHT` takes new-file line numbers of added or context lines; `side: LEFT` takes old-file line numbers of deleted lines. A range must stay inside one hunk. Points without a diff line go in the summary.
- Label each comment **Blocking**, **Design**, **Question** or **Suggestion (non-blocking)**, and order them that way.

## Steps

1. Account: `gh api user --jq .login` must be the reviewer's own account; pending reviews are only visible to their author.
2. Head SHA: `gh api repos/OWNER/REPO/pulls/N --jq .head.sha` → `commit_id`. Check each line reference in the findings with `git show <sha>:<path>`.
3. Existing pending review (GitHub allows one per user per PR): `gh api repos/OWNER/REPO/pulls/N/reviews --paginate --jq '.[] | select(.state=="PENDING") | .id'`. If one exists, ask the reviewer.
4. Existing discussion: read the review summaries, inline threads (`.../pulls/N/comments`) and top-level comments (`.../issues/N/comments`). If someone already raised a point, even partly, tell the reviewer and wait for their answer before posting.
5. Validate: `python3 -I skills/developer/draft-pr-review/validate_pending_review.py OWNER/REPO N payload.json`. Fix everything it reports.
6. Create: `gh api repos/OWNER/REPO/pulls/N/reviews --method POST --input payload.json > response.json`, then read `response.json`. Expect `"state": "PENDING"`.
7. Confirm anchors: `gh api repos/OWNER/REPO/pulls/N/reviews/ID/comments --jq '.[] | "\(.path): \(.diff_hunk | split("\n") | last)"'`. For pending comments `line` is null; the last `diff_hunk` line is the anchored line.
8. Hand over `https://github.com/OWNER/REPO/pull/N/files`. The reviewer reads and edits every comment, which is what makes "reviewed by me" true, then uses "Finish your review" → Comment, Request changes or Approve.

## Changing an existing pending review

1. Fetch the current review and its comments and compare them with the saved payload. If they differ, the reviewer edited them in the UI: ask before overwriting anything.
2. Summary: `gh api repos/OWNER/REPO/pulls/N/reviews/ID --method PUT --input body.json` with `{"body": "..."}`. This works while the review is pending.
3. Inline comments: GitHub's docs don't say whether `PATCH repos/OWNER/REPO/pulls/comments/CID` works on pending comments. Try one comment first. If it fails, ask whether to delete the pending review (`DELETE repos/OWNER/REPO/pulls/N/reviews/ID`) and POST again. Build the new bodies from the current GitHub text so UI edits survive, and take the anchors from the saved payload, since pending comments return `line: null`.
4. Whenever a pending review returns 404, the reviewer most likely discarded it in the UI. Treat that as deliberate and don't recreate it without asking.

## Common mistakes

| Mistake | Result |
|---|---|
| Payload contains `event` | Review is published immediately |
| Anchor outside a diff hunk | POST fails with 422 |
| `commit_id` is not the head | Comments land on outdated lines |
| `--jq` on a failing call | jq error hides the API's 404/422; look at the raw response first |
| Copying review text from a rendered chat or terminal | Markdown is lost; copy from the raw `.md` file instead |
