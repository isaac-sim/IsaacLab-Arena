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
     "body": "_AI-drafted (<agent name>), reviewed by me._\n\n**Blocking: <title>**\n\n<what the code does now>\n\n<why it matters: failure scenario or cost>\n\n<suggestion>"},
    {"path": "pkg/builder.py", "start_line": 304, "line": 331, "start_side": "RIGHT", "side": "RIGHT",
     "body": "_AI-drafted (<agent name>), reviewed by me._\n\n**Design: <title>**\n\n..."},
    {"path": "tests/test_replay.py", "line": 253, "side": "LEFT",
     "body": "_AI-drafted (<agent name>), reviewed by me._\n\n**Question: <about deleted code>**\n\n..."}
  ]
}
```

- No `event` key: that is what makes the review PENDING. Any `event` value publishes it.
- The attribution line is the first line of the body and of every comment.
- Refer to code by its name (class, function, method, variable), with file:line, so each comment can be checked without searching.
- Write each comment as short paragraphs, one point each: what the code does now, why it matters, then the suggestion. Put lists of items in bullets.
- `side: RIGHT` takes new-file line numbers of added or context lines; `side: LEFT` takes old-file line numbers of deleted lines. A range must stay inside one hunk. Points without a diff line go in the summary.
- Label each comment **Blocking**, **Design**, **Question** or **Suggestion (non-blocking)**, and order them that way.

## Steps

1. Account: `gh api user --jq .login` must be the reviewer's own account; pending reviews are only visible to their author.
2. Head SHA: `gh api repos/OWNER/REPO/pulls/N --jq .head.sha` → `commit_id`. Check each line reference in the findings with `git show <sha>:<path>`.
3. Existing pending review (GitHub allows one per user per PR): `gh api repos/OWNER/REPO/pulls/N/reviews --paginate --jq '.[] | select(.state=="PENDING") | .id'`. If one exists, ask the reviewer whether to add the comments to it (see "Adding to an existing pending review").
4. Existing discussion: read the review summaries, inline threads (`.../pulls/N/comments`) and top-level comments (`.../issues/N/comments`). If someone already raised a point, even partly, tell the reviewer and wait for their answer before posting.
5. Validate: `python3 -I skills/developer/draft-pr-review/validate_pending_review.py OWNER/REPO N payload.json`. Fix everything it reports.
6. Create: `gh api repos/OWNER/REPO/pulls/N/reviews --method POST --input payload.json > response.json`, then read `response.json`. Expect `"state": "PENDING"`.
7. Confirm anchors: `gh api repos/OWNER/REPO/pulls/N/reviews/ID/comments --jq '.[] | "\(.path): \(.diff_hunk | split("\n") | last)"'`. For pending comments `line` is null; the last `diff_hunk` line is the anchored line.
8. Hand over `https://github.com/OWNER/REPO/pull/N/files`. The reviewer reads and edits every comment, which is what makes "reviewed by me" true, then uses "Finish your review" → Comment, Request changes or Approve.

## Adding to an existing pending review

A second POST to `.../reviews` fails while a pending review exists. Add threads to it with GraphQL instead:

1. Get the review: `gh api repos/OWNER/REPO/pulls/N/reviews --paginate --jq '.[] | select(.state=="PENDING") | {id, node_id, commit_id}'`. Its `commit_id` must be the PR head; if not, ask the reviewer.
2. Read its comments (`.../reviews/ID/comments`) so the new ones don't repeat what the reviewer already wrote.
3. Validate a payload with `commit_id` and `comments` only (no `body`), as in step 5.
4. Add each comment:

   ```bash
   gh api graphql -f query='mutation($review: ID!, $path: String!, $line: Int!, $body: String!) {
     addPullRequestReviewThread(input: {pullRequestReviewId: $review, path: $path, line: $line, side: RIGHT, body: $body}) {
       thread { comments(first: 1) { nodes { databaseId } } } } }' \
     -f review=<node_id> -f path=pkg/builder.py -F line=339 -F body=@comment.md
   ```

   Use `side: LEFT` for deleted lines, and add `startLine` and `startSide` for a range.
5. Confirm the review is still PENDING and the anchors are right, as in step 7.

## Changing an existing pending review

1. Fetch the current review and its comments and compare them with the saved payload. If they differ, the reviewer edited them in the UI: ask before overwriting anything.
2. Summary: `gh api repos/OWNER/REPO/pulls/N/reviews/ID --method PUT --input body.json` with `{"body": "..."}`. This works while the review is pending.
3. Inline comments: the REST endpoints for single comments (`GET`/`PATCH repos/OWNER/REPO/pulls/comments/CID`) return 404 for pending comments. Take the comment's `node_id` from `.../reviews/ID/comments` and update it with GraphQL, building the new body from the current GitHub text so UI edits survive:

   ```bash
   gh api graphql -f query='mutation($comment: ID!, $body: String!) {
     updatePullRequestReviewComment(input: {pullRequestReviewCommentId: $comment, body: $body}) {
       pullRequestReviewComment { state } } }' \
     -f comment=<node_id> -F body=@comment.md
   ```

4. Whenever a pending review returns 404, the reviewer most likely discarded it in the UI. Treat that as deliberate and don't recreate it without asking.

## Common mistakes

| Mistake | Result |
|---|---|
| Payload contains `event` | Review is published immediately |
| Anchor outside a diff hunk | POST fails with 422 |
| `commit_id` is not the head | Comments land on outdated lines |
| `--jq` on a failing call | jq error hides the API's 404/422; look at the raw response first |
| REST `.../pulls/comments/CID` on a pending comment | 404; read it from `.../reviews/ID/comments` and edit it with GraphQL |
| Copying review text from a rendered chat or terminal | Markdown is lost; copy from the raw `.md` file instead |
