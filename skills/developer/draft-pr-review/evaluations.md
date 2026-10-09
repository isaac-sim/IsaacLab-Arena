# Draft PR Review Evaluations

## Scenario 1: Create A Pending Review From Local Findings

Query: "Here are my review notes for PR 1427 in `/tmp/pr1427-review/notes.md`. Put them on the PR as a
review I can still edit and submit myself, with inline comments on the right lines."

Expected behavior:

- Confirms `gh` is logged in as the reviewer and records the PR head SHA.
- Checks for an existing pending review and for review threads that already raise the same points,
  and reports overlaps before posting.
- Builds one payload with no `event`, the head SHA as `commit_id`, and the attribution line as the
  first line of the summary and of every inline comment. Points without a diff line go in the summary.
- Runs `validate_pending_review.py` and fixes everything it reports before creating the review.
- Creates the review, reads the raw response, confirms `"state": "PENDING"` and the anchored lines,
  and hands over the PR's files link. Never submits the review.

## Scenario 2: Change A Pending Review The Reviewer Already Edited

Query: "Shorten the AI note in my pending review on PR 1427."

Expected behavior:

- Fetches the current review and comments and compares them with the saved payload before writing.
- Keeps the reviewer's UI edits: builds the new bodies from the current GitHub text, not from the
  saved payload.
- Updates the summary with `PUT`, tries `PATCH` on one inline comment first, and asks before deleting
  and recreating the review if that fails.
- Confirms the review is still pending afterwards.

## Scenario 3: The Pending Review Is Gone

Query: "Add the missing test-coverage point to my pending review on PR 1427."

Expected behavior:

- Gets a 404 for the saved review id and finds no pending review on the PR.
- Tells the reviewer that the review was most likely discarded in the UI and asks before creating a
  new one, instead of recreating it automatically.
