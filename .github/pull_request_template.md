<!--
Write for a teammate who knows the project but has not followed the discussion.
Do not use labels from internal plans or discussions.

Use simple English, short sentences, and short paragraphs.
Name classes, functions, and concepts by their actual names instead of generic descriptions,
and explain their responsibilities where useful.
Use bullets for lists, not as a requirement for every explanation.
Keep the detail proportional to the change; a small fix needs only a few sentences.
Leave out reassurances that a reviewer would not ask for, such as "No other behavior changes."
Remove sections that do not apply.
-->

<!--
Start every PR body with 1–2 concise sentences stating the main changes.
Use imperative verbs such as "Add", "Fix", or "Update" to state the changes directly.
For example: "Add a DROID pick-and-place example."
Omit introductory phrases such as "This PR adds".
A reader should understand the scope from these sentences alone.

Put this summary before the first heading, without a separate Summary heading.
Save motivation, background, and implementation details for the sections below.
Replace this comment with the summary.
-->

## Why this change?

<!--
Explain the problem with the current behavior and why it matters.
Give a concrete example when it helps. Base it on behavior that exists in the code;
if only a constructed case shows the problem, say so.

For a follow-up PR, briefly explain what the earlier change enabled
and what this PR adds. Include links, but make this explanation stand alone.
-->

## What changes?

<!--
Explain the solution and how it addresses the problem.
Focus on behavior and responsibilities, not a list of modified files.

Mention breaking changes, required migration steps, and important limitations
when relevant. Distinguish what this PR implements from future work.
-->

<!--
When an API or its usage changes, add a Usage example section using the skeleton below.
Replace the placeholders and move the completed section outside this comment.
Show a small before/after example of the same operation.
Use the same variable names and omit unrelated setup.
Explain any behavior difference that the code alone does not show.
Use the appropriate code-block language for the example.

## Usage example

Before:

```python
# Previous usage
```

After:

```python
# Updated usage
```
-->

## Validation

<!--
Optional; omit this section when there is nothing beyond CI to report,
such as for documentation-only or small, self-evident changes.
State what was tested and the result.
Separate passing checks from checks that failed or could not run.
Do not report that automated tests or pre-commit hooks pass; CI covers that. Describe manual checks,
such as running an example or inspecting a video, and anything CI does not cover.
Mention any important behavior that remains unverified.
-->
