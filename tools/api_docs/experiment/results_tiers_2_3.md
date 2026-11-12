# Results: tiers 2 and 3

Same protocol as `results.md` (tier 1): fresh Sonnet agent per question and arm, arm A with the generated API
reference in addition to the source, arm B source only, identical repository snapshot. Differences from tier 1:

- Each agent first read a one-file prompt (`Read <prompt file> and do what it says`), which adds one tool call to
  every run in both arms and about 3k tokens of fixed overhead.
- Answers are graded against atomic facts (`answers_tier2.md`, `answers_tier3.md`), 1 point per fact, 0.5 for a
  partially stated fact. The questions were authored and fact-checked by a separate agent; I spot-checked six against
  the code.

## Tier 2: discovery questions (20)

The question describes a behaviour or need without naming the symbol.

| Arm | Runs | Mean tokens | Median tokens | Mean tools | Mean secs | Facts hit | Facts % |
|-----|-----:|------------:|--------------:|-----------:|----------:|----------:|--------:|
| A   |   20 |       55773 |         54794 |       5.70 |      16.8 |     79/87 |    90.8 |
| B   |   20 |       56001 |         55163 |       5.20 |      16.1 |     80/87 |    92.5 |

| ID  | A tokens | B tokens | A tools | B tools | A secs | B secs | A facts | B facts |
|-----|---------:|---------:|--------:|--------:|-------:|-------:|--------:|--------:|
| M01 |    54883 |    54691 |       6 |       5 |   16.5 |   13.4 |   4.5/5 |   4.5/5 |
| M02 |    54418 |    55320 |       5 |       4 |   14.9 |   13.3 |   3.5/4 |     4/4 |
| M03 |    54699 |    56529 |       4 |       4 |   12.1 |   11.8 |     4/4 |     4/4 |
| M04 |    53203 |    52537 |       5 |       4 |   13.5 |   12.3 |     3/4 |     3/4 |
| M05 |    59125 |    59130 |       5 |       5 |   17.6 |   16.3 |     4/4 |     4/4 |
| M06 |    54420 |    56785 |       6 |       4 |   15.8 |   15.8 |     4/4 |     4/4 |
| M07 |    54681 |    54215 |       6 |       6 |   16.9 |   14.8 |     4/4 |     3/4 |
| M08 |    51977 |    52390 |       5 |       4 |   12.9 |   10.6 |     4/5 |   4.5/5 |
| M10 |    57589 |    58004 |       6 |       5 |   17.7 |   15.2 |     5/5 |     5/5 |
| M11 |    54706 |    53718 |       5 |       4 |   13.6 |   12.1 |     5/5 |     5/5 |
| M12 |    53511 |    54119 |       4 |       5 |   13.6 |   14.8 |     3/4 |   3.5/4 |
| M13 |    59507 |    69713 |       5 |      14 |   21.9 |   44.0 |     5/5 |     5/5 |
| M14 |    54959 |    56530 |       7 |       8 |   18.9 |   21.5 |     4/4 |     4/4 |
| M15 |    61409 |    57437 |       8 |       4 |   24.3 |   19.0 |     3/4 |     4/4 |
| M16 |    55695 |    52680 |       6 |       5 |   16.9 |   14.6 |     4/4 |     4/4 |
| M17 |    52793 |    52527 |       5 |       5 |   16.3 |   15.1 |     4/5 |     4/5 |
| M18 |    61956 |    56350 |       8 |       5 |   24.3 |   16.1 |     4/4 |     4/4 |
| M19 |    55857 |    55006 |       5 |       4 |   14.7 |   13.5 |     4/4 |     4/4 |
| M20 |    52551 |    53865 |       5 |       5 |   12.7 |   13.6 |     3/4 |     3/4 |
| M21 |    57514 |    58473 |       8 |       4 |   21.3 |   13.1 |     4/5 |     4/5 |

## Tier 3: extra difficult questions (20)

Multi-file tracing, precedence, checklists, differences, error paths and counting.

| Arm | Runs | Mean tokens | Median tokens | Mean tools | Mean secs | Facts hit | Facts % |
|-----|-----:|------------:|--------------:|-----------:|----------:|----------:|--------:|
| A   |   20 |       62156 |         61648 |       6.90 |      25.6 |    88/103 |    85.9 |
| B   |   20 |       61408 |         62407 |       6.50 |      25.9 |    89/103 |    86.4 |

| ID  | A tokens | B tokens | A tools | B tools | A secs | B secs | A facts | B facts |
|-----|---------:|---------:|--------:|--------:|-------:|-------:|--------:|--------:|
| H01 |    70353 |    65555 |       8 |       8 |   30.4 |   30.5 |   4.5/6 |   4.5/6 |
| H02 |    55181 |    52882 |       5 |       4 |   15.3 |   12.1 |     4/5 |     4/5 |
| H04 |    60245 |    62202 |       6 |       5 |   19.9 |   21.5 |     4/5 |     4/5 |
| H05 |    57087 |    59212 |       9 |       8 |   23.1 |   21.3 |     4/5 |   4.5/5 |
| H06 |    69449 |    67436 |       5 |       5 |   34.4 |   35.1 |     5/5 |     5/5 |
| H07 |    56280 |    55348 |       5 |       5 |   23.6 |   20.3 |     4/5 |     4/5 |
| H09 |    65059 |    63471 |       7 |       7 |   22.0 |   21.0 |   3.5/5 |   4.5/5 |
| H10 |    67320 |    65025 |       8 |       8 |   31.4 |   26.7 |     4/5 |  4.25/5 |
| H11 |    61847 |    68353 |       7 |      11 |   22.5 |   39.4 |   4.5/5 |     5/5 |
| H12 |    56946 |    55559 |       7 |       6 |   19.2 |   16.6 |     4/5 |     5/5 |
| H13 |    56640 |    58290 |       6 |       7 |   20.2 |   23.8 |     5/5 |     5/5 |
| H14 |    61448 |    57604 |       7 |       5 |   23.0 |   25.8 |  4.25/5 |   3.5/5 |
| H15 |    57917 |    59101 |       7 |       7 |   19.9 |   25.8 |     4/5 |     4/5 |
| H16 |    64012 |    65777 |       9 |       7 |   41.5 |   37.4 |  4.75/5 |     5/5 |
| H17 |    65505 |    63319 |       8 |       6 |   28.6 |   33.3 |     5/5 |     5/5 |
| H18 |    67967 |    61873 |       6 |       7 |   31.7 |   23.4 |  5.25/6 |  5.25/6 |
| H19 |    59061 |    58774 |       6 |       6 |   26.2 |   23.9 |     4/5 |     4/5 |
| H20 |    58405 |    62694 |       5 |       4 |   19.9 |   27.9 |  4.25/5 |  4.75/5 |
| H21 |    67665 |    62612 |      10 |       6 |   34.8 |   23.8 |   5.5/6 |  4.75/6 |
| H22 |    64725 |    63067 |       7 |       8 |   24.4 |   27.9 |     5/5 |     3/5 |

## Findings

- **The generated reference did not help on any tier.** Tokens, tool calls and time are within noise of the
  source-only arm, and fact coverage is the same or slightly worse: tier 2 A 90.8% vs B 92.5%, tier 3 A 85.9% vs
  B 86.4%. Paired by question, arm B scored higher in 4 of 20 (tier 2) and 7 of 20 (tier 3), arm A in 1 and 3.
- **Discovery by behaviour is solved by grep, not by the reference.** For tier 2 both arms located the right file
  in about 5 tool calls, usually by grepping a distinctive word from the question (an error message, a constant, a
  noun). Several arm A agents said in their answers that they went to the source after finding the entry point, because the reference did not contain the detail asked for.
- **The reference lacks what the hard questions ask for.** It is generated from signatures and docstrings, so it
  says little about control flow, ordering, assert messages, constants inside bodies and cross-file wiring. Arm A
  agents wrote this themselves in their answers ("the API reference does not state ...") and fell back to the
  source. Cost: they read both.
- **Where the reference was trusted alone, answers were weaker.** Examples are tier 1 Q15 and Q19, and tier 3 H09-A,
  which stopped at the reference and left facts out.
- **Difficulty is visible in cost, not in the arm.** From tier 1 to tier 3 the average run grows from about 3 to
  about 7 tool calls and from about 9 s to about 26 s, for both arms. Facts missed in tier 3 were mostly peripheral
  details (a second caller, an extra assert, a default value) rather than the main answer.
- **Wrong answers were rare and mostly the question's fault or a misread order.** The one clearly wrong tier 3
  fact is H22-B's registration order, read from file layout.

## Caveats

- Fact grading was done by me from the answer text, with partial credit; it is not blinded and half-points are
  judgement calls. Differences below roughly 2 percentage points should be read as ties.
- The question author also saw the generated source, not the reference, so the question set is not biased towards
  the reference, and it is not biased against it either: the questions were written to need code facts.
- One sample per cell; durations include parallel-execution noise (up to 20 agents at once).
- A better-suited reference (call graphs, per-class inheritance, a symbol index with usage sites, a docs search tool)
  could behave differently. This measures the autoapi text build of docstrings and signatures.
