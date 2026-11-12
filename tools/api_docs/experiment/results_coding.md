# Coding-task experiment: ten tasks on four source settings

Does the source setting change how well and how cheaply a fresh agent completes real coding tasks? Ten tasks
(`coding/tasks.py`) were each run once on four python-only trees, 40 runs in total, each by a fresh Sonnet agent
working in its own copy of the tree.

| Cond | What the agent sees                                                                 |
|------|-------------------------------------------------------------------------------------|
| c1   | python only, original docstrings                                                    |
| c2   | python only, no docstrings (comments kept)                                          |
| c3   | python only, no docstrings, no comments, obfuscated names                           |
| c4   | c1 tree plus a generated text API reference built from the original docstrings      |

| Task | Category | What                                                                           |
|------|----------|--------------------------------------------------------------------------------|
| T01  | feature  | new `ConstantActionPolicy`, registered, with tests                              |
| T02  | feature  | new `FarFrom` relation: relation, loss strategy, solver mapping, validator, test |
| T03  | feature  | `Pose.inverse()` with frame-naming conventions and a test                       |
| T04  | fix      | assert on incompatible relation pairs (NextTo/NotNextTo, two On parents)        |
| T05  | fix      | find and fix broken code in `isaaclab_arena_gr00t/utils` (no hint), add tests   |
| T06  | tests    | simulator-free tests for the success-combination logic                          |
| T07  | refactor | rename two `EefPose` accessors and their callers                                |
| T08  | refactor | share constructor logic between `NextTo` and `NotNextTo`                        |
| T09  | feature  | tensor support for `wrap_angle_to_pi`, with tests                               |
| T10  | feature  | make DreamZero reconnect attempts and wait configurable                         |

Score is the fraction of rubric checks met (1, 0.5 or 0 per check), judged by a grader agent from the diff of each
attempt against the pristine tree. Defects are real bugs, broken references or unrequested behaviour changes the
grader found. Tokens, tool calls and seconds come from the agent transcripts (`variants/collect_usage.py`; the
harness hand-back call is not counted). Rubrics are in `coding/tasks.py`.

## What the numbers say

- There is no clear winner. Score is 91.6 to 98.8 % and tool calls are 6.4 to 9.0 across the four settings, with one
  run per cell; most of that spread comes from three or four runs.
- The one outright failure is c2 T08 (the refactor): its base class passes wrong arguments to the parent constructor
  and would crash. c4 had the next-worst run (T06, 71 %): its tests start the simulator and its error checks are
  weak.
- The generated API reference (c4 against c1) did not help: +1.9k tokens and +1.0 tool calls per task, and a lower
  score (95.7 against 98.5 %). It did not save the lookups on T05 (20 tool calls against 8) or T02.
- The obfuscated tree (c3) was not worse. It used the fewest tool calls (6.4) and scored 98.8 %. Agents read the
  structure and the surrounding code rather than the names, and copied existing patterns.
- Token use is dominated by the fixed overhead and by task size: T02 (new relation across five files) is 100k tokens,
  the rename T07 is 52k.
- Hard multi-file tasks (T02) were done correctly on all four settings, including the wiring a new relation needs.
  Two attempts (c1, c2) missed the relation-catalogue test that has to list `far_from`.

## Caveats

- One run per (task, condition). Differences under about 10 points of score or 2 tool calls are noise.
- Nothing could be executed (no torch, no simulator). Correctness is judged by reading, not by running tests, so
  the scores are an upper bound.
- Graders are agents and saw the four attempts of a task together, labelled by condition.
- The obfuscated tree is itself broken in places: for example `mode.upper()` in `terminations.py` was renamed to
  `mode.p_c538c1()`, so `check_success` would fail at run time in c3. The c3 T06 agent noticed and the grader did not
  penalise it. The python-only trees do not contain docs, yaml or markdown.
- T07 turned out to be tiny (two definitions and one caller), so it only tests whether the caller is found.
- The prompts were read from files, which adds one tool call to every run.

Raw per-run numbers are in `results_coding.csv`; the diffs of every attempt and the grader scores are in
`coding/diffs.tar.gz` and `coding/scores/`.

## Aggregate over 10 tasks

| What the agent sees                           | Runs | Mean tokens | Mean tools | Mean secs | Score % | Defects/run |
|-----------------------------------------------|-----:|------------:|-----------:|----------:|--------:|------------:|
| original docstrings                           |   10 |      68,549 |        7.8 |        40 |    98.5 |         0.1 |
| no docstrings                                 |   10 |      67,279 |        9.0 |        39 |    91.6 |         0.2 |
| no docstrings, no comments, obfuscated names  |   10 |      67,394 |        6.4 |        35 |    98.8 |         0.0 |
| original docstrings + generated API reference |   10 |      70,441 |        8.8 |        40 |    95.7 |         0.2 |

## By category


### feature

| What the agent sees                           | Runs | Mean tokens | Mean tools | Mean secs | Score % | Defects/run |
|-----------------------------------------------|-----:|------------:|-----------:|----------:|--------:|------------:|
| original docstrings                           |    5 |      69,546 |        8.4 |        42 |    98.8 |         0.2 |
| no docstrings                                 |    5 |      69,863 |        8.8 |        42 |    93.4 |         0.2 |
| no docstrings, no comments, obfuscated names  |    5 |      69,117 |        7.2 |        36 |    97.5 |         0.0 |
| original docstrings + generated API reference |    5 |      71,242 |        7.6 |        39 |   100.0 |         0.0 |

### fix

| What the agent sees                           | Runs | Mean tokens | Mean tools | Mean secs | Score % | Defects/run |
|-----------------------------------------------|-----:|------------:|-----------:|----------:|--------:|------------:|
| original docstrings                           |    2 |      70,832 |        8.5 |        39 |    95.8 |         0.0 |
| no docstrings                                 |    2 |      69,383 |       14.0 |        45 |    95.8 |         0.0 |
| no docstrings, no comments, obfuscated names  |    2 |      70,314 |        7.5 |        39 |   100.0 |         0.0 |
| original docstrings + generated API reference |    2 |      73,824 |       14.5 |        43 |    92.7 |         0.5 |

### refactor

| What the agent sees                           | Runs | Mean tokens | Mean tools | Mean secs | Score % | Defects/run |
|-----------------------------------------------|-----:|------------:|-----------:|----------:|--------:|------------:|
| original docstrings                           |    2 |      58,330 |        4.5 |        19 |   100.0 |         0.0 |
| no docstrings                                 |    2 |      57,496 |        5.5 |        19 |    78.6 |         0.5 |
| no docstrings, no comments, obfuscated names  |    2 |      56,910 |        4.0 |        17 |   100.0 |         0.0 |
| original docstrings + generated API reference |    2 |      58,098 |        5.0 |        19 |   100.0 |         0.0 |

### tests

| What the agent sees                           | Runs | Mean tokens | Mean tools | Mean secs | Score % | Defects/run |
|-----------------------------------------------|-----:|------------:|-----------:|----------:|--------:|------------:|
| original docstrings                           |    1 |      79,440 |       10.0 |        76 |   100.0 |         0.0 |
| no docstrings                                 |    1 |      69,719 |        7.0 |        52 |   100.0 |         0.0 |
| no docstrings, no comments, obfuscated names  |    1 |      73,909 |        5.0 |        61 |   100.0 |         0.0 |
| original docstrings + generated API reference |    1 |      84,358 |       11.0 |        82 |    71.4 |         1.0 |

## Per task

| Task | Category | Cond |  Tokens | Tools | Secs | Score % | Defects |
|------|----------|------|--------:|------:|-----:|--------:|--------:|
| T01  | feature  | c1   |  65,091 |    11 |   36 |     100 |       0 |
| T01  | feature  | c2   |  65,007 |    12 |   38 |      94 |       0 |
| T01  | feature  | c3   |  61,066 |    11 |   29 |      94 |       0 |
| T01  | feature  | c4   |  65,728 |    11 |   33 |     100 |       0 |
| T02  | feature  | c1   | 100,547 |    11 |   94 |     100 |       1 |
| T02  | feature  | c2   | 104,832 |    12 |   98 |      94 |       1 |
| T02  | feature  | c3   |  98,516 |     9 |   67 |     100 |       0 |
| T02  | feature  | c4   | 107,953 |    13 |   99 |     100 |       0 |
| T03  | feature  | c1   |  58,725 |     8 |   21 |      94 |       0 |
| T03  | feature  | c2   |  59,711 |    11 |   25 |      94 |       0 |
| T03  | feature  | c3   |  61,923 |     5 |   26 |      94 |       0 |
| T03  | feature  | c4   |  59,626 |     5 |   19 |     100 |       0 |
| T04  | fix      | c1   |  70,931 |     9 |   42 |     100 |       0 |
| T04  | fix      | c2   |  67,288 |    11 |   45 |     100 |       0 |
| T04  | fix      | c3   |  70,140 |     7 |   32 |     100 |       0 |
| T04  | fix      | c4   |  71,846 |     9 |   39 |      94 |       1 |
| T05  | fix      | c1   |  70,733 |     8 |   36 |      92 |       0 |
| T05  | fix      | c2   |  71,478 |    17 |   45 |      92 |       0 |
| T05  | fix      | c3   |  70,488 |     8 |   46 |     100 |       0 |
| T05  | fix      | c4   |  75,802 |    20 |   46 |      92 |       0 |
| T06  | tests    | c1   |  79,440 |    10 |   76 |     100 |       0 |
| T06  | tests    | c2   |  69,719 |     7 |   52 |     100 |       0 |
| T06  | tests    | c3   |  73,909 |     5 |   61 |     100 |       0 |
| T06  | tests    | c4   |  84,358 |    11 |   82 |      71 |       1 |
| T07  | refactor | c1   |  52,952 |     3 |   12 |     100 |       0 |
| T07  | refactor | c2   |  52,822 |     3 |   12 |     100 |       0 |
| T07  | refactor | c3   |  51,986 |     3 |   12 |     100 |       0 |
| T07  | refactor | c4   |  53,485 |     3 |   10 |     100 |       0 |
| T08  | refactor | c1   |  63,707 |     6 |   25 |     100 |       0 |
| T08  | refactor | c2   |  62,169 |     8 |   25 |      57 |       1 |
| T08  | refactor | c3   |  61,835 |     5 |   22 |     100 |       0 |
| T08  | refactor | c4   |  62,710 |     7 |   29 |     100 |       0 |
| T09  | feature  | c1   |  57,981 |     6 |   30 |     100 |       0 |
| T09  | feature  | c2   |  56,829 |     4 |   22 |     100 |       0 |
| T09  | feature  | c3   |  57,169 |     5 |   22 |     100 |       0 |
| T09  | feature  | c4   |  57,492 |     4 |   21 |     100 |       0 |
| T10  | feature  | c1   |  65,386 |     6 |   30 |     100 |       0 |
| T10  | feature  | c2   |  62,935 |     5 |   24 |      86 |       0 |
| T10  | feature  | c3   |  66,912 |     6 |   35 |     100 |       0 |
| T10  | feature  | c4   |  65,409 |     5 |   25 |     100 |       0 |
