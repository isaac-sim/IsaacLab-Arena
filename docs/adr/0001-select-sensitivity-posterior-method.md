---
status: proposed
---

# Select the sensitivity posterior method from the observation query

Sensitivity reports should distinguish the posterior method from the fitted engine. An empirical posterior should use episodes that exactly match the observation query; a fitted posterior should interpolate when direct matching is insufficient. When fitting is requested, Arena should continue to select NPE for continuous factors and MNPE when any factor is categorical. The initial interface should explain and recommend a method without silently selecting one.

## Proposed interface

Introduce three separate concepts:

- A dataset profile records factor types and ranges, outcome types, category counts, and episode counts.
- An observation query records the requested outcome names and values.
- A method recommendation records `empirical` or `fitted`, the NPE or MNPE engine when fitting applies, matching-episode counts, reasons, and warnings.

The recommendation must inspect both the dataset and the observation query. The same dataset can support an empirical posterior for `success=1` and require a fitted posterior for an exact continuous duration. Start with an inspection command such as `--recommend-method`; require an explicit `--method empirical` or `--method fitted` for new computation entry points until the rules and fitted engines have sufficient calibration coverage.

`SensitivityAnalyzer.analyze(method=...)` is the public computation facade. It requires the library caller to choose a method and prepares the observation query consistently. Empirical analysis returns an `EmpiricalSensitivityResult`; fitted analysis retains its existing posterior-sample tensor. The empirical calculation remains a pure lower-level function, while `fit()` and `sample_posterior()` remain available for callers that need direct control of fitted inference. The existing `generate_report()` function and report CLI temporarily retain their fitted default for backward compatibility.

An empirical report should use matching episode factors directly, compare their fixed-bin distribution with the experiment's sampled distribution, show posterior-to-sampling ratios and uncertainty, and report the number of matching episodes. A fitted report should state whether NPE or MNPE was selected and why.

## Current implementation status

[PR #1133](https://github.com/isaac-sim/IsaacLab-Arena/pull/1133) implements the first step of this proposal on `cvolk/feature/empirical-sensitivity-report`. As of September 2, 2026, its tip is `cc3c60a33` and it is not yet part of `main`.

- Empirical analysis exactly matches the full observation, bins factors from their declared schema, and plots posterior-to-sampling ratios with paired-bootstrap intervals.
- Factor names, bounds, and categorical choices remain dataset-driven. The report does not encode camera-specific labels or units.
- The report shows only the posterior ratio; a separate outcome-rate panel was removed during review.
- The bootstrap count retains a programmatic default of 1,000 without adding a CLI option.
- Fitted analysis still returns its existing sample tensor. The proposed `FittedSensitivityResult` wrapper was removed because it added no required information.
- `EmpiricalMarginal` and `EmpiricalSensitivityResult` retain the non-rectangular per-factor bin data and analysis-wide metadata needed by the renderer. Before treating these types as stable API, audit fields left unused after removing the outcome-rate panel.

The implementation does not correct bounded NPE/MNPE inference or the fitted KDE renderer. Those remain the next independent work item.

## Evidence

The August 21, 2026 DROID wrist-camera sweep contains 1,000 episodes, including 727 successes, with three independently and approximately uniformly sampled translation factors in `[-0.03, 0.03]` metres. Six equal-width bins produced these success rates:

| Factor | -30 to -20 mm | -20 to -10 mm | -10 to 0 mm | 0 to 10 mm | 10 to 20 mm | 20 to 30 mm |
|---|---:|---:|---:|---:|---:|---:|
| Camera offset `[0]` | 66.2% | 73.5% | 70.1% | 77.5% | 75.6% | 72.3% |
| Camera offset `[1]` | 81.1% | 85.2% | 82.9% | 83.6% | 67.6% | 38.0% |
| Camera offset `[2]` | 66.9% | 72.9% | 72.4% | 76.9% | 72.0% | 74.1% |

The raw outcomes strongly associate positive camera offset `[1]` with failure, while offsets `[0]` and `[2]` show no clear marginal association. The current fitted report instead gives every factor a central peak and makes `[2]` appear most concentrated.

A shuffled-label control preserved the false central peaks after removing every relationship between factors and success. The fitted distribution places about 7-9% of its pre-filter samples outside the configured bounds; the `BoxUniform` support check rejects those samples and narrows the accepted distribution. The plotting code then applies an ordinary Gaussian KDE, which gives a known uniform distribution roughly half its correct density at the boundaries. These are separate fitting and rendering problems.

RoboLab uses the same normalized NPE or MNPE plus `BoxUniform` construction, so it does not resolve the fitted-distribution problem. Its fixed-bin histogram avoids the additional KDE boundary bias and is worth adapting with declared, shared factor bounds.

The two provided files named `arena_experiment_result.json` were byte-identical. Source artifact checksum: `7d93734aae7179669910d003e3ed2900bb496ecf384ad3e9dd701a029c457065`. Deterministically extracted 1,000-row JSONL checksum: `8151ece3d04906694f467ba18ce2e80c628bb44b6e30e7ec298633376e139ce7`.

## Considered options

- Always fit NPE or MNPE. Rejected for exact binary queries with many matching episodes because it approximates a posterior already represented directly by the data and currently fails the shuffled-label control.
- Copy RoboLab's approach. Rejected as a complete solution because it shares the bounded fitted-distribution problem, although its histogram is preferable to ordinary KDE.
- Select a method silently. Deferred because sample sufficiency depends on the observation query and intended precision; the tool should first expose its reasoning and warnings.
- Use an empirical posterior for exact matches and a fitted posterior for interpolation. Proposed because it keeps simple cases transparent while retaining fitted inference for continuous or sparse observation queries.

## Required follow-up before fitted reports are trusted

- Preserve declared factor bounds instead of inferring them from observed extrema.
- Represent bounded continuous factors so fitted samples are valid by construction rather than filtered afterward.
- Replace ordinary KDE with fixed-bound histograms or a boundary-corrected estimator.
- Add uniform-null, shuffled-label, known-effect, successful-sample calibration, interaction, and multi-seed tests.
- Decide how recommendations report sample sufficiency without claiming a universal threshold.
- Resolve three episodes in the reference result whose top-level success value disagrees with recorded task progress.

## Expected sequence

1. Add dataset/query inspection and an empirical posterior report. PR #1133 implements the report; dataset/query recommendation remains deferred.
2. Correct bounded fitted inference and add calibration tests.
3. Publish the representative dataset at an immutable revision and update the sensitivity documentation.
