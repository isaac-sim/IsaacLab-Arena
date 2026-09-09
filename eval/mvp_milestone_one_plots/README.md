# Sensitivity analysis — MVP milestone one plots

Parked analysis artifacts from the first sensitivity-analysis milestone (June 2026): policy
success rates and wrist-camera sensitivity for two policies, **pi** and **gn16**, across four
manipulation tasks.

The raw eval datasets (`episode_results.jsonl`) are intentionally **not** committed (kept on the
host / mounted, per the repo's no-datasets rule). Only the scripts and generated plots are parked
here. The scripts read the extracted datasets from `eval/<dataset>/...` and are run from the repo
root inside the container.

## Contents

- `success_rate_pi_vs_gn16.png` — per-task success rate, pi vs gn16 (headline result).
- `compare_pi_vs_gn16_camera.png` — wrist-camera posterior marginals (success-conditioned),
  pi vs gn16, per axis (horizontal / vertical / depth).
- `per_task_marginals/` — single-policy sensitivity reports per dataset (posterior marginals),
  plus the success-vs-failure contrast for tools_container.
- `scripts/success_barplot.py`, `scripts/compare_overlay.py` — the generators (paths point at
  `eval/<dataset>/`).

## Success rates

| Task | pi | gn16 | ratio |
|------|-----|------|-------|
| bagel_on_plate | 65.4% (n=2000) | 8.5% (n=1000) | 7.7x |
| objects_in_box | 55.3% (n=1800) | 15.3% (n=1000) | 3.6x |
| two_bin_sort | 23.5% (n=1600) | 2.8% (n=1000) | 8.4x |
| mustard_on_box* | 25.6% (n=1000) | 2.5% (n=1000) | - |

pi outperforms gn16 on every task (3.6x–8.4x). gn16's best task is objects_in_box.

\* The mustard row is **not a matched comparison**: pi ran `mustard_raisin` over 5 backgrounds,
gn16 ran the `mustard_raisin_box` variant on `home_office` only. Shown for reference, not a clean
model A/B. The other three rows are matched (same task and background per policy).

## Source eval runs (pi | gn16)

| Task | pi zip | gn16 zip |
|------|--------|----------|
| bagel_on_plate | 0625_2k_bagel_plate_banana_bowl_linked | 0625_1k_bagel_plate_banana_bowl_gn16 |
| objects_in_box | 0625_2k_bin_mug_marker_bowl_linked_pi | 0625_1k_bin_mug_marker_bowl_gn16 |
| two_bin_sort | 0625_1.6k_two_bin_pi | 06_25_1k_two_bin_gn16 |
| mustard_on_box | 0625_1k_mustard_raisin_pi | 0625_1k_mustard_raisin_box_gn16 |

## Sensitivity takeaways

- All tasks vary the wrist-camera extrinsics (3-vector) and HDR background. Camera sensitivity is
  read from the success-conditioned posterior marginal: concentration tighter than the uniform
  prior means that axis matters.
- Both policies broadly favor the camera near nominal. For pi the only axis with real
  concentration is **vertical**; horizontal/depth are near-prior (insensitive). gn16's runs are in
  the low-success regime (few successes), so its camera sensitivity is largely unresolved — the
  bumps in its curves are sampling noise, not established preferences.
- `P(displacement | success)` is normalized per policy, so these curves carry **no** information
  about relative success rates between policies — read shape only (see the success-rate plot for
  rates).
