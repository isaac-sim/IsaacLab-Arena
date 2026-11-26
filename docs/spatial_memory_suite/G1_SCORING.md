# G1 benchmark scoring

Each run gives GR00T the task instruction and head-camera RGB. The policy receives the G1 state and commands the whole-body controller. The accumulated depth map is disabled for this baseline. The experiment contains one episode for each of the 30 tasks in each layout. A single attempt per layout measures transfer on those instances and gives no reliability estimate.

## How an episode ends

A task normally requires five consecutive control steps satisfying its final predicate (0.10 s at 50 Hz). Seam inspection uses the ordered tip samples described below. The episode times out at 30 s. Pelvis height below 0.38 m triggers the fall condition. For rows marked “yes”, net contact on the designated guard above 0.5 N triggers a separate termination. The episode JSONL stores all active termination terms together. The success flag is Arena’s native progress-based result. A competing guard or fall on the same step must be reported alongside it. Asset-loading, network, inference and recording errors belong in the execution log. Only completed policy episodes count toward performance.

The bounds below are inclusive, in metres, in the floor-aligned workcell frame. Each pair gives opposite XYZ corners. They constrain body origins. These are the implemented measurements. For example, the trolley predicate checks position and levelness but leaves yaw and wheel alignment unscored, and the clamp measures contacts and travel without a holding-force test. End poses alone do not establish use of spatial memory.

| Task and instruction | Success conditions | Guard failure |
|---|---|---|
| `reverse_gap` — Back through this gap. | First pass the load through the opening, center in (-1.25, -0.03, 0.78) to (-1.05, 0.88, 1.15). Robot forward-axis component exceeds 0.60 along X. load center in (-2.5, -0.03, 0.78) to (-1.36, 0.88, 1.25). robot center in (-2.8, -0.03, 0.48) to (-1.32, 0.88, 1.1). | yes |
| `shelf_bay` — Seat the carton in this bay. | load up-axis z > 0.96. The carton center is inside x [−2.005, −1.90] and z [0.949, 0.97]. Layout A requires y [−0.366, −0.239], B [−0.756, −0.629]. Each of the three neighboring packages finishes within 3 cm of its initial center. Shelf contact with the carton exceeds 0.15 N. | yes |
| `partition_delivery` — Cross this marked threshold. | First pass the load through the opening, center in (-1.6, 0.16, 0.75) to (-1.39, 0.76, 1.25). load center in (-2.5, 0.17, 0.76) to (-1.94, 0.73, 1.3). robot center in (-2.9, 0.04, 0.45) to (-1.8, 0.85, 1.15). | yes |
| `cabinet_reach` — Touch the rear pad. | Pad center y exceeds 0.254 m. | yes |
| `tote_extraction` — Withdraw the tote onto the ledge. | tote center in (-0.28, -0.42, 0.82) to (0.28, -0.24, 0.93). | yes |
| `hanging_pan` — Unhook the pan. | pan center in (-0.5, -0.4, 1.04) to (0.02, 0.01, 1.45). The neighboring pan finishes within 4.5 cm of its initial center. | no |
| `valve_turn` — Turn the recessed lever to its stop. | The lever’s rotated local X axis has vertical component greater than 0.73. | yes |
| `seam_inspection` — Run the probe along the entire seam in one pass. | The probe tip must come within 3.5 cm of each sequential sample along the four weld segments. Samples are spaced at most 4 cm apart. This checks waypoint order and proximity. It permits pauses and excursions between samples. | no |
| `panel_catches` — Depress the two catches together. | Left and right catch centers simultaneously exceed y = 0.043 and 0.042 m respectively. | no |
| `dishwasher` — Seat this plate in this rack slot. | plate center in (-0.07, -0.066, 1.06) to (0.07, 0.045, 1.105). neighbor center in (-0.03, 0.1, 1.05) to (0.03, 0.16, 1.12). Plate face normal has absolute y component above 0.92. | yes |
| `billet_insertion` — Insert the billet to the machine stop. | billet center in (-0.12, 0.015, 1.04) to (0.12, 0.072, 1.16). billet up-axis z > 0.96. | yes |
| `rail_support` — Set the crate on both rails. | crate center in (-0.045, -0.09, 1.08) to (0.045, 0.08, 1.11). crate up-axis z > 0.96. Each rail independently exerts more than 0.15 N on the crate. | no |
| `stack_bracing` — Support the upper crate and pull the spacer free. | spacer center in (-0.03, -0.65, 0.84) to (0.35, -0.31, 1.1). crate center in (-0.045, -0.045, 1.12) to (0.045, 0.045, 1.21). crate up-axis z > 0.96. | no |
| `worktop_clamp` — Tighten the clamp until the worktop is secure. | Jaw center is within ±0.033 m of x = −0.12 (A) or 0.11 (B), y [−0.513, −0.447], z [0.774, 0.797]. Jaw height relative to the clamp exceeds −0.052 m. Each of the pad–jaw and worktop–clamp contact forces exceeds 0.15 N. | no |
| `loaded_drawer` — Close this drawer. | drawer center in (-0.01, -0.023, 0.83) to (0.01, 0.02, 0.85). Tray center relative to the drawer lies between (−0.13, −0.08, 0.015) and (0.13, 0.08, 0.075). | yes |
| `pipe_bracket` — Pass the rigid U-bracket around this pipe. | bracket center in (-0.23, 0.012, 1.035) to (0.23, 0.075, 1.105). | yes |
| `panel_removal` — Rotate the panel out of the frame. | panel center in (-0.45, -0.7, 1) to (0.45, -0.39, 1.48). | yes |
| `hatch_transfer` — Slide the tray through the hatch. | tray center in (0.49, -0.12, 0.82) to (0.6, 0.12, 0.92). tray up-axis z > 0.96. Both cup centers relative to the tray lie between (−0.24, −0.11, 0.06) and (0.24, 0.11, 0.12). | yes |
| `probe_withdrawal` — Pull the probe free. | probe center in (-0.38, -0.55, 0.93) to (0.35, -0.115, 1.35). | yes |
| `display_cover` — Peel the rigid-backed cover off the display. | cover center in (-0.4, -0.6, 1.0) to (0.4, -0.2, 1.4). Both simulated adhesive strips have released. | yes |
| `scoop_pour` — Tip the scoop into this chute. | First tilt the scoop over the chute (rotated up-axis z < 0.80). All six pellet centers must then be in the receiver between (−0.18, −0.14, 0.82) and (0.18, 0.14, 0.99). | yes |
| `panel_corner` — Turn this long load through the corner. | panel center in (-0.37, 0.31, 0.7) to (0.44, 0.72, 1.28). The panel’s rotated local X axis has absolute y component above 0.75. | yes |
| `aisle_retreat` — Back one chassis length out of the aisle. | First pass the chassis through the opening, center in (-0.24, -0.64, 0.74) to (0.24, -0.48, 1.19). Robot forward-axis component exceeds 0.60 along Y. chassis center in (-0.35, -1.55, 0.72) to (0.35, -1.02, 1.22). | yes |
| `platform_step` — Step onto this platform with the crate level. | robot center in (-0.37, 0.105, 0.82) to (0.35, 0.6, 1.17). crate center in (-0.55, 0.05, 0.88) to (0.5, 0.85, 1.45). crate up-axis z > 0.96. | yes |
| `beam_clearance` — Duck beneath this beam carrying a tall item. | First pass the load through the opening, center in (-0.45, -0.1, 0.54) to (0.45, 0.1, 1.1). load center in (-0.52, 0.4, 0.62) to (0.49, 1.1, 1.35). robot center in (-0.52, 0.34, 0.47) to (0.52, 1.3, 1.15). | yes |
| `trolley_dock` — Pivot this cart into its berth. | trolley center in (-0.06, 0.13, -0.025) to (0.06, 0.28, 0.04). trolley up-axis z > 0.96. | no |
| `pillar_pass` — Move sideways past this pillar. | case center in (0.29, 0.5, 0.72) to (1.15, 1.03, 1.35). robot center in (0.29, 0.4, 0.46) to (1.2, 1.3, 1.12). | yes |
| `closing_door` — Carry the parcel through the closing door. | First pass the parcel through the opening, center in (-0.18, -0.08, 0.68) to (0.18, 0.08, 1.3). parcel center in (-0.16, 0.36, 0.72) to (0.16, 1.1, 1.25). robot center in (-0.39, 0.24, 0.45) to (0.39, 1.3, 1.12). | yes |
| `moving_carriage` — Take the part from the crossing carriage. | part center in (-0.72, -0.49, 1.24) to (0.72, 0.16, 1.63). | yes |
| `descending_hood` — Draw the tray clear of the descending hood. | tray center in (-0.24, -0.85, 0.81) to (0.24, -0.52, 1.14). | yes |

## GR00T N1.6 transfer results

The G1 loco-manipulation checkpoint succeeded on 0 of 60 workcell episodes at seed 42. 56 ended on time out, 4 ended on guard contact. The same checkpoint completed the Galileo shelf-to-bin task, with lift at step 59 and placement at step 1,022. The control is scored separately. [Checkpoint revision](https://huggingface.co/nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation/tree/dfe74af855007f26093f362cd2d7a2f404b64b93).

Both layouts used a single RGB stream, G1 state, the workcell's instruction and the checkpoint's 50-command chunks. Neither workcell-specific fine-tuning nor a map-conditioned policy was evaluated. The results measure transfer from one trained task. One trial per layout provides limited evidence about a policy's success probability.

The table uses one result per current layout. Two early guard contacts also occurred under zero commands before the clearance corrections. Dishwasher B originally ended at step 1. Panel-corner B originally ended at step 9, compared with step 8 under zero commands. Both updated layouts passed 100 control steps with standing joint and pelvis targets before timing out in the GR00T reruns.

| Workcell | Layout A | Layout B |
|---|---|---|
| `reverse_gap` | time out at 30 s | time out at 30 s |
| `shelf_bay` | time out at 30 s | time out at 30 s |
| `partition_delivery` | guard contact at 3.98 s | guard contact at 3.14 s |
| `cabinet_reach` | time out at 30 s | time out at 30 s |
| `tote_extraction` | time out at 30 s | time out at 30 s |
| `hanging_pan` | time out at 30 s | time out at 30 s |
| `valve_turn` | time out at 30 s | time out at 30 s |
| `seam_inspection` | time out at 30 s | time out at 30 s |
| `panel_catches` | time out at 30 s | time out at 30 s |
| `dishwasher` | time out at 30 s | time out at 30 s |
| `billet_insertion` | time out at 30 s | time out at 30 s |
| `rail_support` | time out at 30 s | time out at 30 s |
| `stack_bracing` | time out at 30 s | time out at 30 s |
| `worktop_clamp` | time out at 30 s | time out at 30 s |
| `loaded_drawer` | guard contact at 6.54 s | time out at 30 s |
| `pipe_bracket` | time out at 30 s | time out at 30 s |
| `panel_removal` | time out at 30 s | time out at 30 s |
| `hatch_transfer` | time out at 30 s | time out at 30 s |
| `probe_withdrawal` | time out at 30 s | time out at 30 s |
| `display_cover` | time out at 30 s | time out at 30 s |
| `scoop_pour` | time out at 30 s | time out at 30 s |
| `panel_corner` | time out at 30 s | time out at 30 s |
| `aisle_retreat` | time out at 30 s | time out at 30 s |
| `platform_step` | time out at 30 s | time out at 30 s |
| `beam_clearance` | time out at 30 s | time out at 30 s |
| `trolley_dock` | time out at 30 s | time out at 30 s |
| `pillar_pass` | time out at 30 s | time out at 30 s |
| `closing_door` | time out at 30 s | time out at 30 s |
| `moving_carriage` | time out at 30 s | time out at 30 s |
| `descending_hood` | guard contact at 14.24 s | time out at 30 s |
