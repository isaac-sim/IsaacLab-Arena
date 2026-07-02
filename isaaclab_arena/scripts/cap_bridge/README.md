# Legacy GaP Bridge Diagnostics

These scripts are milestone diagnostics from the original external-server bridge. They are retained
for targeted frame, camera, and joint-control debugging, but they are not the supported CAP launch
path.

The canonical integration lives in the `Isaac-cap` superproject. Its `scripts/bootstrap.sh` checks
out this Arena revision and the pinned GaP repositories, and `scripts/run_arena_eval.sh` launches the
managed evaluation. `GapRemotePolicy` then allocates an ephemeral loopback port and owns one GaP
process per episode; no manual second terminal or fixed port is required.

| Script | Diagnostic scope |
|---|---|
| `m1_env_smoke.py` | Environment, joints, and wrist-camera smoke |
| `m2_joint_command.py` | Absolute joint-command response |
| `m3_bridge_client.py` | Legacy external-server motion bridge |
| `m4_framecheck.py` | Camera-to-robot-base frame validation |
| `m4_bridge_client.py` | Legacy RGB-D bridge and optional recording |

For reproducible setup, source pins, current commands, and accepted results, follow `README.md`,
`docs/TEAMMATE_SETUP.md`, and `STATUS.md` in the `Isaac-cap` checkout. Treat these milestone scripts
as diagnostics only.
