# I2RT YAM asset snapshot

This directory contains the self-contained `i2rt_yam_default` USD package used
by the Arena YAM embodiment. The eight source USD files were copied on
2026-08-05 from NVIDIA's Robot Menagerie conversion of the public Google
DeepMind MuJoCo Menagerie YAM asset:

- conversion repository commit: `68ef1e0cc3e863a861b873893f038496e0dfe16b`
- last conversion commit touching `i2rt/yam`: `29806b31e41509731ffe6466547cba5cb09f6e63`
- conversion source directory: `i2rt/yam/generated/i2rt_yam_default/usd`
- entry layer: `i2rt_yam_default.usda`

The Menagerie manifest pins the original Google DeepMind MuJoCo Menagerie
source to `google-deepmind/mujoco_menagerie` commit
`71f066ad0be9cd271f7ed58c030243ef157af9f4`, subpath `i2rt_yam`, with source
digest `5ae901629db944abd3e6cbb0eaf28ea78bbda3483c840d6db1167e999cc0348c`.
That source is MIT licensed; the required notice is preserved in `LICENSE.md`.

All eight source USD files are included and all layer references are relative.
The `i2rt_yam_cable_routing.usda` task layer composes that unmodified snapshot,
retains its authored actuator dynamics, and targets the calibrated high-friction
contact material to the finger collision subtrees. OpenUSD dependency traversal
resolves every layer with no unresolved paths, so runtime does not require
GitLab, Git LFS, Nucleus, or credentials.

- entry-layer SHA-256: `c1bedf1d978d1147f82d1c2cb5e56da1b5003eb14ec78bd0be89258c021404bc`
- eight-file package size: `1,887,475` bytes
- deterministic package-manifest SHA-256: `7a1532b694a51f263a9e09b60d212d40104aee871bcfa6e8c6705fd566698d47`

The manifest digest is produced from the pristine eight-file source package
(and intentionally excludes the task layer) with:

```bash
find i2rt_yam i2rt_yam_default.usda -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum
```

The bundled snapshot is the deterministic default for credential-free local,
wheel, and OSMO execution.
