# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``isaaclab_arena/scripts/imitation_learning/merge_demos.py``.

The merge script is pure ``h5py`` + ``numpy``, so every test below runs in the
no-cameras / no-subprocess Phase 1 slot. Synthetic fixtures are built with the local
:func:`_make_dataset` helper to mirror ``record_demos.py`` HDF5 output without
needing Isaac Sim, the simulator app, or any real recording on disk.
"""

import h5py
import importlib.util
import json
import numpy as np
import os
import sys

import pytest

_MERGE_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "imitation_learning",
    "merge_demos.py",
)
_spec = importlib.util.spec_from_file_location("merge_demos", _MERGE_SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None, f"Could not load {_MERGE_SCRIPT_PATH}"
merge_demos = importlib.util.module_from_spec(_spec)
sys.modules["merge_demos"] = merge_demos
_spec.loader.exec_module(merge_demos)


def _make_dataset(
    path,
    *,
    num_demos=3,
    action_dim=8,
    obs_terms=("robot_joint_pos",),
    obs_dim=7,
    camera_shapes=None,
    format_version=1,
    env_name="",
    sim_dt=0.01,
    decimation=2,
    success=True,
    extra_top_level_demo_attrs=None,
    extra_obs_keys=(),
    base_episode_len=100,
):
    """Build a ``record_demos.py``-shaped synthetic HDF5 file at ``path``.

    The structure mirrors what
    ``isaaclab.utils.datasets.HDF5DatasetFileHandler`` writes when invoked by
    ``ActionStateRecorderManagerCfg`` + optional ``ArenaEnvRecorderManagerCfg``.
    """
    rng = np.random.default_rng(seed=42)
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = format_version
        data = f.create_group("data")
        env_args = {
            "env_name": env_name,
            "type": 2,
            "sim_args": {
                "dt": sim_dt,
                "decimation": decimation,
                "render_interval": 1,
                "num_envs": 1,
            },
        }
        data.attrs["env_args"] = json.dumps(env_args)
        total = 0
        for i in range(num_demos):
            T = base_episode_len + i
            demo = data.create_group(f"demo_{i}")
            demo.attrs["num_samples"] = T
            if success is not None:
                demo.attrs["success"] = success
            if extra_top_level_demo_attrs:
                for k, v in extra_top_level_demo_attrs.items():
                    demo.attrs[k] = v
            demo.create_dataset(
                "actions",
                data=rng.standard_normal((T, action_dim)).astype(np.float32),
                compression="gzip",
            )
            demo.create_dataset(
                "processed_actions",
                data=rng.standard_normal((T, action_dim)).astype(np.float32),
                compression="gzip",
            )
            obs = demo.create_group("obs")
            for term in obs_terms:
                obs.create_dataset(
                    term,
                    data=rng.standard_normal((T, obs_dim)).astype(np.float32),
                    compression="gzip",
                )
            for term in extra_obs_keys:
                obs.create_dataset(
                    term,
                    data=rng.standard_normal((T, obs_dim)).astype(np.float32),
                    compression="gzip",
                )
            if camera_shapes:
                cam = demo.create_group("camera_obs")
                for name, (H, W) in camera_shapes.items():
                    cam.create_dataset(
                        name,
                        data=rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8),
                        compression="gzip",
                    )
            total += T
        data.attrs["total"] = total


def _run_merge(argv):
    return merge_demos.main(argv)


def _h5_demo_names(path):
    with h5py.File(path, "r") as f:
        return sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[-1]),
        )


def test_two_file_happy_path(tmp_path):
    a = tmp_path / "a.hdf5"
    b = tmp_path / "b.hdf5"
    out = tmp_path / "merged.hdf5"
    _make_dataset(str(a), num_demos=3, base_episode_len=100)
    _make_dataset(str(b), num_demos=2, base_episode_len=200)

    assert _run_merge([str(a), str(b), "-o", str(out)]) == 0
    assert out.exists()

    with h5py.File(out, "r") as f:
        assert int(f.attrs["format_version"]) == 1
        data = f["data"]
        demo_names = sorted([k for k in data.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[-1]))
        assert demo_names == [f"demo_{i}" for i in range(5)]
        # a: 100 + 101 + 102 = 303; b: 200 + 201 = 401
        assert int(data.attrs["total"]) == 303 + 401
        for name in demo_names:
            assert bool(data[name].attrs["success"]) is True


def test_three_file_input_order(tmp_path):
    a = tmp_path / "a.hdf5"
    b = tmp_path / "b.hdf5"
    c = tmp_path / "c.hdf5"
    out = tmp_path / "merged.hdf5"
    for path, marker in ((a, 1.0), (b, 2.0), (c, 3.0)):
        _make_dataset(str(path), num_demos=1, base_episode_len=10)
        with h5py.File(path, "r+") as f:
            f["data/demo_0/actions"][...] = marker

    assert _run_merge([str(a), str(b), str(c), "-o", str(out)]) == 0
    with h5py.File(out, "r") as f:
        assert np.allclose(f["data/demo_0/actions"][...], 1.0)
        assert np.allclose(f["data/demo_1/actions"][...], 2.0)
        assert np.allclose(f["data/demo_2/actions"][...], 3.0)


def test_dry_run_writes_nothing(tmp_path):
    a = tmp_path / "a.hdf5"
    out = tmp_path / "merged.hdf5"
    _make_dataset(str(a))
    assert _run_merge([str(a), "-o", str(out), "--dry_run"]) == 0
    assert not out.exists()


def test_overwrite_protects_existing(tmp_path):
    a = tmp_path / "a.hdf5"
    out = tmp_path / "merged.hdf5"
    _make_dataset(str(a))
    out.write_text("placeholder")

    assert _run_merge([str(a), "-o", str(out)]) != 0
    assert out.read_text() == "placeholder"

    assert _run_merge([str(a), "-o", str(out), "--overwrite"]) == 0
    with h5py.File(out, "r") as f:
        assert "data" in f


def test_output_equals_input_rejected(tmp_path):
    a = tmp_path / "a.hdf5"
    _make_dataset(str(a), num_demos=2)
    assert _run_merge([str(a), "-o", str(a)]) != 0
    # input untouched and still readable
    assert _h5_demo_names(a) == ["demo_0", "demo_1"]


def test_single_input_file_works(tmp_path):
    a = tmp_path / "a.hdf5"
    out = tmp_path / "merged.hdf5"
    _make_dataset(str(a), num_demos=4)
    assert _run_merge([str(a), "-o", str(out)]) == 0
    assert _h5_demo_names(out) == [f"demo_{i}" for i in range(4)]


def test_format_version_mismatch(tmp_path):
    a = tmp_path / "a.hdf5"
    b = tmp_path / "b.hdf5"
    out = tmp_path / "merged.hdf5"
    _make_dataset(str(a), format_version=0)
    _make_dataset(str(b), format_version=1)
    assert _run_merge([str(a), str(b), "-o", str(out)]) != 0
    assert not out.exists()


def test_format_version_mismatch_not_silenced_by_schema_flag(tmp_path):
    """format_version is always hard, even with --allow_schema_mismatch."""
    a = tmp_path / "a.hdf5"
    b = tmp_path / "b.hdf5"
    out = tmp_path / "merged.hdf5"
    _make_dataset(str(a), format_version=0)
    _make_dataset(str(b), format_version=1)
    assert _run_merge([str(a), str(b), "-o", str(out), "--allow_schema_mismatch"]) != 0


def test_action_dim_mismatch_strict(tmp_path):
    a = tmp_path / "a.hdf5"
    b = tmp_path / "b.hdf5"
    out = tmp_path / "merged.hdf5"
    _make_dataset(str(a), action_dim=8)
    _make_dataset(str(b), action_dim=6)
    assert _run_merge([str(a), str(b), "-o", str(out)]) != 0
    assert not out.exists()


def test_action_dim_mismatch_allowed(tmp_path):
    a = tmp_path / "a.hdf5"
    b = tmp_path / "b.hdf5"
    out = tmp_path / "merged.hdf5"
    _make_dataset(str(a), action_dim=8, num_demos=3)
    _make_dataset(str(b), action_dim=6, num_demos=2)

    assert _run_merge([str(a), str(b), "-o", str(out), "--allow_schema_mismatch"]) == 0
    with h5py.File(out, "r") as f:
        assert f["data/demo_0/actions"].shape[1] == 8
        assert f["data/demo_3/actions"].shape[1] == 6


def test_missing_obs_key_strict(tmp_path):
    a = tmp_path / "a.hdf5"
    b = tmp_path / "b.hdf5"
    out = tmp_path / "merged.hdf5"
    _make_dataset(str(a), obs_terms=("robot_joint_pos", "left_eef_pos"))
    _make_dataset(str(b), obs_terms=("robot_joint_pos",))
    assert _run_merge([str(a), str(b), "-o", str(out)]) != 0


def test_camera_shape_mismatch(tmp_path):
    a = tmp_path / "a.hdf5"
    b = tmp_path / "b.hdf5"
    out = tmp_path / "merged.hdf5"
    _make_dataset(
        str(a),
        camera_shapes={"robot_head_cam_rgb": (24, 32)},
        base_episode_len=5,
        num_demos=1,
    )
    _make_dataset(
        str(b),
        camera_shapes={"robot_head_cam_rgb": (48, 64)},
        base_episode_len=5,
        num_demos=1,
    )
    assert _run_merge([str(a), str(b), "-o", str(out)]) != 0


def test_env_args_sim_dt_mismatch(tmp_path, capsys):
    a = tmp_path / "a.hdf5"
    b = tmp_path / "b.hdf5"
    out = tmp_path / "merged.hdf5"
    _make_dataset(str(a), sim_dt=0.01)
    _make_dataset(str(b), sim_dt=0.02)
    assert _run_merge([str(a), str(b), "-o", str(out)]) == 0
    # Single readouterr() — calling it twice clears the buffer and the second call returns
    # empty strings, so any stderr output would be silently dropped.
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    # The mismatch should surface as a WARNING or in the validation status row
    assert "sim_args" in combined or "WARN" in combined


def test_empty_input_rejected(tmp_path):
    empty = tmp_path / "empty.hdf5"
    out = tmp_path / "merged.hdf5"
    with h5py.File(empty, "w") as f:
        f.attrs["format_version"] = 1
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "", "type": 2})
        data.attrs["total"] = 0
    assert _run_merge([str(empty), "-o", str(out)]) != 0


def test_forward_compat_new_key(tmp_path):
    """An unknown future recorder key must round-trip via the recursive copy."""
    a = tmp_path / "a.hdf5"
    b = tmp_path / "b.hdf5"
    out = tmp_path / "merged.hdf5"
    _make_dataset(str(a), extra_obs_keys=("future_unknown_term",))
    _make_dataset(str(b), extra_obs_keys=("future_unknown_term",))

    assert _run_merge([str(a), str(b), "-o", str(out)]) == 0
    with h5py.File(out, "r") as f:
        for name in f["data"].keys():
            if not name.startswith("demo_"):
                continue
            assert "future_unknown_term" in f[f"data/{name}/obs"]


def test_extra_top_level_demo_attr(tmp_path):
    a = tmp_path / "a.hdf5"
    out = tmp_path / "merged.hdf5"
    _make_dataset(str(a), extra_top_level_demo_attrs={"recording_timestamp": 1700000000})

    assert _run_merge([str(a), "-o", str(out)]) == 0
    with h5py.File(out, "r") as f:
        assert int(f["data/demo_0"].attrs["recording_timestamp"]) == 1700000000


def test_legacy_file_without_success_attr(tmp_path):
    a = tmp_path / "a.hdf5"
    out = tmp_path / "merged.hdf5"
    _make_dataset(str(a), success=None)
    assert _run_merge([str(a), "-o", str(out)]) == 0


def test_summary_log_format(tmp_path, capsys):
    a = tmp_path / "session_a.hdf5"
    b = tmp_path / "session_b.hdf5"
    out = tmp_path / "merged.hdf5"
    _make_dataset(str(a), num_demos=3)
    _make_dataset(str(b), num_demos=2)

    assert _run_merge([str(a), str(b), "-o", str(out)]) == 0
    text = capsys.readouterr().out
    assert "session_a.hdf5" in text
    assert "session_b.hdf5" in text
    assert "[1/2]" in text
    assert "[2/2]" in text
    assert "merged.hdf5" in text
    assert "Validation" in text
    assert "format_version OK" in text
    assert "demo_0" in text
    assert "demo_4" in text


def test_total_attr_recomputed_from_truth(tmp_path):
    """Even if data.attrs['total'] is wrong on input, output recomputes from per-demo num_samples."""
    a = tmp_path / "a.hdf5"
    out = tmp_path / "merged.hdf5"
    _make_dataset(str(a), num_demos=2, base_episode_len=50)
    with h5py.File(a, "r+") as f:
        f["data"].attrs["total"] = 99999

    assert _run_merge([str(a), "-o", str(out)]) == 0
    with h5py.File(out, "r") as f:
        assert int(f["data"].attrs["total"]) == 50 + 51


def test_output_parent_directory_created(tmp_path):
    """The script should mkdir -p the output's parent directory automatically."""
    a = tmp_path / "a.hdf5"
    _make_dataset(str(a))
    nested_out = tmp_path / "fresh" / "subdir" / "merged.hdf5"
    assert not nested_out.parent.exists()

    assert _run_merge([str(a), "-o", str(nested_out)]) == 0
    assert nested_out.exists()


def test_demo_without_step_info_warns(tmp_path, capsys):
    """A demo with no num_samples attr and no actions dataset should produce a warning."""
    a = tmp_path / "a.hdf5"
    _make_dataset(str(a), num_demos=2)
    # Strip num_samples and remove the actions dataset from demo_1 to simulate a broken demo
    with h5py.File(a, "r+") as f:
        demo = f["data/demo_1"]
        del demo.attrs["num_samples"]
        del demo["actions"]

    out = tmp_path / "merged.hdf5"
    assert _run_merge([str(a), "-o", str(out)]) == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "demo_1" in combined
    assert "num_samples" in combined or "actions" in combined


def test_merged_file_loads_via_isaaclab_handler(tmp_path):
    """Property test: merged file must be readable by isaaclab's HDF5DatasetFileHandler."""
    try:
        from isaaclab.utils.datasets import HDF5DatasetFileHandler
    except ImportError as e:
        pytest.skip(f"isaaclab.utils.datasets not importable in this context: {e}")

    a = tmp_path / "a.hdf5"
    b = tmp_path / "b.hdf5"
    out = tmp_path / "merged.hdf5"
    _make_dataset(str(a), num_demos=2)
    _make_dataset(str(b), num_demos=3)

    assert _run_merge([str(a), str(b), "-o", str(out)]) == 0

    handler = HDF5DatasetFileHandler()
    handler.open(str(out), mode="r")
    try:
        assert handler.get_num_episodes() == 5
        for name in handler.get_episode_names():
            episode = handler.load_episode(name, device="cpu")
            assert episode is not None
            assert bool(episode.success) is True
            assert "actions" in episode.data
    finally:
        handler.close()
