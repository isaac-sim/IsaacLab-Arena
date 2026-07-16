# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test formatting composed configurations as Hydra override help."""

from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from pathlib import Path

import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import OmegaConf

from isaaclab_arena.hydra.config_override_help import format_config_override_help, print_config_override_help


class _Mode(Enum):
    FAST = "fast-value"
    SAFE = "safe-value"


@dataclass
class _WorkerCfg:
    enabled: bool = True
    retries: int | None = None
    mode: _Mode = _Mode.FAST
    labels: list[str] = field(default_factory=lambda: ["first", "two words"])
    output_path: Path = Path("results/output")
    hidden: str = field(default="internal", metadata={"override_help": False})
    derived: str = field(init=False, default="derived")


@dataclass
class _RootCfg:
    workers: dict[str, _WorkerCfg] = field(
        default_factory=lambda: {
            "primary": _WorkerCfg(),
            "backup-worker": _WorkerCfg(enabled=False, mode=_Mode.SAFE),
        }
    )
    empty_mapping: dict[str, str] = field(default_factory=dict)


def test_formats_nested_dataclasses_and_named_mappings_in_declaration_order():
    """Render copyable Hydra paths while preserving the composed config order."""
    lines = format_config_override_help(_RootCfg()).splitlines()

    assert lines == [
        "workers.primary.enabled=true",
        "workers.primary.retries=null",
        'workers.primary.mode="FAST"',
        'workers.primary.labels=["first","two words"]',
        'workers.primary.output_path="results/output"',
        "workers.backup-worker.enabled=false",
        "workers.backup-worker.retries=null",
        'workers.backup-worker.mode="SAFE"',
        'workers.backup-worker.labels=["first","two words"]',
        'workers.backup-worker.output_path="results/output"',
        "empty_mapping={}",
    ]
    OverridesParser.create().parse_overrides(lines)
    recomposed = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(_RootCfg()), OmegaConf.from_dotlist(lines)))
    assert recomposed == _RootCfg()


def test_supports_a_parent_path_and_print_stream():
    """Apply a parent path and print the same representation to a selected stream."""
    output = StringIO()

    print_config_override_help(
        _WorkerCfg(),
        prefix="submission.worker",
        excluded_paths={"submission.worker.enabled"},
        file=output,
    )

    assert output.getvalue().startswith("submission.worker.retries=null\n")
    assert "submission.worker.enabled" not in output.getvalue()
    assert "submission.worker.hidden" not in output.getvalue()
    assert "submission.worker.derived" not in output.getvalue()


def test_rejects_non_dataclass_roots_and_unsupported_values():
    """Fail instead of presenting values that cannot form Hydra overrides."""
    with pytest.raises(AssertionError, match="dataclass instance"):
        format_config_override_help({"value": 1})

    @dataclass
    class _UnsupportedCfg:
        value: object = field(default_factory=object)

    with pytest.raises(TypeError, match="value"):
        format_config_override_help(_UnsupportedCfg())


def test_formats_mapping_keys_that_cannot_form_dotted_paths():
    """Present an irregular mapping as one replaceable value instead of crashing."""

    @dataclass
    class _IrregularMappingCfg:
        values: dict[str, int] = field(default_factory=lambda: {"not.a.path": 3, "two words": 4})

    line = format_config_override_help(_IrregularMappingCfg())
    parsed_override = OverridesParser.create().parse_overrides([line])[0]

    assert line == r"values={not.a.path:3,two\ words:4}"
    assert parsed_override.value() == {"not.a.path": 3, "two words": 4}
