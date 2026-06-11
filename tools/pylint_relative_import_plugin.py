# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pylint plugin that flags relative imports (restores removed built-in W0403)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astroid import nodes
from pylint.checkers import BaseChecker

if TYPE_CHECKING:
    from pylint.lint import PyLinter

_RELATIVE_IMPORT = "relative-import"


class RelativeImportChecker(BaseChecker):
    """Emit ``relative-import`` when a ``from .module import ...`` is used."""

    name = "relative_import"
    msgs = {
        "W0403": (
            "Relative import %r, use absolute import instead.",
            _RELATIVE_IMPORT,
            "Relative imports are discouraged outside package ``__init__.py`` files.",
        ),
    }

    def visit_importfrom(self, node: nodes.ImportFrom) -> None:
        if node.level and node.level > 0:
            self.add_message(_RELATIVE_IMPORT, node=node, args=(node.as_string(),))


def register(linter: PyLinter) -> None:
    linter.register_checker(RelativeImportChecker(linter))
