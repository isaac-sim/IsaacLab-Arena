# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Decorators tagging predicate functions by composition level.

Atomic predicates query scene state directly (poses, contacts, velocities).
Composite predicates are built by combining atomic ones (and/or/not, sequencing).
Tagging is purely informational — used by introspection, docs, and difficulty
scoring — and does not change function behavior.
"""


def atomic(fn):
    fn.type = "atomic"
    return fn


def composite(fn):
    fn.type = "composite"
    return fn
