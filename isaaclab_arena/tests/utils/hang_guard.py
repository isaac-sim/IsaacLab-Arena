# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import faulthandler
import sys
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def exit_if_stuck(timeout_s: float) -> Iterator[None]:
    """Dump the stack of every thread and exit the process if the body runs longer than ``timeout_s``.

    Unlike a Python-level timeout, this also fires when the main thread is blocked inside native code
    (for example a network call made by the simulator), which is where a hung test would otherwise
    wait until the CI job times out. Exiting ends the whole pytest process, since the simulation
    app cannot be reused after an interrupted test.

    Args:
        timeout_s: Maximum run time of the body, in seconds.
    """
    faulthandler.dump_traceback_later(timeout_s, exit=True, file=sys.stderr)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
