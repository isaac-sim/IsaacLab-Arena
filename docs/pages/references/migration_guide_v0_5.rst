Migrating from v0.4 to v0.5
===========================

This guide lists the changes to Arena's public interfaces after ``v0.4.0``.
For new features, see the :doc:`release notes <release_notes>`.


Python environment API
----------------------

Breaking changes
^^^^^^^^^^^^^^^^

Lightwheel USD sources
~~~~~~~~~~~~~~~~~~~~~~

``LightwheelUsdSource`` was split by asset kind. Use ``LightwheelObjectUsdSource(...)`` for objects and
``LightwheelFloorplanUsdSource(...)`` for floorplans.

``LightwheelObjectUsdSource`` requires a ``file_name`` that pins one specific asset, so the same file is
returned every time. ``registry_name`` is an optional category filter.

A Lightwheel asset found in the SDK cache (``~/.cache/lightwheel_sdk``) is used without calling the
Lightwheel servers.


Experiment Runner
-----------------

Behavior changes
^^^^^^^^^^^^^^^^

- **Viewport videos:** in v0.4, ``--record_viewport_video`` wrote no video. It now records the Kit viewer,
  so it requires ``--viz kit``, which opens no window when ``DISPLAY`` is unset, and it enables camera
  support. Each rebuild writes ``viewport-rebuild<i>_0000.mp4`` to the Run directory.
  ``policy_runner.py --record_viewport_video`` writes ``viewport_0000.mp4``.
