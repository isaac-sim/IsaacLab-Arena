Agentic Environment Generation
==============================

Agentic environment generation translates a natural-language description of a
robot task into a validated ``ArenaEnvGraphSpec``. The spec is an editable,
serializable description of the assets, spatial relations, and tasks that Arena
uses to construct and evaluate a simulation environment.

The generation system supports both an interactive GUI for human review and a
CLI for scripted generation and execution.

.. toctree::
   :maxdepth: 1

   system_overview
   gui_runner
   cli_runner

.. note::

   Agentic environment generation is experimental. Generated specs should be
   reviewed and validated before they are used for policy evaluation.
