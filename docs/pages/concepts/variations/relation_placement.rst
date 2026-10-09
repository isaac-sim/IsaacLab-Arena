Relation Placement Variation
============================

Relation placement is a scene-level run-time variation that applies one
coordinated layout to every placed asset during an environment reset. See
:doc:`../object_placement/relations` for the available relations and solver
configuration.

Lifecycle
---------

``ArenaEnvBuilder`` declares ``scene.relation_placement`` before applying Hydra
overrides and binding recorded variation samples. The variation then prepares
the construction poses required before Isaac Lab materialises the scene:

* **Live sampling** creates an initially empty placement pool. The first
  construction-pose draw solves and fills the pool.
* **Recorded replay** seeds construction poses from the bound JSONL rows. Its
  recorded sampler overrides live draws, so the live pool remains unsolved.

After preparation, Arena materialises the scene and installs one reset event for
the complete placement. Each reset draws either a live pooled layout or the
layout assigned by the replay scheduler, records that variation sample, and
writes all owned scene roots through their placement assets.

Live Configuration
------------------

``scene.relation_placement.resample_on_reset`` controls only live placement:

* ``true`` draws a fresh pooled layout for each resetting environment;
* ``false`` restores the fixed per-environment layouts selected during
  construction preparation.

Recorded replay ignores this setting and always applies the scheduler's
assigned row. Configure replay with
``ArenaEnvBuilderCfg.recorded_variation_samples_path`` through an Experiment
Definition. The Policy Runner does not expose replay as a CLI option.

Recording and Replay
--------------------

Each episode JSONL row stores the complete placement under
``variations["scene.relation_placement"]``. A placement sample contains all
non-anchor scene-root poses, so compound assets may contribute multiple roots.
Replay validates root coverage and reset ownership before scene construction.

For the JSONL schema, replay order, and recording commands, see
:ref:`recorded-layouts`.
