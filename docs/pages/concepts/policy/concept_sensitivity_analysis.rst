Sensitivity Analysis
====================

The sensitivity-analysis toolbox answers a single question about a policy:
*which environment conditions drive success?* Given the per-episode results of an
evaluation sweep — where factors such as lighting, object mass, or table material were
varied — it fits a posterior over those factors conditioned on the outcome and renders
one figure summarising which factor values are associated with success.

Why a joint posterior, not a success rate per factor?
-----------------------------------------------------

The simplest analysis would chart a success rate for each factor independently. That hides
the two things that matter most in a multi-factor sweep:

- **Factors interact.** How much light a policy needs can depend on the object — a matte
  object may succeed at low light while a shiny one needs far more. A per-factor
  "success vs light" curve averages over objects and reports one blurry gate that is wrong
  for both. The joint posterior keeps the interaction, so you can condition on a specific
  object and see its gate.
- **Factors confound each other.** If bright-light episodes also happened to use an easy
  object, a per-factor light chart cannot tell which one drove success. Modelling all
  factors together attributes the effect to the factor that actually carries it.

The per-factor rate is a projection of the joint posterior — derivable from it, but not the
other way around. The toolbox therefore always fits the joint — via simulation-based
inference (MNPE or NPE) — and reads the per-factor marginals from it.

How it works
------------

The toolbox is a thin analysis layer over `sbi <https://sbi.readthedocs.io>`_'s
neural posterior estimators. The flow is:

1. **Per-episode recording.** During evaluation, ``episode_writer`` appends one row per
   episode to an ``episode_summary.jsonl`` file.
2. **Schema.** A ``factors.yaml`` declares which of those columns are *factors* to analyse
   (and whether each is continuous or categorical) and which are *outcomes* to condition on.
3. **Inference.** ``SensitivityAnalyzer`` loads the pair, trains an estimator on the full
   ``(theta, x)`` jointly, and samples the joint posterior conditioned on a chosen
   observation (by default, success).
4. **Report.** A smooth density curve for each continuous factor and a probability bar chart
   for each categorical factor.

Inputs
------

**factors.yaml** declares the slice the data came from, the factors to study, and the
outcomes:

.. code-block:: yaml

   slice:
     policy: pi0
     task: PickUpObject
     embodiment: droid

   factors:
     light_intensity:
       type: continuous
       range: [[0.0, 5000.0]]   # one [low, high] pair; inferred from data if omitted
     table_material:
       type: categorical
       choices: [oak, walnut, bamboo]

   outcomes:
     success:
       type: bool

**episode_summary.jsonl** is produced by the eval runner — one JSON object per episode:

.. code-block:: json

   {"job_name": "pi0_sweep", "episode_idx": 0,
    "arena_env_args": {"light_intensity": 3200.0, "table_material": "oak"},
    "outcomes": {"success": 1}}

Choice of estimator
-------------------

``SensitivityAnalyzer`` picks the estimator from the schema automatically:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Schema
     - Estimator
     - Notes
   * - Any categorical factor
     - MNPE
     - Mixed density estimator; handles continuous + categorical factors together.
   * - All continuous factors
     - NPE
     - Restricts to a Gaussian on a single factor, so a meaningful continuous-only
       analysis needs at least two continuous factors.

Continuous factors are normalised to ``[0, 1]`` before fitting and de-normalised when
sampling, so factors on very different scales (e.g. light in the thousands, an offset in
the hundredths) train on equal footing. Outcomes are binary (0/1); the default query
conditions on success (1).

Running a report
----------------

Point the report generator at a ``(factors.yaml, episode_summary.jsonl)`` pair. The output
format follows the file extension (``.png``, ``.pdf``, …); reports are written under
``eval/`` by default.

.. code-block:: bash

   python -m isaaclab_arena.analysis.sensitivity.generate_report \
     --factors_yaml factors.yaml \
     --episode_summary episode_summary.jsonl \
     --output eval/sensitivity_report.png

Pass ``--observation`` to condition on specific outcome values (one per declared outcome,
in schema order); since outcomes are binary, use ``1`` for success or ``0`` for failure.
It defaults to ``1`` (success).

Trying it on synthetic data
---------------------------

A synthetic simulator with a *known* ground truth lets you run the whole pipeline on CPU,
without Isaac Sim — useful for seeing the output shape and for validating the toolbox
(the recovered posterior should reflect the planted relationship):

.. code-block:: bash

   # mixed (MNPE): one continuous + one categorical factor
   python -m isaaclab_arena.tests.utils.synthetic_sensitivity --kind mixed   --output eval/demo.png

   # continuous (NPE): two continuous factors
   python -m isaaclab_arena.tests.utils.synthetic_sensitivity --kind continuous --output eval/demo.png

   # rich: three continuous + two categorical factors
   python -m isaaclab_arena.tests.utils.synthetic_sensitivity --kind rich    --output eval/demo.png

Reading the output
------------------

.. todo::

   Add a sample report figure here and walk through reading it.

Each panel is the posterior over one factor *conditioned on success* — "given the policy
succeeded, which values of this factor were responsible?" For a continuous factor, mass
concentrated at one end of its range means success favoured that end (e.g. a curve rising
toward bright light → the policy is light-gated). For a categorical factor, the tallest
bar is the value most associated with success.

Current scope
-------------

- Outcomes are treated as **binary** (0/1). Conditioning defaults to success; a continuous
  outcome is rejected with a clear error rather than silently averaged.
- Continuous **vector** factors (``dim > 1``) are reserved for a future extension.
- The estimators run on CPU and do not require Isaac Sim, so a report can be generated
  anywhere the evaluation JSONL is available.
