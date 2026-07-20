Sensitivity Analysis
====================

The sensitivity-analysis toolbox answers a single question about a policy:
*which environment conditions drive success?* Given the per-episode results of an
evaluation sweep — where factors such as lighting, object mass, or table material were
varied — it fits a posterior over those factors conditioned on the outcome (e.g. success
rate) and renders one figure summarising which factor values are associated with success.

Two distinct ideas are at work. *Joint* means all factors are modelled together rather than
one at a time, which is what captures interactions and confounds (see the next section).
*Posterior* means the result is conditioned on the outcome: starting from the prior — the
factor values the sweep actually drew, uniform over their observed ranges — it reweights them
by how often each led to the chosen outcome. So the figure answers *given success, which
factor values were in play?*, not merely *how were the factors distributed in the sweep?*

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

1. **Per-episode input.** The analysis reads a single ``episode_results.jsonl`` — one row per
   episode, holding that episode's recorded variation draws and outcomes.
2. **Schema discovery.** The factors are discovered from the data: each entry in a row's
   ``variations`` block becomes a factor — a number is continuous, a numeric vector splits into
   one continuous factor per component, and a string is categorical (its choices are the labels
   observed across the sweep). Continuous ranges are taken from the data's min/max. There is no
   schema file to author; *which* outcome to condition on is chosen at analysis time.
3. **Inference.** ``SensitivityAnalyzer`` trains an estimator on the full ``(theta, x)`` jointly
   — sbi's terms for the factor values (``theta``) and the per-episode outcomes (``x``) — and
   samples the joint posterior conditioned on a chosen observation (by default, success).
4. **Report.** A probability density curve for each continuous factor and a probability bar
   chart for each categorical factor.

Input
-----

The analysis reads a single ``episode_results.jsonl`` written by the per-episode recorder —
one JSON object per episode. Each row's ``variations`` block holds the sampled factor draws,
and the top-level fields named by ``--outcome`` hold the outcomes (any other top-level fields
are ignored):

.. code-block:: json

   {"job_name": "pi0_sweep", "episode_in_env": 0, "success": true,
    "variations": {"light_intensity": 3200.0, "table_material": "oak",
                   "wrist_camera": [0.01, -0.02, 0.0]}}

The factor schema is discovered from these values, so there is no separate schema file: a
number becomes a continuous factor, a numeric vector splits into one continuous factor per
component (named ``key[0]``, ``key[1]``, …), and a string becomes a categorical factor whose
choices are the labels observed across the sweep. A factor that took a single value across
all episodes carries no information and is dropped.

Choice of estimator
-------------------

``SensitivityAnalyzer`` picks the estimator from the discovered factors automatically:

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
     - Models the joint posterior over the continuous factors.

Continuous factors are normalised to ``[0, 1]`` before fitting and de-normalised when
sampling, so factors on very different scales (e.g. light in the thousands, an offset in
the hundredths) train on equal footing. Outcomes are binary (0/1); the default query
conditions on success (1).

Running a report
----------------

Point the report generator at an ``episode_results.jsonl``. The output format follows the
file extension (``.png``, ``.pdf``, …); reports are written under ``eval/`` by default.

.. code-block:: bash

   python -m isaaclab_arena.analysis.sensitivity.generate_report \
     --episode_results episode_results.jsonl \
     --outcome success \
     --output eval/sensitivity_report.png

``--outcome`` selects which per-episode outcome(s) to condition on (top-level field(s) in
each row); it defaults to ``success``. Pass ``--observation`` to set the value per outcome —
since outcomes are binary, use ``1`` for success or ``0`` for failure; it defaults to ``1``
(success). ``--factors`` restricts the analysis to a subset of the recorded variations (by
their ``variations``-block names; a vector variation keeps all its components); by default
every recorded variation is analyzed.

Trying it on synthetic data
---------------------------

A synthetic simulator with a *known* ground truth lets you run the whole pipeline without
Isaac Sim — useful for seeing the output shape and for validating the toolbox
(the recovered posterior should reflect the planted relationship):

.. code-block:: bash

   # mixed: three continuous + two categorical factors (MNPE)
   python -m isaaclab_arena.tests.sensitivity_synthetic --kind mixed --output eval/demo.png

``--kind`` also accepts ``continuous`` (continuous-only factors, which exercises the NPE path).

Reading the output
------------------

.. todo::

   Add a sample report figure here and walk through reading it.

Each panel is the posterior over one factor *conditioned on success*. Intuitively it answers
"given the policy succeeded, which values of this factor were responsible?" More precisely,
among the successful episodes it shows the probability density that the factor took each
value. For a continuous factor, mass concentrated at one end of its range means success
favoured that end — e.g. a curve rising toward bright light means successful episodes were
almost all bright ones, i.e. the policy needs bright light to succeed.
For a categorical factor, the tallest bar is the value most associated with success.

Current scope
-------------

- Outcomes are treated as **binary** (0/1). Conditioning defaults to success; a continuous
  outcome is rejected with a clear error rather than silently averaged.
- A **vector** variation draw (e.g. a camera pose offset) is split into one scalar factor per
  component (``key[0]``, ``key[1]``, …), each analysed independently. Components are named by
  position; semantic names (e.g. a camera's lateral vs. depth axis) are a future extension.
- **Factors should be drawn from the prior** the analyzer assumes — uniform over each
  continuous range, and an equal number of episodes per categorical choice. The posterior is
  taken relative to how the sweep drew the factors, so uneven sampling leaks in: a factor with
  no real effect comes out flat only if it was sampled flat, otherwise its posterior tracks the
  sampling frequency. The analyzer warns when a categorical is sampled unevenly, but the clean
  fix is to balance the draws in the sweep.
- The estimators run on CPU and do not require Isaac Sim, so a report can be generated
  anywhere the evaluation JSONL is available.
- The analysis assumes the ``episode_results.jsonl`` is a single coherent slice — one
  policy, task, and embodiment. **TODO:** add a filter (in the spirit of robolab's
  ``--filter-policy`` / ``--filter-task``) to select that slice from a larger JSONL,
  rather than relying on the caller to pre-filter it.
