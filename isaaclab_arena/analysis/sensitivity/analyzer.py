# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Inference-only analyzers for v0.3 sensitivity analysis.

What this module does in plain English
--------------------------------------
Given a dataset of (factor values, outcome values) pairs from a policy evaluation, the
analyzer learns the *conditional* distribution of factor values given a chosen outcome
value (e.g. "given the episode succeeded, which factor values were most consistent?").
This is the **posterior** ``P(theta | outcome=success)``. Under v0.3's uniform prior,
this posterior's peak is also the operating point ``argmax P(success | theta)`` — so
plotting the marginal posterior over one factor identifies the values that maximize
success rate.

The three concrete analyzers cover the three relevant factor-mix cases:

  - ``NPEAnalyzer``       — **N**eural **P**osterior **E**stimation. Used when *all*
    declared factors are continuous. Trains a normalizing-flow density estimator on
    ``(theta, x)`` pairs and exposes ``posterior.sample`` / ``posterior.log_prob``.
    Limitation: with a binary outcome and a 1D theta, sbi falls back to a Gaussian
    density and the recovered peak reflects the *mean* of successful theta values
    rather than the true *mode* — a known caveat we surface as a [WARN] at fit time.
  - ``MNPEAnalyzer``      — **M**ixed **N**eural **P**osterior **E**stimation. Used when
    the schema has *both* continuous and categorical factors. sbi's MixedDensityEstimator
    routes continuous columns through the same kind of flow NPE uses while routing
    discrete columns through a categorical mass estimator.
  - ``EmpiricalAnalyzer`` — Pure-categorical schemas. Skip the neural fit entirely: under
    a uniform prior the posterior ``P(category | success)`` is *exactly* the normalized
    per-category empirical success rate. No smoothing improves on that, and sbi MNPE
    in version 0.26 also refuses to train without at least one continuous theta column.

``make_analyzer(dataset, outcome_name)`` is the factory: callers don't need to know about
the hierarchy, they just hand it a dataset and outcome name.

How rendering fits in
---------------------
This module is *inference-only*. The sibling ``plotting`` module reads the analyzer's
public queries (``continuous_marginal_density``, ``categorical_marginal_probs``) and
renders matplotlib figures. Decoupling the two means new plot types don't require
analyzer changes, and analyzer changes don't risk breaking the plot.

Public posterior-query surface used by ``plotting.py``:
  - ``BaseAnalyzer.categorical_marginal_probs(factor_name, outcome_value, num_samples)``
  - ``PosteriorAnalyzer.continuous_marginal_density(factor_name, outcome_value, num_grid_points)``
    (NOT defined on ``EmpiricalAnalyzer`` — that analyzer rejects continuous factors at init time)
"""

from __future__ import annotations

import numpy as np
import torch
from abc import ABC, abstractmethod

from isaaclab_arena.analysis.sensitivity.dataset import FactorSpec, SensitivityDataset


class BaseAnalyzer(ABC):
    """Abstract base — owns state validation and the abstract posterior-query surface.

    Subclasses must implement:
      - ``fit`` — train (or no-op) so queries can be called afterwards.
      - ``categorical_marginal_probs`` — return ``P(category | outcome)`` for a categorical factor.
    Continuous-factor queries (``continuous_marginal_density``) live on ``PosteriorAnalyzer``
    only — the empirical analyzer never needs them by construction.
    """

    def __init__(self, dataset: SensitivityDataset, outcome_name: str):
        self.dataset = dataset
        self.outcome_name = outcome_name
        assert (
            outcome_name in dataset.outcome_columns
        ), f"Outcome {outcome_name!r} not found in schema; available: {list(dataset.outcome_columns)}"
        assert len(dataset.schema.factors) > 0, "Schema declares no factors"

    @abstractmethod
    def fit(self, training_batch_size: int = 50) -> None:
        """Train the posterior (or no-op for empirical) so queries can be called afterwards.

        For NPE/MNPE this trains a neural density estimator on ``(theta, x_selected)``,
        where ``x_selected`` is the single outcome column named by ``outcome_name``. For
        the empirical analyzer this is a no-op — the categorical posterior is computed
        directly from the data at query time.
        """

    @abstractmethod
    def categorical_marginal_probs(self, factor_name: str, outcome_value: float, num_samples: int) -> np.ndarray:
        """Return ``P(category | outcome=outcome_value)`` for one categorical factor.

        Output is a 1D numpy array of length ``len(factor.choices)`` whose entries sum to 1.
        For posterior analyzers this is computed by sampling the trained posterior and
        counting category frequencies; for the empirical analyzer it's the normalized
        per-category empirical success rate.
        """

    def _factor_spec(self, factor_name: str) -> FactorSpec:
        """Return the ``FactorSpec`` for ``factor_name``, asserting it exists in the schema."""
        assert (
            factor_name in self.dataset.factor_columns
        ), f"Factor {factor_name!r} not in schema; available: {list(self.dataset.factor_columns)}"
        return next(factor for factor in self.dataset.schema.factors if factor.name == factor_name)


class _NullTracker:
    """A no-op tracker satisfying sbi's ``Tracker`` protocol — discards training metrics.

    By default sbi logs TensorBoard training curves under ``<cwd>/sbi-logs``
    (``get_log_root`` hardcodes the cwd). One-shot report generation fits each analyzer
    once and never reads those curves, yet the default tracker makes fitting fail with
    ``PermissionError`` whenever the cwd isn't writable — e.g. a repo checkout in a
    non-root container. Discarding the metrics removes the write, and with it the hidden
    cwd dependency, so analysis runs from any directory. sbi only calls ``log_metric``
    and ``flush``; the remaining members satisfy the protocol.
    """

    log_dir = None

    def log_metric(self, name, value, step=None):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_params(self, params):
        pass

    def add_figure(self, name, figure, step=None):
        pass

    def flush(self):
        pass


class PosteriorAnalyzer(BaseAnalyzer):
    """Common base for the sbi-driven analyzers (NPE and MNPE).

    NPE and MNPE differ only in *which* sbi inference class they instantiate; everything
    else (training loop, posterior storage, density and sample queries) is identical.
    Subclasses override ``_make_inference`` to choose the class, and the
    binary-outcome WARN hook to surface any method-specific caveats.

    After ``fit()`` returns, ``self.posterior`` is an sbi posterior object that supports
    ``posterior.sample(shape, x=...)`` and (for NPE) ``posterior.log_prob(theta, x=...)``.
    """

    def __init__(self, dataset: SensitivityDataset, outcome_name: str):
        super().__init__(dataset, outcome_name)
        self.posterior = None

    def _inference_cls(self):
        """Return the sbi inference *class* to train with (e.g. ``sbi.inference.NPE``).

        Subclass-specific: ``NPEAnalyzer`` returns ``NPE``, ``MNPEAnalyzer`` returns
        ``MNPE``. The lazy import of sbi lives in the subclass so callers don't pay the
        (heavy) sbi import cost until they actually fit.
        """
        raise NotImplementedError("PosteriorAnalyzer subclasses must implement _inference_cls")

    def _make_inference(self):
        """Instantiate the chosen sbi inference class on the dataset's uniform prior.

        Passes a ``_NullTracker`` so fitting writes no TensorBoard logs and stays
        independent of the launch directory — see ``_NullTracker`` for why.
        """
        return self._inference_cls()(prior=self.dataset.prior, tracker=_NullTracker())

    def fit(self, training_batch_size: int = 50) -> None:
        """Train the chosen sbi estimator on ``(theta, x_selected)`` and stash the posterior.

        Steps:
          1. Slice ``self.dataset.x`` to the single outcome column named by ``outcome_name``.
          2. Surface any method-specific caveats about the outcome (e.g. NPE's
             1D-theta Gaussian fallback) via ``_maybe_warn_binary_outcome``.
          3. Instantiate the sbi inference object (NPE or MNPE) via ``_make_inference``.
          4. Append the simulations and train.
          5. Build a posterior object from the trained estimator and store it on ``self``.
        """
        outcome_column_index = self.dataset.outcome_columns[self.outcome_name]
        selected_outcome_column = self.dataset.x[:, outcome_column_index : outcome_column_index + 1]
        self._maybe_warn_binary_outcome(selected_outcome_column)

        print(
            f"[INFO] {type(self).__name__}: fitting on {self.dataset.theta.shape[0]} samples"
            f" (theta dim={self.dataset.theta.shape[1]},"
            f" x dim={selected_outcome_column.shape[1]})."
        )
        inference = self._make_inference()
        inference.append_simulations(self.dataset.theta, selected_outcome_column)
        density_estimator = inference.train(training_batch_size=training_batch_size)
        self.posterior = inference.build_posterior(density_estimator)

    def _maybe_warn_binary_outcome(self, selected_outcome_column: torch.Tensor) -> None:
        """Optional hook for subclass-specific caveats about the outcome. Default: no-op.

        ``NPEAnalyzer`` overrides this to warn that sbi falls back to a Gaussian density
        when *theta* is 1D, biasing the recovered peak toward the mean of successful
        theta values rather than the true mode. The fallback fires regardless of how
        many outcome columns are logged — it's a property of single-factor analysis.
        For 1-continuous-factor + binary-outcome workloads, prefer ``KDEAnalyzer``.
        """

    def continuous_marginal_density(
        self, factor_name: str, outcome_value: float, num_grid_points: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate ``P(factor_value | outcome=outcome_value)`` over the factor's prior range.

        Returns ``(grid, density)`` as numpy arrays of length ``num_grid_points``, suitable
        for plotting as a smooth curve.

        Two evaluation paths depending on whether other factors are present:
          - **1D theta** (the only declared factor is this one): evaluate
            ``posterior.log_prob`` directly on a regular grid — exact, no sampling.
          - **Multi-dim theta**: sample the posterior at the given outcome value, extract
            this factor's column, and histogram-then-interpolate to a grid. This
            marginalizes over the other factor dims implicitly.
        """
        assert self.posterior is not None, "Call fit() before querying the posterior"
        factor_spec = self._factor_spec(factor_name)
        assert (
            factor_spec.type == "continuous"
        ), f"continuous_marginal_density expects a continuous factor; {factor_name!r} is {factor_spec.type!r}"
        assert (
            factor_spec.range is not None and len(factor_spec.range) == 1
        ), "Continuous-factor marginal expects a populated 1D range"

        factor_column_slice = self.dataset.factor_columns[factor_name]
        observed_outcome = torch.tensor([outcome_value], dtype=torch.float32)
        # ``factor_spec.range`` is always in linear (user-readable) units. For log_uniform
        # factors the analyzer's parameter space is log10(theta), so we generate the grid
        # in log10 space here and convert it back to linear before returning so the caller
        # plots intensities directly on a log-scale x-axis.
        is_log_uniform = factor_spec.distribution == "log_uniform"
        range_low, range_high = factor_spec.range[0]
        analyzer_low = np.log10(range_low) if is_log_uniform else range_low
        analyzer_high = np.log10(range_high) if is_log_uniform else range_high

        if self.dataset.theta.shape[1] == 1:
            grid_tensor = torch.linspace(analyzer_low, analyzer_high, num_grid_points, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                log_probabilities = self.posterior.log_prob(grid_tensor, x=observed_outcome)
            density_numpy = torch.exp(log_probabilities).cpu().numpy()
            grid_numpy = grid_tensor.squeeze(-1).cpu().numpy()
        else:
            with torch.no_grad():
                posterior_samples = self.posterior.sample((10_000,), x=observed_outcome)
            factor_column_samples = posterior_samples[:, factor_column_slice].squeeze(-1).cpu().numpy()
            grid_numpy = np.linspace(analyzer_low, analyzer_high, num_grid_points)
            histogram_density, bin_edges = np.histogram(
                factor_column_samples, bins=40, range=(analyzer_low, analyzer_high), density=True
            )
            density_numpy = np.interp(grid_numpy, 0.5 * (bin_edges[:-1] + bin_edges[1:]), histogram_density)

        if is_log_uniform:
            grid_numpy = np.power(10.0, grid_numpy)  # log10 → linear for display
        return grid_numpy, density_numpy

    def categorical_marginal_probs(self, factor_name: str, outcome_value: float, num_samples: int) -> np.ndarray:
        """Estimate ``P(category | outcome)`` by sampling the trained posterior.

        Draws ``num_samples`` from ``posterior(theta | x=outcome_value)``, extracts the
        factor's column (which sbi returns as floats over the BoxUniform support), rounds
        to the nearest integer in ``[0, num_choices - 1]``, and tallies frequencies.
        Result is a length-``num_choices`` numpy array that sums to 1.
        """
        assert self.posterior is not None, "Call fit() before querying the posterior"
        factor_spec = self._factor_spec(factor_name)
        assert factor_spec.type == "categorical"
        assert factor_spec.choices is not None
        factor_column_slice = self.dataset.factor_columns[factor_name]
        num_choices = len(factor_spec.choices)

        observed_outcome = torch.tensor([outcome_value], dtype=torch.float32)
        with torch.no_grad():
            posterior_samples = self.posterior.sample((num_samples,), x=observed_outcome)
        factor_column_samples = posterior_samples[:, factor_column_slice].squeeze(-1).cpu().numpy()
        clipped_codes = np.clip(np.round(factor_column_samples), 0, num_choices - 1).astype(int)
        return np.bincount(clipped_codes, minlength=num_choices) / num_samples


class NPEAnalyzer(PosteriorAnalyzer):
    """Neural Posterior Estimation analyzer for continuous-only factor schemas.

    Use this when every declared factor is continuous (no categoricals). Internally
    trains ``sbi.inference.NPE``, which fits a normalizing-flow density over
    ``(theta, x_selected)`` and exposes both ``sample`` and ``log_prob`` on the result.

    **Caveat for binary outcomes (1D x):** sbi's flow code falls back to a Gaussian
    density when the output space is 1D, which biases the recovered posterior peak
    toward the *mean* of successful theta values rather than the true *mode* of the
    success curve. We surface a [WARN] at fit time so users see this in plain text
    rather than buried in sbi's UserWarning stream.
    """

    def _inference_cls(self):
        from sbi.inference import NPE

        return NPE

    def _maybe_warn_binary_outcome(self, selected_outcome_column: torch.Tensor) -> None:
        """Warn if theta is 1D and the outcome is binary — the configuration that triggers
        the sbi Gaussian fallback. Multi-factor theta (dim ≥ 2) escapes the fallback.
        """
        if self.dataset.theta.shape[1] > 1:
            return
        unique_values = set(selected_outcome_column.flatten().tolist())
        if unique_values.issubset({0.0, 1.0}):
            print(
                f"[WARN] Theta is 1D ({self.dataset.schema.factors[0].name!r}) and outcome"
                f" {self.outcome_name!r} is binary. sbi NPE falls back to a Gaussian density"
                " in 1D theta space, so the recovered posterior peak reflects the *mean* of"
                " successful theta values rather than the true *mode* of the success curve."
                " For this configuration prefer KDEAnalyzer (uniform prior + binary outcome"
                " → KDE on successful-theta samples is the correct posterior)."
            )


class MNPEAnalyzer(PosteriorAnalyzer):
    """Mixed Neural Posterior Estimation analyzer for schemas with at least one of each type.

    Use this when the schema mixes continuous and categorical factors. Internally trains
    ``sbi.inference.MNPE``, whose mixed density estimator routes continuous theta columns
    through a normalizing flow while routing categorical columns through a categorical
    mass estimator. The continuous-first / categorical-after column ordering in
    ``factor_columns`` matches MNPE's expected layout exactly.

    sbi MNPE 0.26 requires at least one continuous theta column. For pure-categorical
    schemas use ``EmpiricalAnalyzer`` instead — ``make_analyzer`` dispatches correctly.
    """

    def _inference_cls(self):
        from sbi.inference import MNPE

        return MNPE


class EmpiricalAnalyzer(BaseAnalyzer):
    """Frequency-table analyzer for pure-categorical factor schemas — no neural fit.

    Use this when every declared factor is categorical. Under v0.3's uniform prior,
    Bayes' rule simplifies ``P(category | success) ∝ P(success | category) · P(category)``
    to ``P(category | success) ∝ P(success | category)`` — i.e. the posterior is *exactly*
    the per-category empirical success rate, normalized to sum to 1. No neural network
    can do better than this with a uniform prior; smoothing only hurts.

    Also covers a sbi limitation: MNPE 0.26 refuses to train if theta has zero continuous
    columns. The empirical path sidesteps that entirely.

    Rejects continuous factors at construction time — ``make_analyzer`` shouldn't even
    dispatch here for mixed schemas, but the explicit guard makes the constraint clear.
    """

    def __init__(self, dataset: SensitivityDataset, outcome_name: str):
        super().__init__(dataset, outcome_name)
        has_continuous_factor = any(factor.type == "continuous" for factor in dataset.schema.factors)
        assert not has_continuous_factor, (
            "EmpiricalAnalyzer is only valid for all-categorical schemas. For mixed"
            " continuous + categorical factors, use MNPEAnalyzer."
        )

    def fit(self, training_batch_size: int = 50) -> None:
        """No-op — the posterior is computed directly from the data at query time."""
        print(f"[INFO] {type(self).__name__}: no neural fit needed for pure-categorical schema.")

    def categorical_marginal_probs(self, factor_name: str, outcome_value: float, num_samples: int) -> np.ndarray:
        """Return ``P(category | outcome) = per_category_success_rate / sum(per_category_success_rate)``.

        For each category, computes the fraction of rows assigned to it whose outcome
        column is ``>= 0.5`` (treating outcome as binary). Then normalizes across
        categories so the result sums to 1. ``outcome_value`` and ``num_samples`` are
        accepted for interface compatibility with ``PosteriorAnalyzer`` but not used —
        empirical analysis treats outcome as binary (success vs not-success).
        """
        factor_spec = self._factor_spec(factor_name)
        assert factor_spec.type == "categorical"
        assert factor_spec.choices is not None
        factor_column_slice = self.dataset.factor_columns[factor_name]
        num_choices = len(factor_spec.choices)
        outcome_column_index = self.dataset.outcome_columns[self.outcome_name]

        empirical_theta_codes = self.dataset.theta[:, factor_column_slice].squeeze(-1).long().cpu().numpy()
        empirical_outcomes = self.dataset.x[:, outcome_column_index].cpu().numpy()
        empirical_rates = np.zeros(num_choices)
        for code in range(num_choices):
            category_mask = empirical_theta_codes == code
            if category_mask.any():
                empirical_rates[code] = float((empirical_outcomes[category_mask] >= 0.5).mean())
        total_rate = float(empirical_rates.sum())
        if total_rate > 0:
            return empirical_rates / total_rate
        return np.full(num_choices, 1.0 / num_choices)


class KDEAnalyzer(BaseAnalyzer):
    """KDE-based analyzer for the 1-continuous-factor + binary-outcome case.

    Under v0.3's uniform prior and a binary outcome, Bayes' rule reduces to
    ``P(theta | success=1) ∝ P(success=1 | theta) · P(theta) = P(success=1 | theta) · const``.
    The empirical density of *successful*-theta samples (i.e. rows where the chosen outcome
    is 1) is directly proportional to ``P(success=1 | theta)``, and a Gaussian KDE over
    those samples gives a smoothed estimate of that conditional density. No neural fit,
    no Gaussian-shape constraint.

    This is the right primitive for our MVP-1 case (1 continuous factor, binary outcome):
    sbi NPE forces a Gaussian shape when theta is 1D — biasing the recovered peak toward
    the mean of successful-theta values rather than the true mode of the success curve.
    KDE has no such constraint and recovers multi-modal / plateau / skewed shapes faithfully.

    Conceptual sibling of :class:`EmpiricalAnalyzer` (which does the same trick for purely
    categorical theta via frequency counts). For multi-factor or non-binary-outcome
    workloads, :func:`make_analyzer` dispatches to NPE/MNPE instead.

    Caveats:
      - Bandwidth is scipy's Scott rule default; haven't tuned for non-uniform sample
        distributions or sparse data. May over-smooth the empirical mode.
      - Only ``continuous_marginal_density`` is implemented and only for
        ``outcome_value >= 0.5`` (i.e. success conditioning). Failure conditioning would
        require fitting a second KDE over failed-theta samples; left out for v0.3 simplicity.
    """

    def __init__(self, dataset: SensitivityDataset, outcome_name: str):
        super().__init__(dataset, outcome_name)
        num_continuous = sum(1 for factor in dataset.schema.factors if factor.type == "continuous")
        num_categorical = sum(1 for factor in dataset.schema.factors if factor.type == "categorical")
        assert num_continuous == 1 and num_categorical == 0, (
            f"KDEAnalyzer requires exactly one continuous factor and no categoricals; got {num_continuous} continuous,"
            f" {num_categorical} categorical."
        )
        self._kde = None
        self._num_successful_samples = 0
        self._num_total_samples = 0

    def fit(self, training_batch_size: int = 50) -> None:
        """Fit a Gaussian KDE on the successful-theta samples (no neural network involved)."""
        from scipy.stats import gaussian_kde

        outcome_column_index = self.dataset.outcome_columns[self.outcome_name]
        outcome_values = self.dataset.x[:, outcome_column_index].cpu().numpy()
        theta_values = self.dataset.theta[:, 0].cpu().numpy()
        success_mask = outcome_values >= 0.5
        self._num_total_samples = int(len(theta_values))
        self._num_successful_samples = int(success_mask.sum())

        if self._num_successful_samples < 2:
            print(
                f"[WARN] KDEAnalyzer: only {self._num_successful_samples} successful samples"
                f" / {self._num_total_samples} total — KDE undefined, marginal will be uniform."
            )
            return

        successful_theta = theta_values[success_mask]
        if float(np.std(successful_theta)) < 1e-9:
            print(
                "[WARN] KDEAnalyzer: all successful theta values are identical — KDE bandwidth"
                " degenerate, marginal will be uniform."
            )
            return

        self._kde = gaussian_kde(successful_theta)
        print(
            f"[INFO] KDEAnalyzer: fit Gaussian KDE on {self._num_successful_samples} successful"
            f" theta samples / {self._num_total_samples} total."
        )

    def continuous_marginal_density(
        self, factor_name: str, outcome_value: float, num_grid_points: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the KDE-based posterior over the factor's prior range.

        ``outcome_value >= 0.5`` is treated as "success conditioning" (the only case
        currently supported); the KDE is evaluated on a uniform grid spanning the
        declared factor range. ``outcome_value < 0.5`` (failure conditioning) returns
        a uniform density as a placeholder — extend by fitting a second KDE on failed
        samples if/when that case is needed.
        """
        factor_spec = self._factor_spec(factor_name)
        assert factor_spec.type == "continuous", "KDEAnalyzer only handles continuous factors"
        assert (
            factor_spec.range is not None and len(factor_spec.range) == 1
        ), "Continuous-factor marginal expects a populated 1D range"
        # For log_uniform factors the analyzer's parameter space is log10(theta) (see
        # ``dataset._build_factor_tensor``), so the KDE was fit on log10 values and the
        # grid we evaluate on must also be in log space. We convert the linear grid back
        # for display at the very end so the caller plots intensities directly on a
        # log-scale x-axis.
        is_log_uniform = factor_spec.distribution == "log_uniform"
        range_low, range_high = factor_spec.range[0]
        analyzer_low = np.log10(range_low) if is_log_uniform else range_low
        analyzer_high = np.log10(range_high) if is_log_uniform else range_high
        analyzer_grid = np.linspace(analyzer_low, analyzer_high, num_grid_points)
        linear_grid = np.power(10.0, analyzer_grid) if is_log_uniform else analyzer_grid

        if outcome_value < 0.5 or self._kde is None:
            uniform_density = 1.0 / max(analyzer_high - analyzer_low, 1e-9)
            return linear_grid, np.full_like(analyzer_grid, uniform_density)
        return linear_grid, self._kde(analyzer_grid)

    def categorical_marginal_probs(self, factor_name: str, outcome_value: float, num_samples: int) -> np.ndarray:
        raise NotImplementedError(
            "KDEAnalyzer is for continuous factors only. Categorical schemas dispatch to"
            " EmpiricalAnalyzer via make_analyzer."
        )


def make_analyzer(dataset: SensitivityDataset, outcome_name: str) -> BaseAnalyzer:
    """Construct the right analyzer for the dataset's factor mix and outcome shape.

    Dispatch table (checked top-to-bottom):
      - 1 continuous + 0 categorical AND the outcome is binary → :class:`KDEAnalyzer`
        (avoids sbi NPE's 1D-theta Gaussian-shape constraint; exact under uniform prior)
      - any continuous + any categorical → :class:`MNPEAnalyzer`
      - all categorical (zero continuous) → :class:`EmpiricalAnalyzer`
      - all continuous (zero categorical) → :class:`NPEAnalyzer`
        (the multi-continuous-factor case; theta is multi-D so no Gaussian fallback)

    The KDE branch checks the outcome column's binary-ness on the dataset itself rather
    than the schema, since outcome ``type: float`` in factors.yaml covers both continuous
    durations and binary 0/1 success rates.
    """
    num_continuous_factors = sum(1 for factor in dataset.schema.factors if factor.type == "continuous")
    num_categorical_factors = sum(1 for factor in dataset.schema.factors if factor.type == "categorical")
    assert num_continuous_factors + num_categorical_factors > 0, "Schema declares no factors"

    if num_continuous_factors == 1 and num_categorical_factors == 0:
        outcome_column_index = dataset.outcome_columns[outcome_name]
        outcome_values = dataset.x[:, outcome_column_index]
        unique_outcome_values = set(outcome_values.flatten().tolist())
        if unique_outcome_values.issubset({0.0, 1.0}):
            return KDEAnalyzer(dataset, outcome_name)

    if num_continuous_factors > 0 and num_categorical_factors > 0:
        return MNPEAnalyzer(dataset, outcome_name)
    if num_categorical_factors > 0:
        return EmpiricalAnalyzer(dataset, outcome_name)
    return NPEAnalyzer(dataset, outcome_name)
