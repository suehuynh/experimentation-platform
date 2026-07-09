"""
Experimentation Platform: Sequential Testing via mSPRT.

This module implements the mixture Sequential Probability Ratio Test
(mSPRT), enabling continuous monitoring of A/B experiments without
inflating the false positive rate -- a guarantee that fixed-horizon
t-tests cannot provide.

The core idea: at each observation, compute the ratio of how likely
the accumulated data is under H1 (an effect exists) vs H0 (no effect).
This likelihood ratio is a martingale under H0, meaning its expected
value remains bounded regardless of when you stop -- which is what
makes inference valid at any stopping time.

Reject H0 when the mixture likelihood ratio exceeds 1 / alpha.

Key reference:
    Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021).
    Time-uniform, nonparametric, nonasymptotic confidence sequences.
    Annals of Statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# 1. Sequential result container
# ---------------------------------------------------------------------------

@dataclass
class SequentialResult:
    """Mutable container tracking the state of an ongoing sequential test.

    Unlike ReadoutResult (which is frozen and represents a final decision),
    SequentialResult is mutable because sequential tests accumulate
    observations over time and their state evolves with each update.

    Attributes:
        n_control: Current number of control group observations.
        n_treatment: Current number of treatment group observations.
        mixture_likelihood_ratio: Current value of the mSPRT statistic.
            Reject H0 when this exceeds 1 / alpha.
        always_valid_pvalue: Always-valid p-value = min(1, 1 / mixture_likelihood_ratio).
            Valid for inference at any stopping time without FPR inflation.
        rejected: True if the null hypothesis has been rejected at the
            current stopping time (mixture_likelihood_ratio >= 1 / alpha).
        alpha: Significance level used for the rejection threshold.
        effect_history: List of per-update observed mean differences,
            useful for plotting the trajectory of the experiment over time.
        mlr_history: List of per-update mixture likelihood ratio values,
            useful for visualising how evidence accumulates over time.
    """
    n_control: int = 0
    n_treatment: int = 0
    mixture_likelihood_ratio: float = 1.0
    always_valid_pvalue: float = 1.0
    rejected: bool = False
    alpha: float = 0.05
    effect_history: list[float] = field(default_factory=list)
    mlr_history: list[float] = field(default_factory=list)

    def summary(self) -> str:
        rejection_threshold = 1 / self.alpha
        status = (
            "REJECTED: sufficient evidence to stop the experiment."
            if self.rejected
            else "CONTINUING: insufficient evidence to reject H0."
        )
        return (
            f"Sequential Test Snapshot (mSPRT)\n"
            f"{'=' * 50}\n"
            f"Observations    : control n={self.n_control:,}, "
            f"treatment n={self.n_treatment:,}\n"
            f"{'─' * 50}\n"
            f"MLR             : {self.mixture_likelihood_ratio:.4f} "
            f"(threshold: {rejection_threshold:.2f})\n"
            f"Always-valid p  : {self.always_valid_pvalue:.4f} "
            f"(alpha: {self.alpha})\n"
            f"Updates logged  : {len(self.mlr_history):,}\n"
            f"{'─' * 50}\n"
            f"Status          : {status}\n"
        )


# ---------------------------------------------------------------------------
# 2. mSPRT sequential tester
# ---------------------------------------------------------------------------

class SequentialTester:
    """Implements mSPRT for continuous, always-valid A/B test monitoring.

    The mixture likelihood ratio (MLR) is updated incrementally as new
    observations arrive. The mixing distribution over effect sizes is
    chosen as a normal prior N(0, tau^2), which yields a closed-form
    update and is a standard choice in practice.

    The MLR update at time t given new (control, treatment) batches:

        MLR_t = MLR_{t-1} * likelihood_ratio_increment

    where the increment integrates the normal likelihood ratio over the
    prior on effect sizes, yielding a closed-form Gaussian mixture.

    Rejection rule: reject H0 when MLR_t >= 1 / alpha.
    Always-valid p-value: p_t = min(1, 1 / MLR_t).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        tau: float = 1.0,
    ):
        """Initialise the sequential tester.

        Args:
            alpha: Significance level. Experiment is stopped and H0
                rejected when MLR >= 1 / alpha. Defaults to 0.05.
            tau: Standard deviation of the normal mixing prior over
                effect sizes. Controls sensitivity to effect magnitude:
                smaller tau is more sensitive to small effects but slower
                to respond to large ones. Defaults to 1.0; tune to the
                minimum detectable effect you care about.
        """
        assert 0 < alpha < 1, f"alpha must be in (0, 1) exclusively {alpha}"
        assert tau > 0, f"tau must be positive {tau}"
        self._alpha = alpha
        self._tau = tau
        self._result = SequentialResult(alpha=alpha)

    @property
    def result(self) -> SequentialResult:
        """Current sequential test state, updated after each call to update().

        Returns:
            The live SequentialResult instance tracking all accumulated
            observations and the current MLR.
        """
        return self._result

    def reset(self) -> None:
        """Reset the tester to its initial state, discarding all accumulated data.

        Use between independent experiments to avoid carrying over state.
        """
        self._result = SequentialResult(alpha=self._alpha)

    def _compute_likelihood_ratio_increment(
        self,
        control_batch: np.ndarray,
        treatment_batch: np.ndarray,
    ) -> float:
        """Compute the mSPRT likelihood ratio increment for one batch of observations.

        Uses the closed-form Gaussian mixture likelihood ratio. Under the
        normal mixing prior N(0, tau^2) on the effect size delta, the
        integrated likelihood ratio for a batch has the form:

            LR = sqrt(sigma^2 / (sigma^2 + n * tau^2))
                 * exp(n^2 * tau^2 * d_bar^2 /
                        (2 * sigma^2 * (sigma^2 + n * tau^2)))

        where:
            d_bar = mean(treatment_batch) - mean(control_batch)
            sigma^2 = pooled variance of the batch
            n = harmonic mean of batch sizes (approximation for unequal n)

        Args:
            control_batch: New control observations since the last update.
            treatment_batch: New treatment observations since the last update.

        Returns:
            Scalar likelihood ratio increment >= 0. Values > 1 indicate
            the batch provided evidence in favour of H1; values < 1
            indicate evidence in favour of H0.
        """
        n_c = len(control_batch)
        n_t = len(treatment_batch)
        d_bar = np.mean(treatment_batch) - np.mean(control_batch)
        pooled_var = (
                (n_c - 1) * np.var(control_batch, ddof=1) +
                (n_t - 1) * np.var(treatment_batch, ddof=1)
            ) / (n_c + n_t - 2)
        if pooled_var == 0:
            return 1
        
        n_harm = 2 * n_c * n_t / (n_c + n_t)  # harmonic mean of sizes
        tau2 = self._tau ** 2
        denom = pooled_var + n_harm * tau2
        lr = np.sqrt(pooled_var / denom) * np.exp(
            n_harm**2 * tau2 * d_bar**2 / (2 * pooled_var * denom))
        
        return float(lr)

    def update(
        self,
        control_batch: np.ndarray,
        treatment_batch: np.ndarray,
    ) -> SequentialResult:
        """Incorporate a new batch of observations and update the MLR.

        Designed for incremental use: call update() each time a new batch
        of data arrives (e.g. daily) rather than passing the full history.
        Each call multiplies the current MLR by the new batch's likelihood
        ratio increment, accumulating evidence over time.

        Args:
            control_batch: New control observations since the last update.
                Must be non-empty and free of NaN/Inf.
            treatment_batch: New treatment observations since the last update.
                Must be non-empty and free of NaN/Inf.

        Returns:
            Updated SequentialResult reflecting the new accumulated state.
        """
        assert len(control_batch) > 0 and len(treatment_batch) > 0, f"Control/Treatment Batch is empty. Control Batch: {len(control_batch)}. Treatment Batch: {len(treatment_batch)}"
        assert np.all(np.isfinite(control_batch)) and np.all(np.isfinite(treatment_batch)), f"Control/Batch has infinite values."
        assert not self._result.rejected, "Experiment already rejected H0. Call reset() before reuse."

        self._result.n_control += len(control_batch)
        self._result.n_treatment += len(treatment_batch)

        lr_increment = self._compute_likelihood_ratio_increment(control_batch, treatment_batch)
        self._result.mixture_likelihood_ratio *= lr_increment

        self._result.always_valid_pvalue = min(
            1.0, 1.0 / self._result.mixture_likelihood_ratio)

        self._result.rejected = (self._result.mixture_likelihood_ratio >= 1 / self._alpha)

        observed_diff = np.mean(treatment_batch) - np.mean(control_batch)
        self._result.effect_history.append(observed_diff)
        self._result.mlr_history.append(self._result.mixture_likelihood_ratio)

        return self._result

    def run_full_experiment(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        batch_size: int = 50,
    ) -> SequentialResult:
        """Simulate a full sequential experiment by streaming data in batches.

        Splits pre-collected control and treatment arrays into sequential
        batches and calls update() on each, simulating how the MLR would
        have evolved had data arrived incrementally. Useful for retrospective
        analysis and for comparing sequential vs. fixed-horizon decisions
        on the same dataset.

        Args:
            control: Full control group observations.
            treatment: Full treatment group observations.
            batch_size: Number of new observations per group per update.
                Smaller batches simulate more frequent monitoring.

        Returns:
            Final SequentialResult after all batches have been processed.
        """
        self.reset()  # ensure clean state before full run
        n_batches = min(len(control), len(treatment)) // batch_size
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            self.update(control[start:end], treatment[start:end])
            if self._result.rejected:
                break  # early stopping: H0 rejected, no need to continue
        return self._result