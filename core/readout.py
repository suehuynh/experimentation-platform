"""
Experimentation Platform: Statistical Readout Engine.

This module implements the statistical decision layer of the A/B testing
pipeline. It consumes raw metric observations from control and treatment
groups and produces calibrated significance decisions, point estimates,
and confidence intervals.

Two complementary readout strategies are provided:
    - TTestReadout: parametric, assumes approximate normality (valid for
      large samples via CLT). Fast and standard for most continuous metrics.
    - BootstrapReadout: non-parametric resampling approach. Correct for
      skewed or heavy-tailed metrics (e.g. revenue per user) where the
      normality assumption breaks down.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReadoutResult:
    """Immutable container for the statistical output of a single experiment readout.

    Attributes:
        control_mean: Sample mean of the control group metric.
        treatment_mean: Sample mean of the treatment group metric.
        absolute_difference: Point estimate of the treatment effect
            (treatment_mean - control_mean).
        relative_difference: Lift expressed as a fraction of the control
            mean (absolute_difference / control_mean). None if
            control_mean is zero.
        p_value: Probability of observing a difference this large or
            larger under the null hypothesis of no effect.
        ci_lower: Lower bound of the confidence interval for the
            absolute difference.
        ci_upper: Upper bound of the confidence interval for the
            absolute difference.
        significant: True if p_value < alpha AND the confidence interval
            excludes zero. Convenience flag for downstream consumers.
        alpha: The significance threshold used to compute `significant`.
        n_control: Number of observations in the control group.
        n_treatment: Number of observations in the treatment group.
        method: Human-readable label identifying the readout strategy
            used (e.g. 't-test', 'bootstrap').
    """
    control_mean: float
    treatment_mean: float
    absolute_difference: float
    relative_difference: float | None
    p_value: float
    ci_lower: float
    ci_upper: float
    significant: bool
    alpha: float
    n_control: int
    n_treatment: int
    method: str

    def summary(self) -> str:
        """Return a human-readable one-paragraph readout suitable for a
        growth team memo or experiment console display.
        """
        rel_diff_str = (
            f"{self.relative_difference:+.2%}"
            if self.relative_difference is not None
            else "N/A (control mean is zero)"
        )
        conclusion = (
            f"Statistically significant at alpha={self.alpha}. "
            "Recommend further review before launch decision."
            if self.significant
            else f"Not significant at alpha={self.alpha}. "
            "Insufficient evidence to reject the null hypothesis."
        )
        return (
            f"Experiment Readout ({self.method})\n"
            f"{'=' * 50}\n"
            f"Sample sizes    : control n={self.n_control:,}, "
            f"treatment n={self.n_treatment:,}\n"
            f"Control mean    : {self.control_mean:.4f}\n"
            f"Treatment mean  : {self.treatment_mean:.4f}\n"
            f"Absolute lift   : {self.absolute_difference:+.4f}\n"
            f"Relative lift   : {rel_diff_str}\n"
            f"{'─' * 50}\n"
            f"p-value         : {self.p_value:.4f} (alpha={self.alpha})\n"
            f"95% CI          : [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]\n"
            f"{'─' * 50}\n"
            f"Conclusion      : {conclusion}\n"
        )

# ---------------------------------------------------------------------------
# 2. Abstract base class
# ---------------------------------------------------------------------------

class ReadoutStrategy(ABC):
    """Abstract base class defining the interface for all readout strategies.

    Concrete subclasses implement `compute()` with strategy-specific
    statistical machinery. All strategies share the same input/output
    contract so they are interchangeable in the experimentation pipeline.
    """

    def __init__(self, alpha: float = 0.05):
        """Initialize the readout strategy with a significance threshold.

        Args:
            alpha: Type I error rate / significance level. Experiment is
                declared significant when p_value < alpha. Defaults to
                0.05 (industry standard for most A/B tests).
        """
        assert 0 < alpha < 1, f"alpha must be in (0, 1) exclusively {alpha}"
        self._alpha = alpha

    @abstractmethod
    def compute(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
    ) -> ReadoutResult:
        """Compute statistical significance and effect size estimates.

        Args:
            control: 1-D array of metric observations for the control group.
            treatment: 1-D array of metric observations for the treatment group.

        Returns:
            A fully populated ReadoutResult instance.
        """
        ...

    def _validate_inputs(self, control: np.ndarray, treatment: np.ndarray) -> None:
        """Guard against degenerate input arrays before any computation.

        Args:
            control: Control group metric observations.
            treatment: Treatment group metric observations.

        Raises:
            ValueError: If either array is empty or contains non-finite values
                (NaN or Inf), which would silently corrupt downstream statistics.
        """
        if len(control) == 0 or len(treatment) == 0:
            raise ValueError("Control and treatment arrays must be non-empty.")
        
        if np.any(~np.isfinite(control)) or np.any(~np.isfinite(treatment)):
            raise ValueError("Input arrays must not contain NaN or Inf values.")

    def _build_result(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        p_value: float,
        ci_lower: float,
        ci_upper: float,
        method: str,
    ) -> ReadoutResult:
        """Assemble a ReadoutResult from raw statistical outputs.

        Centralises mean computation, relative lift, and the significance
        flag so concrete subclasses only need to supply the three values
        that differ between strategies: p_value, ci_lower, ci_upper.

        Args:
            control: Control group metric observations.
            treatment: Treatment group metric observations.
            p_value: Strategy-computed p-value.
            ci_lower: Lower confidence interval bound for the absolute difference.
            ci_upper: Upper confidence interval bound for the absolute difference.
            method: Label identifying the calling strategy (e.g. 't-test').

        Returns:
            Fully populated ReadoutResult.
        """
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        absolute_difference = treatment_mean - control_mean
        relative_difference = absolute_difference / control_mean if control_mean != 0 else None
        significant = (p_value < self._alpha) and (ci_lower > 0 or ci_upper < 0)
        return ReadoutResult(control_mean, treatment_mean,
                            absolute_difference, relative_difference,
                            p_value, ci_lower, ci_upper,
                            significant, alpha=self._alpha, 
                            n_control=len(control), n_treatment=len(treatment), 
                            method=method)

# ---------------------------------------------------------------------------
# 3. T-Test readout (parametric)
# ---------------------------------------------------------------------------

class TTestReadout(ReadoutStrategy):
    """Parametric readout using Welch's two-sample t-test.

    Welch's t-test (the default in scipy.stats.ttest_ind) is preferred
    over Student's t-test because it does not assume equal variances
    between groups -- a realistic assumption in production experiments
    where group sizes and variance can differ due to assignment noise or
    metric skew.

    Valid for large samples regardless of underlying metric distribution
    (Central Limit Theorem). For small samples or heavily skewed metrics
    (e.g. revenue per user), prefer BootstrapReadout.
    """

    def compute(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
    ) -> ReadoutResult:
        """Run Welch's t-test and return a fully populated ReadoutResult.

        Args:
            control: 1-D array of metric observations for the control group.
            treatment: 1-D array of metric observations for the treatment group.

        Returns:
            ReadoutResult populated with t-test p-value and analytical
            confidence interval derived from the t-distribution.
        """
        
        # Calculate Welch's t-test statistics
        self._validate_inputs(control, treatment)
        t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
        
        # Calculate Degree of Freedon
        se_control = np.var(control, ddof=1) / len(control)
        se_treatment = np.var(treatment, ddof=1) / len(treatment)
        se_diff = np.sqrt(se_control + se_treatment)
        df = (se_control + se_treatment)**2 / (
                    se_control**2 / (len(control) - 1) +
                    se_treatment**2 / (len(treatment) - 1))
        
        # Calculate CI
        t_crit = stats.t.ppf(1 - self._alpha / 2, df=df)
        diff = np.mean(treatment) - np.mean(control)
        ci_lower = diff - t_crit * se_diff
        ci_upper = diff + t_crit * se_diff
        return self._build_result(control, treatment, p_value,
                                  ci_lower, ci_upper, method="t-test")


# ---------------------------------------------------------------------------
# 4. Bootstrap readout (non-parametric)
# ---------------------------------------------------------------------------

class BootstrapReadout(ReadoutStrategy):
    """Non-parametric readout using percentile bootstrap resampling.

    Constructs a confidence interval by repeatedly resampling the
    observed data with replacement and computing the difference in means
    across each resample. The empirical distribution of these resampled
    differences is used to derive confidence bounds without assuming
    normality.

    Preferred over TTestReadout when:
        - Metric distributions are heavily skewed (e.g. revenue, LTV)
        - Sample sizes are small (CLT hasn't kicked in)
        - You want assumption-free inference

    Slower than TTestReadout (O(n_bootstraps * n) vs O(n)).
    """

    def __init__(self, alpha: float = 0.05, n_bootstraps: int = 10_000):
        """Initialize the bootstrap readout strategy.

        Args:
            alpha: Significance level. Defaults to 0.05.
            n_bootstraps: Number of bootstrap resamples. Higher values
                produce more stable CI estimates at the cost of compute.
                10,000 is the industry standard default.
        """
        super().__init__(alpha)
        assert n_bootstraps > 0, "number of bootstraps has to be larger than 0"
        self._n_bootstraps  = n_bootstraps

    def compute(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
    ) -> ReadoutResult:
        """Run percentile bootstrap and return a fully populated ReadoutResult.

        Args:
            control: 1-D array of metric observations for the control group.
            treatment: 1-D array of metric observations for the treatment group.

        Returns:
            ReadoutResult populated with bootstrap p-value and empirical
            confidence interval derived from the resampled distribution.
        """
        self._validate_inputs(control, treatment)
        observed_diff = np.mean(treatment) - np.mean(control)
        boot_diffs = []
        for i in range(self._n_bootstraps):
            resample_control = np.random.choice(control, size=len(control), replace=True)
            resample_treatment = np.random.choice(treatment, size=len(treatment), replace=True)
            boot_diffs.append(np.mean(resample_treatment) - np.mean(resample_control))
        
        boot_diffs = np.array(boot_diffs)
        # CI via percentile method:
        ci_lower = np.percentile(boot_diffs, 100 * self._alpha / 2)
        ci_upper = np.percentile(boot_diffs, 100 * (1 - self._alpha / 2))

        # p_value via two-sided test against null of zero
        boot_diffs = boot_diffs - observed_diff
        p_value = np.mean(np.abs(boot_diffs)>= abs(observed_diff))
        return self._build_result(control, treatment, p_value,
            ci_lower, ci_upper, method="bootstrap")