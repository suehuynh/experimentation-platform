"""
Experimentation Platform: Guardrail Metric Monitoring.

Guardrail metrics are business-health indicators that must not degrade
during an experiment, even when the primary metric improves. They protect
against:

    - Metric gaming: a model optimising watch time at the expense of
      ad revenue or user satisfaction.
    - Unintended side effects: a UI change improving click-through rate
      while increasing crash rate or load time.
    - Cannibalisation: gains in one product surface coming at the cost
      of another.

Key design differences from primary metric readouts:

    1. ONE-SIDED TESTING: guardrails only care about degradation (a
       decrease in revenue is bad; an increase is fine). This means
       we use a one-sided p-value testing H0: delta >= 0 vs H1: delta < 0.

    2. LENIENT ALPHA: because missing a real degradation (false negative)
       is more costly than a false alarm (false positive), guardrail
       alpha is typically set higher than primary metric alpha
       (e.g. 0.10 or 0.20 vs 0.05), increasing sensitivity to harm.

    3. MULTIPLE GUARDRAILS: experiments commonly monitor 3-10 guardrail
       metrics simultaneously. This module applies Bonferroni correction
       across guardrail p-values (more conservative than BH, appropriate
       because guardrails are a small, pre-specified set where
       familywise error rate control is preferred over FDR control).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Guardrail status enum
# ---------------------------------------------------------------------------

class GuardrailStatus(Enum):
    """Decision status for a single guardrail metric evaluation.

    Attributes:
        PASSING: No significant degradation detected. Safe to proceed.
        FAILING: Statistically significant degradation detected at the
            Bonferroni-corrected alpha level. Experiment should be
            paused or stopped pending investigation.
        INCONCLUSIVE: Degradation trend observed but not yet significant.
            Warrants continued monitoring before a ship decision.
    """
    PASSING = "PASSING"
    FAILING = "FAILING"
    INCONCLUSIVE = "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# 2. Per-guardrail result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GuardrailResult:
    """Statistical readout for a single guardrail metric.

    Attributes:
        metric_name: Human-readable label for the guardrail metric
            (e.g. 'ad_revenue_per_session', 'crash_rate', 'p99_latency').
        n_control: Number of control observations for this metric.
        n_treatment: Number of treatment observations for this metric.
        control_mean: Sample mean of the control group.
        treatment_mean: Sample mean of the treatment group.
        absolute_difference: treatment_mean - control_mean. Negative
            values indicate degradation for metrics where higher is better.
        one_sided_pvalue: p-value from the one-sided t-test testing
            H0: delta >= 0 vs H1: delta < 0 (treatment is worse).
        bonferroni_pvalue: Bonferroni-corrected p-value
            (one_sided_pvalue * n_guardrails). Capped at 1.0.
        corrected_alpha: The Bonferroni-adjusted alpha threshold
            (raw_alpha / n_guardrails) used for the FAILING decision.
        status: GuardrailStatus enum value summarising the decision.
        direction: 'higher_is_better' or 'lower_is_better', controls
            which tail of the distribution constitutes degradation.
    """
    metric_name: str
    n_control: int
    n_treatment: int
    control_mean: float
    treatment_mean: float
    absolute_difference: float
    one_sided_pvalue: float
    bonferroni_pvalue: float
    corrected_alpha: float
    status: GuardrailStatus
    direction: str


# ---------------------------------------------------------------------------
# 3. Overall guardrail suite result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GuardrailSuiteResult:
    """Immutable container summarising all guardrail evaluations for an experiment.

    Attributes:
        guardrail_results: List of GuardrailResult instances, one per
            monitored metric.
        any_failing: True if at least one guardrail is in FAILING status.
            The experiment should not ship if this is True.
        n_failing: Count of guardrails in FAILING status.
        n_inconclusive: Count of guardrails in INCONCLUSIVE status.
        raw_alpha: The unadjusted significance level before Bonferroni
            correction (e.g. 0.10).
        n_guardrails: Total number of guardrail metrics monitored.
    """
    guardrail_results: list[GuardrailResult]
    any_failing: bool
    n_failing: int
    n_inconclusive: int
    raw_alpha: float
    n_guardrails: int

    def failing_guardrails(self) -> list[GuardrailResult]:
        """Return only the guardrails currently in FAILING status.

        Returns:
            List of GuardrailResult where status is GuardrailStatus.FAILING,
            sorted by bonferroni_pvalue ascending (most significant first).
        """
        failing_results = [guardrail_result for guardrail_result in self.guardrail_results if guardrail_result.status == GuardrailStatus.FAILING]
        failing_metrics = sorted(failing_results, key=lambda x: x.bonferroni_pvalue)
        return failing_metrics

    def summary(self) -> str:
        """Return a human-readable guardrail monitoring report.

        Returns:
            Multi-line string covering all guardrail statuses, corrected
            alpha, and a ship/no-ship recommendation.
        """
        corrected_alpha = self.raw_alpha / self.n_guardrails
        failing_names = [r.metric_name for r in self.guardrail_results
                        if r.status == GuardrailStatus.FAILING]
        if self.any_failing:
            verdict = (f"DO NOT SHIP: {self.n_failing} guardrail(s) failing "
                    f"— {', '.join(failing_names)}.")
        elif self.n_inconclusive > 0:
            verdict = f"MONITOR: no failures but {self.n_inconclusive} guardrail(s) inconclusive."
        else:
            verdict = "SHIP: all guardrails passing."

        header = (
            f"Guardrail Monitoring Report\n"
            f"{'=' * 65}\n"
            f"Guardrails      : {self.n_guardrails}  |  "
            f"Raw alpha: {self.raw_alpha}  |  "
            f"Bonferroni alpha: {corrected_alpha:.4f}\n"
            f"{'─' * 65}\n"
            f"{'Metric':<25} {'Ctrl':>8} {'Trt':>8} {'Diff':>8} "
            f"{'BF p-val':>10} {'Status':>13}\n"
            f"{'─' * 65}\n"
        )
        rows = ""
        for r in self.guardrail_results:
            rows += (
                f"{r.metric_name:<25} {r.control_mean:>8.4f} "
                f"{r.treatment_mean:>8.4f} {r.absolute_difference:>+8.4f} "
                f"{r.bonferroni_pvalue:>10.4f} {r.status.value:>13}\n"
            )
        return header + rows + f"{'─' * 65}\n" + f"Verdict         : {verdict}\n"


# ---------------------------------------------------------------------------
# 4. Guardrail monitor
# ---------------------------------------------------------------------------

class GuardrailMonitor:
    """Evaluates a suite of guardrail metrics for a single experiment.

    Applies Bonferroni correction across all guardrail p-values and
    classifies each metric as PASSING, FAILING, or INCONCLUSIVE based
    on its corrected significance and the direction of any observed effect.

    Bonferroni is preferred over BH here because:
        - The number of guardrails is small (typically 3-10).
        - We want to control the familywise error rate (probability of
          ANY false alarm), not just the false discovery rate.
        - A single false FAILING verdict could unjustly block a good ship.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        inconclusive_threshold: float = 0.20,
    ):
        """Initialise the guardrail monitor.

        Args:
            alpha: Raw significance level before Bonferroni correction.
                Defaults to 0.10 (more lenient than primary metric alpha
                of 0.05, to increase sensitivity to degradation).
            inconclusive_threshold: p-value threshold below which a
                guardrail is flagged as INCONCLUSIVE rather than PASSING,
                even if it does not meet the FAILING threshold. Signals
                a trend worth watching. Defaults to 0.20.
        """
        assert 0 < alpha < 1, f"alpha must be in range (0,1) {alpha}"
        assert inconclusive_threshold > alpha, f"inclusive threshold must be larger than alpha. alpha = {alpha} > threshold {inconclusive_threshold}"
        self._alpha = alpha
        self._inconclusive_threshold = inconclusive_threshold

    def _one_sided_pvalue(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        direction: str,
    ) -> tuple[float, float]:
        """Compute a one-sided Welch t-test p-value for degradation.

        Args:
            control: Control group metric observations.
            treatment: Treatment group metric observations.
            direction: 'higher_is_better' — degradation means treatment
                mean is significantly lower than control mean.
                'lower_is_better' — degradation means treatment mean is
                significantly higher than control mean (e.g. crash rate,
                latency).

        Returns:
            Tuple of (absolute_difference, one_sided_pvalue) where
            one_sided_pvalue tests specifically for degradation in the
            given direction.
        """
        absolute_difference = np.mean(treatment) - np.mean(control)
        t_stat, two_sided_pvalue = stats.ttest_ind(treatment, control, equal_var=False)
        if direction == "higher_is_better":
            if t_stat < 0:
                one_sided_pvalue = two_sided_pvalue / 2
            else:
                one_sided_pvalue = 1 - two_sided_pvalue / 2
        elif direction == "lower_is_better":
            if t_stat < 0:
                one_sided_pvalue = 1 - two_sided_pvalue / 2
            else:
                one_sided_pvalue = two_sided_pvalue / 2
        else:
            raise ValueError(f"Direction must be 'higher_is_better' or 'lower_is_better', got: {direction}")
        
        return (absolute_difference, one_sided_pvalue)

    def _classify(
        self,
        one_sided_pvalue: float,
        bonferroni_pvalue: float,
        corrected_alpha: float,
        absolute_difference: float,
        direction: str,
    ) -> GuardrailStatus:
        """Classify a guardrail as PASSING, FAILING, or INCONCLUSIVE.

        Args:
            one_sided_pvalue: Raw one-sided p-value for degradation.
            bonferroni_pvalue: Bonferroni-corrected p-value.
            corrected_alpha: Bonferroni-adjusted alpha threshold.
            absolute_difference: treatment_mean - control_mean.
            direction: 'higher_is_better' or 'lower_is_better'.

        Returns:
            GuardrailStatus enum value.
        """
        # absolute difference is not in degradation direction
        if direction == 'higher_is_better' and absolute_difference >= 0:
            return GuardrailStatus.PASSING
        if direction == 'lower_is_better' and absolute_difference <= 0:
            return GuardrailStatus.PASSING
        
        # absolute difference is in degradation direction
        if direction == 'higher_is_better' and absolute_difference < 0:
            if bonferroni_pvalue < corrected_alpha:
                return GuardrailStatus.FAILING
            if one_sided_pvalue < self._inconclusive_threshold:
                return GuardrailStatus.INCONCLUSIVE
            else:
                return GuardrailStatus.PASSING
        if direction == 'lower_is_better' and absolute_difference > 0:
            if bonferroni_pvalue < corrected_alpha:
                return GuardrailStatus.FAILING
            if one_sided_pvalue < self._inconclusive_threshold:
                return GuardrailStatus.INCONCLUSIVE
            else:
                return GuardrailStatus.PASSING

    def compute(
        self,
        metrics: dict[str, tuple[np.ndarray, np.ndarray, str]],
    ) -> GuardrailSuiteResult:
        """Evaluate all guardrail metrics and return a suite-level result.

        Args:
            metrics: Dict mapping metric name to a tuple of
                (control_observations, treatment_observations, direction)
                where direction is 'higher_is_better' or 'lower_is_better'.
                Example:
                    {
                        "ad_revenue": (ctrl_rev, trt_rev, "higher_is_better"),
                        "crash_rate": (ctrl_crash, trt_crash, "lower_is_better"),
                        "p99_latency": (ctrl_lat, trt_lat, "lower_is_better"),
                    }

        Returns:
            GuardrailSuiteResult with per-metric results and an overall
            ship/no-ship flag.
        """
        n_guardrails = len(metrics)
        corrected_alpha = self._alpha / n_guardrails
        guardrail_results = []
        for metric, (control, treatment, direction) in metrics.items():
            absolute_difference, one_sided_pvalue = self._one_sided_pvalue(control, treatment, direction)
            bonferroni_pvalue = min(one_sided_pvalue * n_guardrails, 1.0)
            status = self._classify(one_sided_pvalue, bonferroni_pvalue, corrected_alpha, absolute_difference, direction)
            guardrail_result = GuardrailResult(metric_name=metric,
                                     n_control=len(control),
                                     n_treatment=len(treatment),
                                     control_mean=np.mean(control),
                                     treatment_mean=np.mean(treatment),
                                     absolute_difference=absolute_difference,
                                     one_sided_pvalue=one_sided_pvalue,
                                     bonferroni_pvalue=bonferroni_pvalue,
                                     corrected_alpha=corrected_alpha,
                                     direction=direction,
                                     status=status)
            guardrail_results.append(guardrail_result)
        
        n_failing = 0
        n_inconclusive = 0
        any_failing = False
        for guardrail_result in guardrail_results:
            if guardrail_result.status == GuardrailStatus.FAILING:
                any_failing = True
                n_failing += 1
            if guardrail_result.status == GuardrailStatus.INCONCLUSIVE:
                n_inconclusive += 1
        
        return GuardrailSuiteResult(guardrail_results=guardrail_results,
                                    any_failing=any_failing,
                                    n_failing=n_failing,
                                    n_inconclusive=n_inconclusive,
                                    raw_alpha=self._alpha,
                                    n_guardrails=n_guardrails)