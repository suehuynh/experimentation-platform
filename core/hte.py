"""
Experimentation Platform: Heterogeneous Treatment Effect Detection.

This module identifies subgroups of users for whom the treatment effect
differs meaningfully from the average treatment effect (ATE). Detecting
heterogeneity is critical for:

    - Avoiding harm: a positive ATE can mask negative effects on
      high-value subgroups (e.g. power users, mobile users).
    - Personalisation: understanding which segments benefit most
      enables targeted rollout rather than all-or-nothing decisions.
    - Guardrail enforcement: some segments may be contractually or
      ethically protected from degradation.

Multiple testing is a first-class concern here: testing K subgroups
independently at alpha inflates the familywise false positive rate.
This module applies the Benjamini-Hochberg (BH) procedure to control
the false discovery rate (FDR) across all subgroup comparisons.

Key reference:
    Benjamini, Y., & Hochberg, Y. (1995).
    Controlling the false discovery rate: a practical and powerful
    approach to multiple testing. JRSS-B.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.readout import ReadoutStrategy, ReadoutResult, TTestReadout


# ---------------------------------------------------------------------------
# 1. Per-subgroup result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubgroupResult:
    """Statistical readout for a single subgroup comparison.

    Attributes:
        subgroup_name: Human-readable label for the subgroup
            (e.g. 'device=mobile', 'tenure=new_user').
        n_control: Number of control observations in this subgroup.
        n_treatment: Number of treatment observations in this subgroup.
        readout: Full ReadoutResult for this subgroup's comparison,
            including p-value, CI, and means.
        raw_pvalue: Unadjusted p-value from the subgroup readout,
            stored separately for BH correction input.
        bh_adjusted_pvalue: Benjamini-Hochberg adjusted p-value.
            Populated by HTEDetector after all subgroups are tested.
            None until BH correction has been applied.
        significant_after_correction: True if bh_adjusted_pvalue < alpha.
            None until BH correction has been applied.
    """
    subgroup_name: str
    n_control: int
    n_treatment: int
    readout: ReadoutResult
    raw_pvalue: float
    bh_adjusted_pvalue: float | None = None
    significant_after_correction: bool | None = None


# ---------------------------------------------------------------------------
# 2. HTE result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HTEResult:
    """Immutable container summarising HTE detection across all subgroups.

    Attributes:
        overall_result: ReadoutResult for the full experiment population
            (the average treatment effect, ATE), used as the baseline
            for comparing subgroup effects.
        subgroup_results: List of SubgroupResult instances, one per
            subgroup tested, with BH-adjusted p-values populated.
        heterogeneity_detected: True if at least one subgroup shows a
            statistically significant effect after BH correction AND
            that effect differs in direction or magnitude from the ATE.
        alpha: Significance level used for both the overall test and
            the BH-corrected subgroup tests.
        fdr_level: The FDR level used for BH correction. Typically set
            equal to alpha, but may be relaxed (e.g. 0.10) for
            exploratory subgroup analysis.
    """
    overall_result: ReadoutResult
    subgroup_results: list[SubgroupResult]
    heterogeneity_detected: bool
    alpha: float
    fdr_level: float

    def significant_subgroups(self) -> list[SubgroupResult]:
        """Return only the subgroups that survived BH correction.

        Returns:
            List of SubgroupResult where significant_after_correction
            is True, sorted by absolute effect size descending so the
            most impactful subgroups surface first.
        """
        corrected_subgroup_results = [subgroup_result for subgroup_result in self.subgroup_results if subgroup_result.significant_after_correction]
        corrected_subgroup_results = sorted(corrected_subgroup_results, key=lambda x: abs(x.readout.absolute_difference), reverse=True)
        return corrected_subgroup_results

    def summary(self) -> str:
        sig = self.significant_subgroups()
        n_tested = len(self.subgroup_results)
        n_significant = len(sig)
        conclusion = (
            "Heterogeneity detected: review subgroup effects before shipping."
            if self.heterogeneity_detected
            else f"No significant heterogeneity detected at FDR level {self.fdr_level}."
        )
        header = (
            f"HTE Detection Report\n"
            f"{'=' * 60}\n"
            f"Overall ATE     : {self.overall_result.absolute_difference:+.4f} "
            f"(p={self.overall_result.p_value:.4f})\n"
            f"Subgroups tested: {n_tested}  |  "
            f"Significant after BH correction: {n_significant}\n"
            f"{'─' * 60}\n"
        )
        if sig:
            rows = f"{'Subgroup':<25} {'n_ctrl':>7} {'n_trt':>7} "
            rows += f"{'Effect':>9} {'BH p-value':>12}\n"
            rows += f"{'─' * 60}\n"
            for s in sig:
                rows += (
                    f"{s.subgroup_name:<25} {s.n_control:>7} {s.n_treatment:>7} "
                    f"{s.readout.absolute_difference:>+9.4f} "
                    f"{s.bh_adjusted_pvalue:>12.4f}\n"
                )
        else:
            rows = "No subgroups survived BH correction.\n"
        return header + rows + f"{'─' * 60}\n" + f"Conclusion      : {conclusion}\n"


# ---------------------------------------------------------------------------
# 3. BH correction utility
# ---------------------------------------------------------------------------

def benjamini_hochberg(
    pvalues: list[float],
    fdr_level: float,
) -> list[float]:
    """Apply the Benjamini-Hochberg procedure to control the false discovery rate.

    Ranks p-values from smallest to largest and compares each against a
    linearly increasing threshold (k/m) * fdr_level, where k is the rank
    and m is the total number of tests. The largest k for which the
    p-value is below its threshold determines the rejection set: all
    hypotheses at ranks 1..k are rejected together.

    Args:
        pvalues: List of raw (unadjusted) p-values, one per hypothesis.
        fdr_level: Target false discovery rate (e.g. 0.05). Controls the
            expected proportion of rejected hypotheses that are false positives.

    Returns:
        List of BH-adjusted p-values in the same order as the input.
        Adjusted p-values are computed as:
            p_adjusted[i] = min(p_raw[i] * m / rank[i], 1.0)
        where rank is 1-indexed position in the sorted order.
        Enforces monotonicity by taking a cumulative minimum from
        largest to smallest rank after adjustment.
    """
    m = len(pvalues)
    paired_values = [[org_idx, pvalue] for org_idx, pvalue in enumerate(pvalues)]
    p_raw = sorted(paired_values, key=lambda x: x[1])
    adjusted_p = [[org_idx, min(pvalue * m / (k + 1), 1.0)] for k, (org_idx, pvalue) in enumerate(p_raw)]
    curr_min = adjusted_p[-1][1]
    for i in range(len(adjusted_p) - 2, -1, -1):
        if adjusted_p[i][1] > curr_min:
            adjusted_p[i][1] = curr_min
        curr_min = adjusted_p[i][1]
    
    adjust_pvalues = [None] * m
    for org_idx, adjusted_pvalue in adjusted_p:
        adjust_pvalues[org_idx] = adjusted_pvalue

    return adjust_pvalues
# ---------------------------------------------------------------------------
# 4. HTE detector
# ---------------------------------------------------------------------------

class HTEDetector:
    """Tests for heterogeneous treatment effects across predefined subgroups.

    Runs a separate readout for each subgroup, then applies Benjamini-
    Hochberg FDR correction across all subgroup p-values to control the
    false discovery rate. Flags heterogeneity when at least one subgroup
    survives correction and diverges from the overall ATE direction.

    Subgroups must be pre-specified before analysis begins. Post-hoc
    subgroup mining -- defining subgroups after seeing the data -- is not
    supported by design, as it invalidates the FDR guarantee.
    """

    def __init__(
        self,
        readout: ReadoutStrategy | None = None,
        fdr_level: float = 0.05,
    ):
        """Initialise the HTE detector.

        Args:
            readout: ReadoutStrategy used for each subgroup comparison.
                Defaults to TTestReadout(alpha=0.05) if not provided.
            fdr_level: Target false discovery rate for BH correction.
                Defaults to 0.05. May be relaxed to 0.10 for exploratory
                analyses where missing a real subgroup effect is costly.
        """
        if readout is None:
            self._readout = TTestReadout()
        else:
            self._readout = readout
        
        assert 0 < fdr_level < 1, f"fdr_level has to be between 0 and 1 {fdr_level}"
        self._fdr_level = fdr_level

    def _run_subgroup(
        self,
        subgroup_name: str,
        control_mask: np.ndarray,
        treatment_mask: np.ndarray,
        control_metric: np.ndarray,
        treatment_metric: np.ndarray,
    ) -> SubgroupResult:
        """Run the readout strategy on a single subgroup's observations.

        Args:
            subgroup_name: Human-readable label for this subgroup.
            control_mask: Boolean array of length n_control; True for
                observations belonging to this subgroup.
            treatment_mask: Boolean array of length n_treatment; True for
                observations belonging to this subgroup.
            control_metric: Full control group metric array (unfiltered).
            treatment_metric: Full treatment group metric array (unfiltered).

        Returns:
            SubgroupResult with raw_pvalue populated and BH fields as None
            (BH correction is applied later across all subgroups together).

        Raises:
            ValueError: If the subgroup is empty in either arm, which
                would make the readout degenerate.
        """
        control_group = control_metric[control_mask]
        treatment_group = treatment_metric[treatment_mask]
        assert len(control_group) > 0, f"Filtered control groups in {subgroup_name} is zero-length."
        assert len(treatment_group) > 0, f"Filtered treatment groups in {subgroup_name} is zero-length."

        results = self._readout.compute(control_group, treatment_group)
        return SubgroupResult(subgroup_name=subgroup_name,
                              n_control=len(control_group),
                              n_treatment=len(treatment_group),
                              readout=results,
                              raw_pvalue=results.p_value,
                              bh_adjusted_pvalue=None,
                              significant_after_correction=None)
    
    def compute(
        self,
        control_metric: np.ndarray,
        treatment_metric: np.ndarray,
        subgroups: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> HTEResult:
        """Detect heterogeneous treatment effects across predefined subgroups.

        Runs a readout for the full population (ATE) and for each
        subgroup, applies BH correction across all subgroup p-values,
        then flags heterogeneity if any subgroup survives correction.

        Args:
            control_metric: Full control group metric observations.
            treatment_metric: Full treatment group metric observations.
            subgroups: Dict mapping subgroup name to a tuple of
                (control_mask, treatment_mask), where each mask is a
                boolean array selecting the relevant observations.
                Example:
                    {
                        "device=mobile": (ctrl_mobile_mask, trt_mobile_mask),
                        "tenure=new":    (ctrl_new_mask,    trt_new_mask),
                    }

        Returns:
            HTEResult with BH-corrected subgroup results and a
            heterogeneity flag.
        """
        #   1. Compute the overall ATE readout on the full arrays.
        overall_result = self._readout.compute(control_metric, treatment_metric)
        ate = overall_result.absolute_difference
        #   2. Run _run_subgroup() for each entry in subgroups, collecting 
        # a list of SubgroupResult objects (with BH fields still None).
        subgroup_results = []
        raw_pvalues = []
        for subgroup, (control_mask, treatment_mask) in subgroups.items():
            subgroup_result = self._run_subgroup(subgroup_name=subgroup, 
                               control_mask=control_mask,
                               treatment_mask=treatment_mask,
                               control_metric=control_metric,
                               treatment_metric=treatment_metric)
            subgroup_results.append(subgroup_result)
            raw_pvalues.append(subgroup_result.raw_pvalue)
        
        #   3. Extract the raw p-values from all SubgroupResults into a list,
        #      then pass that list to benjamini_hochberg() to get adjusted
        #      p-values back in the same order.
        adjusted_pvalues = benjamini_hochberg(pvalues=raw_pvalues, fdr_level=self._fdr_level)
        #   4. Rebuild each SubgroupResult as a new frozen instance with
        #      bh_adjusted_pvalue and significant_after_correction populated.
        #      (Since SubgroupResult is frozen you cannot mutate it in place
        #      -- construct a new one using dataclasses.replace() or by
        #      passing all fields to SubgroupResult() again.)
        adjusted_subgroup_results = []
        for i, (subgroup, (control_mask, treatment_mask)) in enumerate(subgroups.items()):
            significant_after_correction = adjusted_pvalues[i] < self._fdr_level
            adjusted_subgroup_result = SubgroupResult(subgroup_name=subgroup,
                                n_control=np.sum(control_mask),
                                n_treatment=np.sum(treatment_mask),
                                readout=subgroup_results[i].readout,
                                raw_pvalue=subgroup_results[i].raw_pvalue,
                                bh_adjusted_pvalue=adjusted_pvalues[i],
                                significant_after_correction=significant_after_correction)
            adjusted_subgroup_results.append(adjusted_subgroup_result)
        #   5. Determine heterogeneity_detected: True if at least one
        #      subgroup is significant after correction AND its
        #      absolute_difference has a different sign from the overall
        #      ATE, OR its magnitude differs from the ATE by more than
        #      a meaningful threshold (use 50% of the ATE as a simple
        #      heuristic -- i.e. subgroup effect is less than half or
        #      more than 1.5x the ATE in absolute terms).
        hte_result = False
        for adjusted_subgroup_result in adjusted_subgroup_results:
            subgroup_effect = adjusted_subgroup_result.readout.absolute_difference
            different_sign = (ate * subgroup_effect) < 0
            different_magnitude = (abs(subgroup_effect) < 0.5 * abs(ate) or
                                   abs(subgroup_effect) > 1.5 * abs(ate))
            if (adjusted_subgroup_result.significant_after_correction and
                    (different_sign or different_magnitude)):
                hte_result = True
                break

        #   6. Return HTEResult with all fields populated.
        return HTEResult(
            overall_result=overall_result,
            subgroup_results=adjusted_subgroup_results,
            heterogeneity_detected=hte_result,
            alpha=self._readout._alpha,
            fdr_level=self._fdr_level,
        )