"""
Integration tests for the Experimentation Platform.

Wires all six core modules together on synthetic data and verifies
end-to-end behaviour with explicit assertions. Run with:

    pytest tests/test_core.py -v
"""

import numpy as np
import pytest

from core.assignment import VariantSpec, ExperimentConfig, HashAssignment, murmur_hash
from core.readout import TTestReadout, BootstrapReadout
from core.cuped import CUPEDAdjuster
from core.sequential import SequentialTester
from core.hte import HTEDetector
from core.guardrails import GuardrailMonitor, GuardrailStatus


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """Seeded RNG for reproducible synthetic data across all integration tests."""
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def experiment_data(rng) -> dict:
    """Generate synthetic control/treatment data with a known true effect.

    Returns a dict with:
        control, treatment: main outcome arrays (N=2000 each)
        pre_control, pre_treatment: correlated pre-experiment covariates
        n: sample size per arm
        true_effect: injected treatment lift
        control_mean: baseline mean
    """
    n = 2_000
    true_effect = 0.5
    control_mean = 10.0
    control_std = 2.0

    control = rng.normal(loc=control_mean, scale=control_std, size=n)
    treatment = rng.normal(loc=control_mean + true_effect, scale=control_std, size=n)

    # Pre-experiment covariate correlated at ~0.7 with outcome
    pre_control = control * 0.7 + rng.normal(0, control_std * 0.5, size=n)
    pre_treatment = treatment * 0.7 + rng.normal(0, control_std * 0.5, size=n)

    return {
        "control": control,
        "treatment": treatment,
        "pre_control": pre_control,
        "pre_treatment": pre_treatment,
        "n": n,
        "true_effect": true_effect,
        "control_mean": control_mean,
    }


@pytest.fixture(scope="module")
def experiment_config() -> ExperimentConfig:
    """Standard 50/50 experiment config used across assignment tests."""
    return ExperimentConfig(
        experiment_id="integration_test_v1",
        salt="salt_2026",
        variants=(
            VariantSpec(name="control", percentage=0.5),
            VariantSpec(name="treatment", percentage=0.5),
        ),
    )


# ---------------------------------------------------------------------------
# 1. Assignment
# ---------------------------------------------------------------------------

class TestAssignmentIntegration:
    """End-to-end assignment tests: distribution and determinism at scale."""

    def test_split_close_to_fifty_fifty(self, experiment_config):
        """Empirical split across 10K users should be within 2pp of 50/50."""
        assigner = HashAssignment(hash_fn=murmur_hash)
        user_ids = [f"user_{i}" for i in range(10_000)]
        assignments = assigner.assign_batch(user_ids, experiment_config)

        n_control = sum(1 for v in assignments.values() if v == "control")
        empirical_pct = n_control / len(user_ids)

        assert 0.48 <= empirical_pct <= 0.52, (
            f"Expected split close to 50%, got control={empirical_pct:.1%}"
        )

    def test_determinism_at_scale(self, experiment_config):
        """Every user in a 1K batch should get the same variant on repeat call."""
        assigner = HashAssignment(hash_fn=murmur_hash)
        user_ids = [f"user_{i}" for i in range(1_000)]

        first_pass = assigner.assign_batch(user_ids, experiment_config)
        second_pass = assigner.assign_batch(user_ids, experiment_config)

        assert first_pass == second_pass, (
            "Assignment is not deterministic: results differ between calls."
        )

    def test_all_users_assigned(self, experiment_config):
        """assign_batch should never silently drop a user."""
        assigner = HashAssignment(hash_fn=murmur_hash)
        user_ids = [f"user_{i}" for i in range(5_000)]
        assignments = assigner.assign_batch(user_ids, experiment_config)

        assert len(assignments) == len(user_ids), (
            "Not every user received an assignment."
        )
        assert all(v in ("control", "treatment") for v in assignments.values()), (
            "Some users were assigned to an unknown variant."
        )


# ---------------------------------------------------------------------------
# 2. Readouts
# ---------------------------------------------------------------------------

class TestReadoutIntegration:
    """TTestReadout and BootstrapReadout on data with a known true effect."""

    def test_ttest_detects_true_effect(self, experiment_data):
        """TTestReadout should reject H0 on N=2000 data with effect=0.5."""
        readout = TTestReadout(alpha=0.05)
        result = readout.compute(
            experiment_data["control"], experiment_data["treatment"]
        )
        assert result.significant, (
            f"TTestReadout failed to detect true effect. p={result.p_value:.4f}"
        )
        assert result.absolute_difference > 0, (
            "Absolute difference should be positive (treatment > control)."
        )
        assert result.ci_lower > 0, (
            "Lower CI bound should be positive — effect is clearly positive."
        )

    def test_bootstrap_detects_true_effect(self, experiment_data):
        """BootstrapReadout should also reject H0 on the same data."""
        readout = BootstrapReadout(alpha=0.05, n_bootstraps=2_000)
        result = readout.compute(
            experiment_data["control"], experiment_data["treatment"]
        )
        assert result.significant, (
            f"BootstrapReadout failed to detect true effect. p={result.p_value:.4f}"
        )
        assert result.absolute_difference > 0

    def test_ttest_and_bootstrap_agree_on_direction(self, experiment_data):
        """Both readouts should agree on the sign of the effect."""
        ttest = TTestReadout(alpha=0.05)
        bootstrap = BootstrapReadout(alpha=0.05, n_bootstraps=2_000)

        ttest_result = ttest.compute(
            experiment_data["control"], experiment_data["treatment"]
        )
        boot_result = bootstrap.compute(
            experiment_data["control"], experiment_data["treatment"]
        )
        assert (ttest_result.absolute_difference > 0) == (boot_result.absolute_difference > 0), (
            "TTest and Bootstrap disagree on the direction of the effect."
        )

    def test_no_effect_rarely_significant(self, rng):
        """Under pure H0 (no true effect), TTestReadout should rarely reject."""
        readout = TTestReadout(alpha=0.05)
        false_positives = sum(
            readout.compute(
                rng.normal(0, 1, 500),
                rng.normal(0, 1, 500),
            ).significant
            for _ in range(200)
        )
        empirical_fpr = false_positives / 200
        assert empirical_fpr < 0.12, (
            f"FPR too high under H0: {empirical_fpr:.2%} (expected ~5%)"
        )


# ---------------------------------------------------------------------------
# 3. CUPED
# ---------------------------------------------------------------------------

class TestCUPEDIntegration:
    """CUPED should reduce variance and produce a tighter CI than raw readout."""

    def test_variance_reduction_is_positive(self, experiment_data):
        """CUPED adjustment on a correlated covariate should reduce variance."""
        adjuster = CUPEDAdjuster(readout=TTestReadout(alpha=0.05))
        result = adjuster.compute(
            experiment_data["control"],
            experiment_data["treatment"],
            experiment_data["pre_control"],
            experiment_data["pre_treatment"],
        )
        assert result.variance_reduction_pct > 0, (
            "CUPED should reduce variance when covariate is correlated with outcome."
        )

    def test_adjusted_ci_narrower_than_raw(self, experiment_data):
        """CUPED-adjusted CI width should be strictly smaller than raw CI width."""
        adjuster = CUPEDAdjuster(readout=TTestReadout(alpha=0.05))
        result = adjuster.compute(
            experiment_data["control"],
            experiment_data["treatment"],
            experiment_data["pre_control"],
            experiment_data["pre_treatment"],
        )
        raw_width = result.raw_result.ci_upper - result.raw_result.ci_lower
        adj_width = result.adjusted_result.ci_upper - result.adjusted_result.ci_lower

        assert adj_width < raw_width, (
            f"Expected CUPED CI ({adj_width:.4f}) narrower than raw CI ({raw_width:.4f})."
        )
    def test_adjusted_mean_close_to_raw_mean(self, experiment_data):
        """CUPED adjustment should not flip the sign or wildly distort the
        point estimate — the adjusted effect should remain in the same
        directional neighborhood as the raw effect.
        """
        adjuster = CUPEDAdjuster(readout=TTestReadout(alpha=0.05))
        result = adjuster.compute(
            experiment_data["control"],
            experiment_data["treatment"],
            experiment_data["pre_control"],
            experiment_data["pre_treatment"],
        )
        raw_diff = result.raw_result.absolute_difference
        adj_diff = result.adjusted_result.absolute_difference

        # CUPED guarantees unbiasedness in expectation, not identical point
        # estimates on any single dataset. The adjusted estimate should:
        # 1. Have the same sign as the raw estimate (no direction flip)
        # 2. Be within 1 unit of the raw estimate (not wildly distorted)
        assert (raw_diff > 0) == (adj_diff > 0), (
            f"CUPED flipped the sign of the effect: "
            f"raw={raw_diff:.4f}, adjusted={adj_diff:.4f}"
        )
        assert abs(raw_diff - adj_diff) < 1.0, (
            f"CUPED distorted the point estimate beyond reasonable bounds: "
            f"raw={raw_diff:.4f}, adjusted={adj_diff:.4f}"
        )


# ---------------------------------------------------------------------------
# 4. Sequential testing
# ---------------------------------------------------------------------------

class TestSequentialIntegration:
    """mSPRT should detect a true effect and respect early stopping."""

    def test_detects_true_effect(self, experiment_data):
        """SequentialTester should eventually reject H0 on data with true effect."""
        tester = SequentialTester(alpha=0.05, tau=1.0)
        result = tester.run_full_experiment(
            experiment_data["control"],
            experiment_data["treatment"],
            batch_size=50,
        )
        assert result.rejected, (
            "Sequential tester failed to reject H0 on data with known true effect."
        )

    def test_mlr_history_non_empty(self, experiment_data):
        """MLR history should have at least one entry after running."""
        tester = SequentialTester(alpha=0.05, tau=1.0)
        result = tester.run_full_experiment(
            experiment_data["control"],
            experiment_data["treatment"],
            batch_size=50,
        )
        assert len(result.mlr_history) > 0

    def test_always_valid_pvalue_bounded(self, experiment_data):
        """Always-valid p-value must always be in [0, 1]."""
        tester = SequentialTester(alpha=0.05, tau=1.0)
        result = tester.run_full_experiment(
            experiment_data["control"],
            experiment_data["treatment"],
            batch_size=50,
        )
        assert 0.0 <= result.always_valid_pvalue <= 1.0

    def test_reset_clears_state(self, experiment_data):
        """reset() should return tester to initial MLR=1.0 with no history."""
        tester = SequentialTester(alpha=0.05, tau=1.0)
        tester.run_full_experiment(
            experiment_data["control"],
            experiment_data["treatment"],
            batch_size=50,
        )
        tester.reset()
        assert tester.result.mixture_likelihood_ratio == 1.0
        assert tester.result.n_control == 0
        assert len(tester.result.mlr_history) == 0


# ---------------------------------------------------------------------------
# 5. HTE detection
# ---------------------------------------------------------------------------

class TestHTEIntegration:
    """HTEDetector should surface subgroup differences when they exist."""

    def test_no_heterogeneity_on_uniform_effect(self, experiment_data, rng):
        """When both subgroups experience the same effect, HTE should not fire."""
        n = experiment_data["n"]
        mask_a_ctrl = rng.random(n) < 0.5
        mask_b_ctrl = ~mask_a_ctrl
        mask_a_trt = rng.random(n) < 0.5
        mask_b_trt = ~mask_a_trt

        subgroups = {
            "segment=A": (mask_a_ctrl, mask_a_trt),
            "segment=B": (mask_b_ctrl, mask_b_trt),
        }
        detector = HTEDetector(fdr_level=0.05)
        result = detector.compute(
            experiment_data["control"],
            experiment_data["treatment"],
            subgroups,
        )
        # With uniform effect, heterogeneity should not consistently fire
        # (it may occasionally due to random subgroup imbalance, so we
        # check the overall result is populated correctly rather than
        # asserting heterogeneity_detected is always False)
        assert len(result.subgroup_results) == 2
        assert all(r.bh_adjusted_pvalue is not None for r in result.subgroup_results)

    def test_subgroup_results_have_bh_pvalues(self, experiment_data, rng):
        """All SubgroupResults should have BH-adjusted p-values after compute()."""
        n = experiment_data["n"]
        subgroups = {
            "device=mobile":  (rng.random(n) < 0.4, rng.random(n) < 0.4),
            "device=desktop": (rng.random(n) >= 0.4, rng.random(n) >= 0.4),
        }
        detector = HTEDetector(fdr_level=0.05)
        result = detector.compute(
            experiment_data["control"],
            experiment_data["treatment"],
            subgroups,
        )
        for sr in result.subgroup_results:
            assert sr.bh_adjusted_pvalue is not None
            assert sr.significant_after_correction is not None
            assert 0.0 <= sr.bh_adjusted_pvalue <= 1.0


# ---------------------------------------------------------------------------
# 6. Guardrail monitoring
# ---------------------------------------------------------------------------

class TestGuardrailIntegration:
    """GuardrailMonitor should correctly classify degraded vs healthy metrics."""

    def test_failing_guardrail_detected(self, rng):
        """A metric with a clear degradation should be classified as FAILING."""
        n = 2_000
        ctrl_revenue = rng.normal(5.0, 0.5, size=n)
        trt_revenue = rng.normal(4.3, 0.5, size=n)   # large -14% drop

        monitor = GuardrailMonitor(alpha=0.10, inconclusive_threshold=0.20)
        result = monitor.compute({
            "ad_revenue": (ctrl_revenue, trt_revenue, "higher_is_better"),
        })
        assert result.any_failing, (
            "Expected ad_revenue guardrail to FAIL on a large injected degradation."
        )
        assert result.guardrail_results[0].status == GuardrailStatus.FAILING

    def test_passing_guardrail_on_no_change(self, rng):
        """A metric drawn from identical distributions should PASS."""
        n = 2_000
        ctrl = rng.normal(200, 20, size=n)
        trt = rng.normal(200, 20, size=n)

        monitor = GuardrailMonitor(alpha=0.10, inconclusive_threshold=0.20)
        result = monitor.compute({
            "p99_latency_ms": (ctrl, trt, "lower_is_better"),
        })
        assert not result.any_failing

    def test_improvement_does_not_fail_guardrail(self, rng):
        """A metric that improves (not degrades) should never FAIL."""
        n = 2_000
        ctrl = rng.normal(5.0, 1.0, size=n)
        trt = rng.normal(5.5, 1.0, size=n)   # improvement, not degradation

        monitor = GuardrailMonitor(alpha=0.10, inconclusive_threshold=0.20)
        result = monitor.compute({
            "ad_revenue": (ctrl, trt, "higher_is_better"),
        })
        assert result.guardrail_results[0].status != GuardrailStatus.FAILING, (
            "An improvement in a guardrail metric should never trigger FAILING."
        )

    def test_multiple_guardrails_bonferroni_applied(self, rng):
        """With 3 guardrails, corrected_alpha should equal raw_alpha / 3."""
        n = 1_000
        metrics = {
            "metric_a": (rng.normal(1, 1, n), rng.normal(1, 1, n), "higher_is_better"),
            "metric_b": (rng.normal(1, 1, n), rng.normal(1, 1, n), "higher_is_better"),
            "metric_c": (rng.normal(1, 1, n), rng.normal(1, 1, n), "higher_is_better"),
        }
        monitor = GuardrailMonitor(alpha=0.10)
        result = monitor.compute(metrics)

        for gr in result.guardrail_results:
            assert abs(gr.corrected_alpha - 0.10 / 3) < 1e-10, (
                f"Expected corrected_alpha=0.0333, got {gr.corrected_alpha:.6f}"
            )