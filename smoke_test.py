"""
End-to-end smoke test for the Experimentation Platform.

Wires all six modules together on synthetic data and prints
human-readable output for each stage. Not a pytest suite —
run directly to verify the full pipeline is wired correctly
before building the Streamlit console on top.

Usage:
    python smoke_test.py
"""

import numpy as np

from core.assignment import VariantSpec, ExperimentConfig, HashAssignment, murmur_hash
from core.readout import TTestReadout, BootstrapReadout
from core.cuped import CUPEDAdjuster
from core.sequential import SequentialTester
from core.hte import HTEDetector
from core.guardrails import GuardrailMonitor

RNG = np.random.default_rng(42)
SECTION = "=" * 60


def section(title: str) -> None:
    print(f"\n{SECTION}\n{title}\n{SECTION}")


# ---------------------------------------------------------------------------
# Synthetic data parameters
# ---------------------------------------------------------------------------
N = 2_000          # users per arm
TRUE_EFFECT = 0.5  # treatment lifts metric by 0.5 units
CONTROL_MEAN = 10.0
CONTROL_STD = 2.0


def make_data(effect: float = TRUE_EFFECT) -> tuple[np.ndarray, np.ndarray]:
    control = RNG.normal(loc=CONTROL_MEAN, scale=CONTROL_STD, size=N)
    treatment = RNG.normal(loc=CONTROL_MEAN + effect, scale=CONTROL_STD, size=N)
    return control, treatment


# ---------------------------------------------------------------------------
# 1. Assignment
# ---------------------------------------------------------------------------
section("1. HashAssignment — determinism + distribution check")

config = ExperimentConfig(
    experiment_id="smoke_test_v1",
    salt="salt_2026",
    variants=(
        VariantSpec(name="control", percentage=0.5),
        VariantSpec(name="treatment", percentage=0.5),
    ),
)
assigner = HashAssignment(hash_fn=murmur_hash)
user_ids = [f"user_{i}" for i in range(10_000)]
assignments = assigner.assign_batch(user_ids, config)

n_control = sum(1 for v in assignments.values() if v == "control")
n_treatment = sum(1 for v in assignments.values() if v == "treatment")
print(f"Control:   {n_control:,} ({n_control/len(user_ids):.1%})")
print(f"Treatment: {n_treatment:,} ({n_treatment/len(user_ids):.1%})")
print(f"Determinism check: {assigner.assign('user_42', config) == assigner.assign('user_42', config)}")


# ---------------------------------------------------------------------------
# 2. T-Test and Bootstrap readouts
# ---------------------------------------------------------------------------
section("2. TTestReadout + BootstrapReadout")

control, treatment = make_data()

ttest = TTestReadout(alpha=0.05)
ttest_result = ttest.compute(control, treatment)
print(ttest_result.summary())

bootstrap = BootstrapReadout(alpha=0.05, n_bootstraps=2_000)
boot_result = bootstrap.compute(control, treatment)
print(boot_result.summary())


# ---------------------------------------------------------------------------
# 3. CUPED variance reduction
# ---------------------------------------------------------------------------
section("3. CUPED — variance reduction with pre-experiment covariate")

# Pre-experiment covariate: correlated with outcome (r ≈ 0.7)
pre_control = control * 0.7 + RNG.normal(0, CONTROL_STD * 0.5, size=N)
pre_treatment = treatment * 0.7 + RNG.normal(0, CONTROL_STD * 0.5, size=N)

adjuster = CUPEDAdjuster(readout=TTestReadout(alpha=0.05))
cuped_result = adjuster.compute(control, treatment, pre_control, pre_treatment)
print(cuped_result.summary())


# ---------------------------------------------------------------------------
# 4. Sequential testing (mSPRT)
# ---------------------------------------------------------------------------
section("4. SequentialTester — mSPRT continuous monitoring")

tester = SequentialTester(alpha=0.05, tau=1.0)
seq_result = tester.run_full_experiment(control, treatment, batch_size=50)
print(seq_result.summary())
print(f"MLR history length: {len(seq_result.mlr_history)} updates")
print(f"Stopped early: {seq_result.rejected}")


# ---------------------------------------------------------------------------
# 5. HTE detection
# ---------------------------------------------------------------------------
section("5. HTEDetector — subgroup heterogeneity")

# Simulate: mobile users see 2x the effect, desktop users see no effect
mobile_mask_ctrl = RNG.random(N) < 0.4   # 40% mobile
mobile_mask_trt = RNG.random(N) < 0.4

desktop_mask_ctrl = ~mobile_mask_ctrl
desktop_mask_trt = ~mobile_mask_trt

subgroups = {
    "device=mobile":  (mobile_mask_ctrl, mobile_mask_trt),
    "device=desktop": (desktop_mask_ctrl, desktop_mask_trt),
}

detector = HTEDetector(fdr_level=0.05)
hte_result = detector.compute(control, treatment, subgroups)
print(hte_result.summary())


# ---------------------------------------------------------------------------
# 6. Guardrail monitoring
# ---------------------------------------------------------------------------
section("6. GuardrailMonitor — secondary metric protection")

# Simulate: ad revenue slightly hurt, crash rate fine, latency fine
ad_revenue_ctrl = RNG.normal(5.0, 1.0, size=N)
ad_revenue_trt = RNG.normal(4.7, 1.0, size=N)   # -6% degradation

crash_rate_ctrl = RNG.normal(0.02, 0.005, size=N)
crash_rate_trt = RNG.normal(0.02, 0.005, size=N)  # no change

latency_ctrl = RNG.normal(200, 20, size=N)
latency_trt = RNG.normal(202, 20, size=N)          # small increase

metrics = {
    "ad_revenue_per_session": (ad_revenue_ctrl, ad_revenue_trt, "higher_is_better"),
    "crash_rate":             (crash_rate_ctrl, crash_rate_trt, "lower_is_better"),
    "p99_latency_ms":         (latency_ctrl, latency_trt,       "lower_is_better"),
}

monitor = GuardrailMonitor(alpha=0.10, inconclusive_threshold=0.20)
suite_result = monitor.compute(metrics)
print(suite_result.summary())


# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------
section("FINAL SHIP DECISION")

primary_ok = ttest_result.significant and ttest_result.absolute_difference > 0
guardrails_ok = not suite_result.any_failing
hte_ok = not hte_result.heterogeneity_detected

print(f"Primary metric significant and positive : {primary_ok}")
print(f"All guardrails passing                  : {guardrails_ok}")
print(f"No harmful heterogeneity detected       : {hte_ok}")
print()
if primary_ok and guardrails_ok and hte_ok:
    print("RECOMMENDATION: SHIP ✓")
elif not guardrails_ok:
    failing = [r.metric_name for r in suite_result.failing_guardrails()]
    print(f"RECOMMENDATION: DO NOT SHIP — guardrail failure: {failing}")
elif not hte_ok:
    print("RECOMMENDATION: INVESTIGATE — heterogeneous effects detected before shipping")
else:
    print("RECOMMENDATION: WAIT — primary metric not yet significant")