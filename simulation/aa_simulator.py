"""
Experimentation Platform: A/A Simulation Validator.

This module empirically validates the statistical calibration of the
experimentation platform's readout layer. By running thousands of A/A
tests (experiments where both groups are drawn from the identical
distribution, so no true effect exists), it confirms that the platform's
false positive rate (FPR) converges to the configured significance level
alpha -- the foundational guarantee any credible A/B testing system must
provide before being trusted with real product decisions.

A correctly calibrated platform should produce:
    FPR ≈ alpha (typically 0.05), within sampling noise.

Systematic deviation in either direction signals a broken platform:
    FPR >> alpha : inflated false positives (anti-conservative)
    FPR << alpha : deflated false positives (over-conservative, high FNR)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.readout import ReadoutStrategy, ReadoutResult, TTestReadout


# ---------------------------------------------------------------------------
# 1. Simulation result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AASimulationResult:
    """Immutable container summarising the outcome of an A/A simulation run.

    Attributes:
        n_simulations: Total number of A/A experiments simulated.
        n_false_positives: Number of runs where the readout incorrectly
            declared significance (the only possible error in an A/A test,
            since the true effect is always zero).
        empirical_fpr: Observed false positive rate
            (n_false_positives / n_simulations).
        expected_fpr: The alpha level the readout was configured with;
            the theoretical FPR the platform should converge toward.
        fpr_std_error: Standard error of the empirical FPR estimate,
            computed as sqrt(expected_fpr * (1 - expected_fpr) / n_simulations).
            Used to assess whether empirical_fpr is within acceptable
            sampling noise of expected_fpr.
        calibrated: True if empirical_fpr falls within a 99% confidence
            band around expected_fpr (i.e. within ~3 standard errors).
        readout_method: Label identifying which readout strategy was
            validated (e.g. 't-test', 'bootstrap').
    """
    n_simulations: int
    n_false_positives: int
    empirical_fpr: float
    expected_fpr: float
    fpr_std_error: float
    calibrated: bool
    readout_method: str

    def summary(self) -> str:
        """Return a human-readable calibration report for logging or display.

        Returns:
            Multi-line string covering simulation parameters, empirical FPR,
            standard error, and a plain-English calibration verdict.
        """
        band_low = self.expected_fpr - 3 * self.fpr_std_error
        band_high = self.expected_fpr + 3 * self.fpr_std_error
        verdict = (
            "PASS: platform is correctly calibrated."
            if self.calibrated
            else "FAIL: FPR deviates beyond acceptable sampling noise."
        )
        return (
            f"A/A Simulation Calibration Report ({self.readout_method})\n"
            f"{'=' * 55}\n"
            f"Simulations     : {self.n_simulations:,}\n"
            f"False positives : {self.n_false_positives:,}\n"
            f"{'─' * 55}\n"
            f"Empirical FPR   : {self.empirical_fpr:.2%}\n"
            f"Expected FPR    : {self.expected_fpr:.2%}\n"
            f"Std error       : {self.fpr_std_error:.4f}\n"
            f"99% band        : [{band_low:.2%}, {band_high:.2%}]\n"
            f"{'─' * 55}\n"
            f"Verdict         : {verdict}\n"
        )

# ---------------------------------------------------------------------------
# 2. Data generator
# ---------------------------------------------------------------------------

class MetricGenerator:
    """Synthetic metric data generator for A/A simulation.

    Generates paired control/treatment samples from the same underlying
    distribution, ensuring no true effect exists by construction. Both
    groups share identical distributional parameters -- this is what
    makes an experiment an A/A test rather than an A/B test.

    Distribution choice does not affect A/A test validity as long as
    both groups are drawn from the same distribution. For large sample
    sizes, CLT ensures the sampling distribution of the mean is
    approximately normal regardless of the raw metric shape.
    """

    def __init__(
        self,
        n_control: int = 1_000,
        n_treatment: int = 1_000,
        mean: float = 10.0,
        std: float = 2.0,
        random_seed: int | None = None,
    ):
        """Initialise the generator with shared distributional parameters.

        Args:
            n_control: Number of observations to draw for the control group
                per simulation run.
            n_treatment: Number of observations to draw for the treatment
                group per simulation run.
            mean: Mean of the shared normal distribution both groups are
                drawn from.
            std: Standard deviation of the shared normal distribution.
            random_seed: Optional seed for the numpy RNG. Set for
                reproducible simulation runs; leave None for fresh
                randomness each run.
        """
        if n_control <= 0 or n_treatment <=0:
            raise ValueError("Sample sizes must be larger than 0.")
        if std <= 0:
            raise ValueError("Standard Deviation must be larger than 0.")
        
        self._n_control = n_control
        self._n_treatment = n_treatment
        self._mean = mean
        self._std = std
        self._random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        """Draw one paired (control, treatment) sample from the shared distribution.

        Both arrays are drawn independently from the same normal distribution,
        so the expected difference in means is zero by construction.

        Returns:
            Tuple of (control, treatment) as 1-D numpy arrays of lengths
            n_control and n_treatment respectively.
        """
        control = self._rng.normal(loc=self._mean, scale=self._std,
                                   size=self._n_control)
        treatment = self._rng.normal(loc=self._mean, scale=self._std,
                                   size=self._n_treatment)
        return (control, treatment)


# ---------------------------------------------------------------------------
# 3. Simulation runner
# ---------------------------------------------------------------------------

class AASimulationRunner:
    """Empirical calibration validator for ReadoutStrategy implementations.

    Runs a configurable number of A/A experiments and measures the
    platform's empirical false positive rate. Accepts any ReadoutStrategy
    subclass (TTestReadout, BootstrapReadout, or future implementations)
    so calibration can be validated independently for each strategy.

    The calibration criterion uses a 99% confidence band around the
    expected FPR (alpha +/- 3 * std_error), following the same
    multiple-testing-aware standard that production experimentation
    platforms apply to their own self-tests.
    """

    def __init__(
        self,
        readout: ReadoutStrategy,
        generator: MetricGenerator,
        n_simulations: int = 10_000,
    ):
        """Initialise the simulation runner.

        Args:
            readout: The ReadoutStrategy instance to validate. Its alpha
                level is used as the expected FPR benchmark.
            generator: MetricGenerator instance controlling sample sizes
                and the shared null distribution.
            n_simulations: Number of A/A experiments to run. Higher values
                narrow the confidence band on the empirical FPR estimate.
                10,000 is the industry standard for platform validation.
        """
        if n_simulations <= 0:
            raise ValueError("Number of simulations must be larger than 0.")
        self._readout = readout
        self._generator = generator
        self._n_simulations = n_simulations

    def run(self) -> AASimulationResult:
        """Execute all A/A simulation runs and return calibration results.

        For each simulation:
            1. Draw a fresh (control, treatment) pair from the generator.
            2. Pass both arrays to the readout strategy's compute() method.
            3. Record whether the readout declared significance (a false
               positive, since no true effect exists in an A/A test).

        After all runs, compute the empirical FPR and assess calibration
        against the 99% confidence band.

        Returns:
            AASimulationResult summarising calibration outcome across all runs.
        """
        false_positives = 0

        for i in range(self._n_simulations):
            control, treatment = self._generator.sample()
            result = self._readout.compute(control, treatment)
            readout_method = result.method
            if result.significant:
                false_positives += 1

        empirical_fpr = false_positives / self._n_simulations
        expected_fpr = self._readout._alpha
        fpr_std_error = np.sqrt(expected_fpr * (1 - expected_fpr) / self._n_simulations)
        calibrated = abs(empirical_fpr - expected_fpr) <= 3 * fpr_std_error
        
        return AASimulationResult(
            n_simulations=self._n_simulations,
            n_false_positives=false_positives,
            empirical_fpr=empirical_fpr,
            expected_fpr=expected_fpr,
            fpr_std_error=fpr_std_error,
            calibrated=calibrated,
            readout_method = readout_method
        )

    def run_comparison(self) -> dict[str, AASimulationResult]:
        """Run the simulation with both TTestReadout and BootstrapReadout
        and return results side by side.

        Useful for quickly comparing calibration across strategies using
        the same generator and n_simulations settings, without having to
        construct two separate AASimulationRunner instances manually.

        Returns:
            Dict mapping strategy label (e.g. 't-test', 'bootstrap') to
            its AASimulationResult.
        """
        
        ttest_simulation = AASimulationRunner(readout=TTestReadout(alpha=self._readout._alpha), 
                                              generator=self._generator,
                                              n_simulations=self._n_simulations)
        
        ttest_result = ttest_simulation.run()

        from core.readout import BootstrapReadout
        bootstrap_simulation = AASimulationRunner(BootstrapReadout(alpha=self._readout._alpha), 
                                                  generator=self._generator,
                                                  n_simulations=self._n_simulations)
        bootstrap_result = bootstrap_simulation.run()
        return {ttest_result.readout_method: ttest_result, 
                bootstrap_result.readout_method: bootstrap_result}