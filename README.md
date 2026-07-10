# Experimentation Platform

A production-grade A/B testing engine built from statistical first principles. This project implements the full decision pipeline used by modern experimentation platforms — from deterministic user assignment through variance reduction, sequential monitoring, subgroup analysis, and guardrail protection — validated on 10,000 simulated A/A tests.

**Live demo:** [experimentation-platform.streamlit.app](https://ab-testing-experimentation-platform.streamlit.app/)

---

## Motivation

Most A/B testing tutorials stop at a t-test. Production systems at companies like Google, Airbnb, and Netflix layer on variance reduction, always-valid sequential monitoring, and heterogeneous effect detection to make faster, safer decisions with smaller samples. This project builds each layer from scratch to demonstrate understanding of the statistical machinery — not just the APIs.

---

## What's inside

### Core modules (`core/`)

| Module | What it does |
|---|---|
| `assignment.py` | Deterministic, salted hash-based user assignment with fractional mapping. Stable under ramp changes — no explicit user reassignment bookkeeping needed. |
| `readout.py` | Welch's t-test (`TTestReadout`) and percentile bootstrap (`BootstrapReadout`) with shared result container, CI computation, and significance flags. |
| `cuped.py` | CUPED variance reduction via pre-experiment covariate regression. Estimates theta from pooled data to avoid bias, shows empirical CI width reduction vs raw readout. |
| `sequential.py` | Mixture Sequential Probability Ratio Test (mSPRT) with closed-form Gaussian mixture update. Always-valid p-values — peeking does not inflate FPR. |
| `hte.py` | Heterogeneous treatment effect detection across pre-specified subgroups with Benjamini-Hochberg FDR correction. Flags subgroups diverging in direction or magnitude from the ATE. |
| `guardrails.py` | One-sided Welch t-test guardrail monitoring with Bonferroni correction. Classifies each metric as PASSING, FAILING, or INCONCLUSIVE. |

### Simulation (`simulation/`)

| Module | What it does |
|---|---|
| `aa_simulator.py` | Runs 10,000 simulated A/A tests and measures empirical FPR convergence to alpha. Validates platform calibration before trusting it with real decisions. |

### Tests (`tests/`)

| File | What it covers |
|---|---|
| `test_assignment.py` | Determinism, boundary conditions, coverage, distribution sanity, ramp stability, validation guards |
| `test_core.py` | End-to-end integration tests across all six modules on synthetic data with known true effects |

---

## Key design decisions

**Fractional hash mapping over modulo bucketing.** Hash outputs are normalized to `[0, 1)` before comparing to variant cutoff thresholds. This decouples bucket assignment from the number of variants, making ramp changes (growing or shrinking a variant's allocation) safe without reshuffling existing users.

**Stateless assignment.** `HashAssignment.assign()` is a pure function of `(user_id, config)` — no state stored between calls. Ramp stability is guaranteed by determinism and monotonic threshold growth, not by bookkeeping.

**Pooled theta estimation in CUPED.** The regression coefficient is estimated from the combined control + treatment population to avoid using treatment assignment in its own estimation, which would introduce bias under any true effect.

**Bonferroni over BH for guardrails.** Guardrail suites are small (3-10 metrics) and pre-specified. Familywise error rate control (Bonferroni) is preferred over FDR control (BH) because a single false FAILING verdict can unjustly block a good ship decision.

**One-sided testing for guardrails.** Guardrails only penalize degradation. An improvement in a guardrail metric should never trigger a FAILING status — the directional check is enforced before any significance threshold is applied.

---

## Validation

The A/A simulation (`aa_simulator.py`) runs 10,000 experiments where both groups are drawn from the identical distribution. A calibrated platform should produce an empirical FPR within a 99% confidence band around the configured alpha:

```
Expected FPR : 5.00%
99% band     : [4.34%, 5.66%]
```

Both `TTestReadout` and `BootstrapReadout` are validated independently.

---

## Streamlit console

A four-tab interactive console over the full pipeline:

- **Experiment Simulator** — configure assignment, readout method, CUPED, sequential monitoring, HTE detection, and guardrail checks end-to-end on synthetic or uploaded CSV data
- **Readout Explorer** — upload data or enter summary statistics; compare T-Test and Bootstrap side by side with distribution plots
- **Power Calculator** — required sample size from effect size, alpha, and variance; interactive MDE vs sample size and power vs sample size curves
- **A/A Validator** — live FPR convergence chart as simulations run, with calibration band

---

## Setup

```bash
git clone https://github.com/suehuynh/experimentation-platform-implementation.git
cd experimentation-platform-implementation
pip install -e .
pytest
streamlit run app.py
```

**Dependencies:** `numpy`, `scipy`, `pandas`, `matplotlib`, `mmh3`, `streamlit`

---

## Project structure

```
experimentation-platform-implementation/
├── core/
│   ├── assignment.py
│   ├── readout.py
│   ├── cuped.py
│   ├── sequential.py
│   ├── hte.py
│   └── guardrails.py
├── simulation/
│   └── aa_simulator.py
├── tests/
│   ├── conftest.py
│   ├── test_assignment.py
│   └── test_core.py
├── app.py
├── smoke_test.py
└── pyproject.toml
```

---

## References

- Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). [Improving the sensitivity of online controlled experiments by utilizing pre-experiment data.](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf) WSDM 2013.
- Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021). [Time-uniform, nonparametric, nonasymptotic confidence sequences.](https://arxiv.org/abs/1810.08240) Annals of Statistics.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. JRSS-B.

---

*Built by [Sue Huynh](https://github.com/suehuynh) — incoming MSc Data Science, Brown University (2026).*
