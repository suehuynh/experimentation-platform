# Experimentation Platform from Scratch

A working A/B testing engine built from first principles with the actual statistical machinery underneath modern experimentation platforms.

## Why this project

Most A/B testing tutorials stop at a t-test. Production experimentation platforms (Airbnb, Netflix, Google) layer on variance reduction, sequential monitoring, and heterogeneous effect detection to ship decisions faster and more safely. This project builds each layer from scratch and validates it empirically.

## Components
- **Hash-based assignment** — deterministic, salted, fractional-mapping bucketing with stable behavior under ramp changes
- **A/A simulation validation** — 10,000 simulated A/A tests confirming ~5% false positive rate
- **CUPED variance reduction** — pre-period covariate adjustment, with empirical sample-size savings shown
- **Sequential testing (mSPRT)** — always-valid p-values, safe continuous monitoring without inflating false positive rate
- **Heterogeneous treatment effect detection** — subgroup-level effect flagging
- **Guardrail metrics** — secondary metric monitoring and alerting logic
- **Streamlit experiment console** — assign → simulate → readout → decision, end to end

## Architecture

```
experimentation_platform/
├── core/
│   ├── assignment.py
│   ├── readout.py
│   ├── cuped.py
│   ├── sequential.py
│   ├── hte.py
│   └── guardrails.py
├── simulation/
│   └── aa_simulator.py
├── app/
│   └── streamlit_console.py
├── tests/
│   ├── test_core.py
│   ├── test_assignment.py
│   └── conftest.py
├── smoke_test.py
├── requirements.txt
├── README.md
└── pyproject.toml
```

## Setup

```bash
git clone https://github.com/suehuynh/experimentation-platform-implementation.git
cd experimentation-platform-implementation
pip install -e .
pytest
```

## Status

In progress — built in weekly milestones, August 2026.

## Author

Sue Huynh — [suehuynh.com]([https://suehuynh.com](https://suehuynh.framer.website/)) | MSc Data Science @ Brown University
