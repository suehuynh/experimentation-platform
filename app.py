"""
Experimentation Platform Console
app.py — Streamlit entry point

Four-tab interface over the core statistical engine:
    1. Experiment Simulator  — end-to-end pipeline on synthetic or uploaded data
    2. Readout Explorer      — paste / upload data, choose readout method
    3. Power Calculator      — required sample size from effect / alpha / variance
    4. A/A Validator         — empirical FPR convergence chart
"""

import io
from typing import Literal

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from core.assignment import VariantSpec, ExperimentConfig, HashAssignment, murmur_hash
from core.readout import TTestReadout, BootstrapReadout
from core.cuped import CUPEDAdjuster
from core.sequential import SequentialTester
from core.hte import HTEDetector
from core.guardrails import GuardrailMonitor, GuardrailStatus
from simulation.aa_simulator import AASimulationRunner, MetricGenerator

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Experimentation Platform",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling — monospace data aesthetic, slate + amber accent
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: -0.02em;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 3px solid #f59e0b;
        padding: 0.75rem 1rem;
        border-radius: 0 4px 4px 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    .status-pass  { color: #16a34a; font-weight: 600; }
    .status-fail  { color: #dc2626; font-weight: 600; }
    .status-inc   { color: #d97706; font-weight: 600; }
    .status-sig   { color: #2563eb; font-weight: 600; }
    .mono         { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("# ⚗️ Experimentation Platform")
st.markdown(
    "Statistical machinery for A/B testing — assignment, readout, "
    "CUPED, sequential testing, HTE detection, and guardrail monitoring."
)
st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Experiment Simulator",
    "📊 Readout Explorer",
    "🎯 Power Calculator",
    "🔁 A/A Validator",
])


# ===========================================================================
# TAB 1 — Experiment Simulator
# ===========================================================================

with tab1:
    st.markdown("### Experiment Simulator")
    st.markdown(
        "Configure an experiment end-to-end: assign users, choose a readout "
        "method, apply CUPED variance reduction, run sequential monitoring, "
        "detect subgroup heterogeneity, and check guardrail metrics."
    )

    # --- Data source ---
    data_source = st.radio(
        "Data source",
        ["Synthetic data", "Upload CSV"],
        horizontal=True,
    )

    control_data: np.ndarray | None = None
    treatment_data: np.ndarray | None = None
    covariate_control: np.ndarray | None = None
    covariate_treatment: np.ndarray | None = None

    if data_source == "Synthetic data":
        c1, c2, c3, c4 = st.columns(4)
        n = c1.number_input("Users per arm", min_value=100, max_value=50_000,
                            value=2_000, step=100)
        control_mean = c2.number_input("Control mean", value=10.0, step=0.5)
        effect_size = c3.number_input("True effect (lift)", value=0.5, step=0.1)
        std = c4.number_input("Std deviation", min_value=0.1, value=2.0, step=0.1)

        seed = st.number_input("Random seed", value=42, step=1)
        rng = np.random.default_rng(int(seed))

        control_data = rng.normal(loc=control_mean, scale=std, size=int(n))
        treatment_data = rng.normal(loc=control_mean + effect_size, scale=std, size=int(n))
        covariate_control = control_data * 0.7 + rng.normal(0, std * 0.5, size=int(n))
        covariate_treatment = treatment_data * 0.7 + rng.normal(0, std * 0.5, size=int(n))

    else:
        st.markdown(
            "Upload a CSV with columns: `control`, `treatment`, "
            "and optionally `pre_control`, `pre_treatment` for CUPED."
        )
        uploaded = st.file_uploader("Upload experiment CSV", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            if "control" not in df.columns or "treatment" not in df.columns:
                st.error("CSV must contain 'control' and 'treatment' columns.")
            else:
                control_data = df["control"].dropna().values
                treatment_data = df["treatment"].dropna().values
                if "pre_control" in df.columns and "pre_treatment" in df.columns:
                    covariate_control = df["pre_control"].dropna().values
                    covariate_treatment = df["pre_treatment"].dropna().values

    if control_data is not None and treatment_data is not None:
        st.divider()

        # --- Readout config ---
        st.markdown("#### Readout")
        rc1, rc2 = st.columns(2)
        readout_method = rc1.selectbox("Method", ["T-Test", "Bootstrap"])
        alpha = rc2.slider("Alpha (α)", min_value=0.01, max_value=0.20,
                           value=0.05, step=0.01)

        if readout_method == "T-Test":
            readout = TTestReadout(alpha=alpha)
        else:
            n_boot = st.number_input("Bootstrap resamples", value=5_000,
                                     min_value=1_000, max_value=20_000, step=1_000)
            readout = BootstrapReadout(alpha=alpha, n_bootstraps=int(n_boot))

        result = readout.compute(control_data, treatment_data)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Control mean", f"{result.control_mean:.4f}")
        col_b.metric("Treatment mean", f"{result.treatment_mean:.4f}")
        col_c.metric("Absolute lift", f"{result.absolute_difference:+.4f}")
        rel = f"{result.relative_difference:+.2%}" if result.relative_difference else "N/A"
        col_d.metric("Relative lift", rel)

        col_e, col_f, col_g = st.columns(3)
        col_e.metric("p-value", f"{result.p_value:.4f}")
        col_f.metric("95% CI", f"[{result.ci_lower:+.4f}, {result.ci_upper:+.4f}]")
        sig_label = "✅ Significant" if result.significant else "❌ Not significant"
        col_g.metric("Decision", sig_label)

        # --- CUPED ---
        st.divider()
        st.markdown("#### CUPED Variance Reduction")
        if covariate_control is not None and covariate_treatment is not None:
            run_cuped = st.checkbox("Apply CUPED adjustment", value=True)
            if run_cuped:
                try:
                    adjuster = CUPEDAdjuster(readout=readout)
                    cuped = adjuster.compute(control_data, treatment_data,
                                             covariate_control, covariate_treatment)
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Covariate correlation",
                               f"{cuped.covariate_correlation:.4f}")
                    cc2.metric("Variance reduction",
                               f"{cuped.variance_reduction_pct:.1f}%")
                    raw_w = cuped.raw_result.ci_upper - cuped.raw_result.ci_lower
                    adj_w = cuped.adjusted_result.ci_upper - cuped.adjusted_result.ci_lower
                    cc3.metric("CI width reduction",
                               f"{(1 - adj_w/raw_w)*100:.1f}%")
                    st.caption(
                        f"Adjusted p-value: {cuped.adjusted_result.p_value:.4f}  |  "
                        f"Theta (θ): {cuped.theta:.4f}"
                    )
                except Exception as e:
                    st.warning(f"CUPED failed: {e}")
        else:
            st.info(
                "Upload a CSV with `pre_control` and `pre_treatment` columns, "
                "or use synthetic data to enable CUPED."
            )

        # --- Sequential ---
        st.divider()
        st.markdown("#### Sequential Monitoring (mSPRT)")
        sc1, sc2 = st.columns(2)
        batch_size = sc1.number_input("Batch size", value=50, min_value=10,
                                      max_value=500, step=10)
        tau = sc2.number_input("Prior τ (sensitivity)", value=1.0,
                               min_value=0.1, max_value=5.0, step=0.1)

        tester = SequentialTester(alpha=alpha, tau=tau)
        seq_result = tester.run_full_experiment(control_data, treatment_data,
                                                batch_size=int(batch_size))

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(seq_result.mlr_history, color="#2563eb", linewidth=1.5,
                label="Mixture Likelihood Ratio")
        ax.axhline(1 / alpha, color="#dc2626", linewidth=1,
                   linestyle="--", label=f"Rejection threshold (1/α={1/alpha:.0f})")
        ax.set_xlabel("Update (batch)", fontsize=9)
        ax.set_ylabel("MLR", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_title("mSPRT — Evidence accumulation over time", fontsize=10)
        st.pyplot(fig)
        plt.close()

        seq_col1, seq_col2, seq_col3 = st.columns(3)
        seq_col1.metric("Final MLR", f"{seq_result.mixture_likelihood_ratio:.2f}")
        seq_col2.metric("Always-valid p",
                        f"{seq_result.always_valid_pvalue:.4f}")
        seq_col3.metric("Rejected H0",
                        "Yes ✅" if seq_result.rejected else "No ❌")

        # --- HTE ---
        st.divider()
        st.markdown("#### Heterogeneous Treatment Effects")
        st.caption(
            "Define boolean subgroup masks. For synthetic data, random 50/50 "
            "device splits are used as a demonstration."
        )

        rng2 = np.random.default_rng(99)
        n_obs = len(control_data)
        mobile_ctrl = rng2.random(n_obs) < 0.4
        mobile_trt = rng2.random(n_obs) < 0.4

        subgroups = {
            "device=mobile":  (mobile_ctrl, mobile_trt),
            "device=desktop": (~mobile_ctrl, ~mobile_trt),
        }
        fdr_level = st.slider("FDR level (BH correction)", 0.01, 0.20,
                              value=0.05, step=0.01)
        detector = HTEDetector(readout=readout, fdr_level=fdr_level)
        hte = detector.compute(control_data, treatment_data, subgroups)

        hte_rows = []
        for sr in hte.subgroup_results:
            hte_rows.append({
                "Subgroup": sr.subgroup_name,
                "n control": sr.n_control,
                "n treatment": sr.n_treatment,
                "Effect": f"{sr.readout.absolute_difference:+.4f}",
                "Raw p": f"{sr.raw_pvalue:.4f}",
                "BH p": f"{sr.bh_adjusted_pvalue:.4f}",
                "Significant": "✅" if sr.significant_after_correction else "❌",
            })
        st.dataframe(pd.DataFrame(hte_rows), use_container_width=True)
        het_label = "⚠️ Heterogeneity detected" if hte.heterogeneity_detected \
            else "✅ No significant heterogeneity"
        st.caption(f"Overall ATE: {hte.overall_result.absolute_difference:+.4f} — {het_label}")

        # --- Guardrails ---
        st.divider()
        st.markdown("#### Guardrail Metrics")
        st.caption(
            "For demonstration, two synthetic guardrail metrics are generated "
            "alongside the primary metric. Upload a CSV with additional columns "
            "to monitor real guardrails."
        )

        rng3 = np.random.default_rng(7)
        n_g = len(control_data)
        guardrail_metrics = {
            "ad_revenue_per_session": (
                rng3.normal(5.0, 1.0, n_g),
                rng3.normal(4.85, 1.0, n_g),
                "higher_is_better",
            ),
            "crash_rate": (
                rng3.normal(0.02, 0.005, n_g),
                rng3.normal(0.02, 0.005, n_g),
                "lower_is_better",
            ),
        }
        g_alpha = st.slider("Guardrail alpha", 0.05, 0.30, value=0.10, step=0.05)
        monitor = GuardrailMonitor(alpha=g_alpha)
        suite = monitor.compute(guardrail_metrics)

        g_rows = []
        for gr in suite.guardrail_results:
            status_emoji = {"PASSING": "✅", "FAILING": "❌", "INCONCLUSIVE": "⚠️"}
            g_rows.append({
                "Metric": gr.metric_name,
                "Control mean": f"{gr.control_mean:.4f}",
                "Treatment mean": f"{gr.treatment_mean:.4f}",
                "Diff": f"{gr.absolute_difference:+.4f}",
                "BF p-value": f"{gr.bonferroni_pvalue:.4f}",
                "Status": f"{status_emoji.get(gr.status.value, '')} {gr.status.value}",
            })
        st.dataframe(pd.DataFrame(g_rows), use_container_width=True)

        # --- Final verdict ---
        st.divider()
        st.markdown("#### Ship Decision")
        primary_ok = result.significant and result.absolute_difference > 0
        guardrails_ok = not suite.any_failing
        hte_ok = not hte.heterogeneity_detected

        v1, v2, v3 = st.columns(3)
        v1.metric("Primary metric", "✅ Pass" if primary_ok else "❌ Fail")
        v2.metric("Guardrails", "✅ Pass" if guardrails_ok else "❌ Fail")
        v3.metric("HTE check", "✅ Pass" if hte_ok else "⚠️ Review")

        if primary_ok and guardrails_ok and hte_ok:
            st.success("**RECOMMENDATION: SHIP** — primary metric significant, "
                       "all guardrails passing, no harmful heterogeneity.")
        elif not guardrails_ok:
            failing = [r.metric_name for r in suite.failing_guardrails()]
            st.error(f"**DO NOT SHIP** — guardrail failure: {', '.join(failing)}")
        elif not hte_ok:
            st.warning("**INVESTIGATE** — heterogeneous effects detected. "
                       "Review subgroup impacts before shipping.")
        else:
            st.info("**WAIT** — primary metric not yet significant.")


# ===========================================================================
# TAB 2 — Readout Explorer
# ===========================================================================

with tab2:
    st.markdown("### Readout Explorer")
    st.markdown(
        "Upload a CSV or paste summary statistics to run a statistical readout "
        "and compare T-Test vs Bootstrap outputs side by side."
    )

    input_mode = st.radio("Input mode", ["Upload CSV", "Enter summary stats"],
                          horizontal=True)

    r_control: np.ndarray | None = None
    r_treatment: np.ndarray | None = None

    if input_mode == "Upload CSV":
        st.markdown("CSV must have columns `control` and `treatment`.")
        r_file = st.file_uploader("Upload CSV", type="csv", key="readout_csv")
        if r_file:
            r_df = pd.read_csv(r_file)
            if "control" in r_df.columns and "treatment" in r_df.columns:
                r_control = r_df["control"].dropna().values
                r_treatment = r_df["treatment"].dropna().values
            else:
                st.error("CSV must contain 'control' and 'treatment' columns.")
    else:
        st.markdown("Simulate data from summary statistics.")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**Control**")
            c_mean = st.number_input("Mean", value=10.0, key="c_mean")
            c_std = st.number_input("Std", value=2.0, min_value=0.01, key="c_std")
            c_n = st.number_input("N", value=1000, min_value=10, key="c_n")
        with sc2:
            st.markdown("**Treatment**")
            t_mean = st.number_input("Mean", value=10.4, key="t_mean")
            t_std = st.number_input("Std", value=2.0, min_value=0.01, key="t_std")
            t_n = st.number_input("N", value=1000, min_value=10, key="t_n")

        r_rng = np.random.default_rng(0)
        r_control = r_rng.normal(c_mean, c_std, int(c_n))
        r_treatment = r_rng.normal(t_mean, t_std, int(t_n))

    if r_control is not None and r_treatment is not None:
        r_alpha = st.slider("Alpha", 0.01, 0.20, value=0.05, step=0.01,
                            key="r_alpha")
        r_n_boot = st.number_input("Bootstrap resamples", value=5_000,
                                   min_value=1_000, max_value=20_000,
                                   step=1_000, key="r_boot")

        if st.button("Run readout comparison"):
            with st.spinner("Running T-Test..."):
                tt = TTestReadout(alpha=r_alpha)
                tt_res = tt.compute(r_control, r_treatment)

            with st.spinner(f"Running Bootstrap ({int(r_n_boot):,} resamples)..."):
                bt = BootstrapReadout(alpha=r_alpha, n_bootstraps=int(r_n_boot))
                bt_res = bt.compute(r_control, r_treatment)

            col_tt, col_bt = st.columns(2)

            with col_tt:
                st.markdown("#### T-Test")
                st.text(tt_res.summary())

            with col_bt:
                st.markdown("#### Bootstrap")
                st.text(bt_res.summary())

            # Distribution plot
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            for ax, data, label, color in zip(
                axes,
                [r_control, r_treatment],
                ["Control", "Treatment"],
                ["#94a3b8", "#2563eb"],
            ):
                ax.hist(data, bins=40, color=color, alpha=0.7, edgecolor="none")
                ax.axvline(np.mean(data), color="#dc2626", linewidth=1.5,
                           linestyle="--", label=f"Mean={np.mean(data):.2f}")
                ax.set_title(label, fontsize=10)
                ax.legend(fontsize=8)
            fig.suptitle("Metric distributions", fontsize=11)
            st.pyplot(fig)
            plt.close()


# ===========================================================================
# TAB 3 — Power Calculator
# ===========================================================================

with tab3:
    st.markdown("### Power Calculator")
    st.markdown(
        "Calculate the minimum sample size required to detect an effect, "
        "or explore the trade-off between sample size, power, and MDE."
    )

    pc1, pc2, pc3, pc4 = st.columns(4)
    p_baseline = pc1.number_input("Baseline mean", value=10.0, step=0.1)
    p_mde = pc2.number_input("Minimum detectable effect (absolute)",
                              value=0.5, min_value=0.01, step=0.05)
    p_std = pc3.number_input("Expected std deviation",
                              value=2.0, min_value=0.01, step=0.1)
    p_alpha = pc4.slider("Alpha (α)", 0.01, 0.20, value=0.05, step=0.01,
                         key="p_alpha")
    p_power = st.slider("Desired power (1-β)", 0.50, 0.99, value=0.80, step=0.01)

    from scipy import stats as sp_stats

    def required_n(mde: float, std: float, alpha: float, power: float) -> int:
        z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
        z_beta = sp_stats.norm.ppf(power)
        effect_size = mde / std
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))

    n_required = required_n(p_mde, p_std, p_alpha, p_power)

    r1, r2, r3 = st.columns(3)
    r1.metric("Required n per arm", f"{n_required:,}")
    r2.metric("Total users required", f"{n_required * 2:,}")
    r3.metric("Effect size (Cohen's d)", f"{p_mde / p_std:.3f}")

    st.divider()
    st.markdown("#### Sample size vs. MDE trade-off")

    mde_range = np.linspace(max(0.05, p_mde * 0.2), p_mde * 3, 50)
    n_range = [required_n(m, p_std, p_alpha, p_power) for m in mde_range]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(mde_range, n_range, color="#2563eb", linewidth=2)
    ax.axvline(p_mde, color="#f59e0b", linestyle="--", linewidth=1.5,
               label=f"Current MDE={p_mde}")
    ax.axhline(n_required, color="#dc2626", linestyle="--", linewidth=1,
               label=f"Required n={n_required:,}")
    ax.scatter([p_mde], [n_required], color="#dc2626", zorder=5, s=60)
    ax.set_xlabel("Minimum detectable effect (absolute)", fontsize=9)
    ax.set_ylabel("Required n per arm", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=8)
    ax.set_title("How sample size scales with MDE", fontsize=10)
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.markdown("#### Power vs. sample size")

    n_sweep = np.linspace(50, n_required * 2, 80).astype(int)
    z_alpha = sp_stats.norm.ppf(1 - p_alpha / 2)
    power_sweep = [
        sp_stats.norm.cdf((p_mde / p_std) * np.sqrt(n / 2) - z_alpha)
        for n in n_sweep
    ]

    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.plot(n_sweep, power_sweep, color="#16a34a", linewidth=2)
    ax2.axhline(p_power, color="#f59e0b", linestyle="--", linewidth=1.5,
                label=f"Target power={p_power:.0%}")
    ax2.axvline(n_required, color="#dc2626", linestyle="--", linewidth=1,
                label=f"Required n={n_required:,}")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_xlabel("Sample size per arm", fontsize=9)
    ax2.set_ylabel("Statistical power", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.set_title("Power as a function of sample size", fontsize=10)
    st.pyplot(fig2)
    plt.close()


# ===========================================================================
# TAB 4 — A/A Validator
# ===========================================================================

with tab4:
    st.markdown("### A/A Validator")
    st.markdown(
        "Run simulated A/A tests (no true effect) and measure the empirical "
        "false positive rate. A correctly calibrated platform should converge "
        "to your configured alpha."
    )

    av1, av2, av3 = st.columns(3)
    av_n_sim = av1.number_input("Number of simulations", value=2_000,
                                min_value=200, max_value=10_000, step=200)
    av_n_obs = av2.number_input("Observations per arm", value=500,
                                min_value=50, max_value=5_000, step=50)
    av_alpha = av3.slider("Alpha", 0.01, 0.20, value=0.05, step=0.01,
                          key="av_alpha")
    av_method = st.selectbox("Readout method", ["T-Test", "Bootstrap", "Both"])
    if av_method == "Bootstrap":
        av_n_boot = st.number_input("Bootstrap resamples per simulation",
                                    value=1_000, min_value=500,
                                    max_value=5_000, step=500)

    if st.button("Run A/A validation"):
        generator = MetricGenerator(
            n_control=int(av_n_obs),
            n_treatment=int(av_n_obs),
            random_seed=0,
        )

        methods_to_run = (
            ["T-Test", "Bootstrap"] if av_method == "Both" else [av_method]
        )

        for method_name in methods_to_run:
            st.markdown(f"#### {method_name}")

            if method_name == "T-Test":
                readout_inst = TTestReadout(alpha=av_alpha)
            else:
                n_b = int(av_n_boot) if av_method != "T-Test" else 1_000
                readout_inst = BootstrapReadout(alpha=av_alpha, n_bootstraps=n_b)

            runner = AASimulationRunner(
                readout=readout_inst,
                generator=generator,
                n_simulations=int(av_n_sim),
            )

            progress = st.progress(0, text=f"Running {method_name} simulations...")
            fp_counts = []
            false_positives = 0

            for i in range(int(av_n_sim)):
                ctrl_b, trt_b = generator.sample()
                res = readout_inst.compute(ctrl_b, trt_b)
                if res.significant:
                    false_positives += 1
                fp_counts.append(false_positives / (i + 1))
                if (i + 1) % max(1, int(av_n_sim) // 20) == 0:
                    progress.progress(
                        (i + 1) / int(av_n_sim),
                        text=f"{method_name}: {i+1:,}/{int(av_n_sim):,} runs"
                    )

            progress.empty()

            empirical_fpr = false_positives / int(av_n_sim)
            std_err = np.sqrt(av_alpha * (1 - av_alpha) / int(av_n_sim))
            band_low = av_alpha - 3 * std_err
            band_high = av_alpha + 3 * std_err
            calibrated = band_low <= empirical_fpr <= band_high

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Empirical FPR", f"{empirical_fpr:.2%}")
            mc2.metric("Expected FPR", f"{av_alpha:.2%}")
            mc3.metric("99% band",
                       f"[{band_low:.2%}, {band_high:.2%}]")
            mc4.metric("Calibrated",
                       "✅ Yes" if calibrated else "❌ No")

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(fp_counts, color="#2563eb", linewidth=1.2,
                    label="Empirical FPR (running)")
            ax.axhline(av_alpha, color="#dc2626", linestyle="--",
                       linewidth=1.5, label=f"Target α={av_alpha}")
            ax.axhline(band_low, color="#94a3b8", linestyle=":",
                       linewidth=1, label="99% band")
            ax.axhline(band_high, color="#94a3b8", linestyle=":", linewidth=1)
            ax.fill_between(range(len(fp_counts)),
                            band_low, band_high,
                            color="#94a3b8", alpha=0.15)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_xlabel("Simulations run", fontsize=9)
            ax.set_ylabel("Empirical FPR", fontsize=9)
            ax.legend(fontsize=8)
            ax.set_title(f"{method_name} FPR convergence", fontsize=10)
            st.pyplot(fig)
            plt.close()

            st.divider()