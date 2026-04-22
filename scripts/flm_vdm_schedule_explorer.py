#!/usr/bin/env python3
"""Interactive explorer for the FLM VDM time warp and noise schedule.

Run from the repo root with:

    streamlit run scripts/flm_vdm_schedule_explorer.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from omegaconf import OmegaConf
from numpy.polynomial.hermite import hermgauss
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline
from scipy.special import expit
from scipy.special import log_ndtr

REPO_ROOT = Path(__file__).resolve().parents[1]


CONFIG_PATH = REPO_ROOT / "configs" / "algo" / "flm_vdm.yaml"


@dataclass(frozen=True)
class ScheduleParams:
    tau_min: float
    tau_max: float
    warp_mix: float
    gamma_min: float
    gamma_max: float
    vocab_size: int
    lut_points: int
    plot_points: int


def load_algo_defaults() -> dict[str, float]:
    cfg = OmegaConf.load(CONFIG_PATH)
    return {
        "t_min": float(cfg.t_min),
        "t_max": float(cfg.t_max),
        "warp_mix": float(cfg.warp_mix),
        "gamma_min": float(cfg.gamma_min),
        "gamma_max": float(cfg.gamma_max),
    }


def compute_alpha_exact(gamma: np.ndarray, vocab_size: int, n_gh: int = 100) -> np.ndarray:
    """Replica of the LUT source used in utils.py, without the training stack."""
    gamma = np.asarray(gamma, dtype=np.float64)
    sigma = np.maximum(1.0 - gamma, 1e-12)
    m_c = gamma / sigma

    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x
    log_cdf = log_ndtr(z_nodes[None, :] + m_c[:, None])
    log_prod_c = (vocab_size - 1) * log_cdf
    q_c = np.sum(w * np.exp(log_prod_c), axis=-1)

    alpha = vocab_size / (vocab_size - 1.0) * (q_c - 1.0 / vocab_size)
    alpha += (gamma - 1.0) * 1e-10
    return np.clip(alpha, 0.0, 1.0)


def build_luts_like_repo(vocab_size: int, n_points: int = 10000) -> tuple[CubicSpline, CubicSpline]:
    gamma_vals = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    alpha_vals = compute_alpha_exact(gamma_vals, vocab_size=vocab_size)

    lut_g2a = CubicSpline(gamma_vals, alpha_vals)

    sorted_indices = np.argsort(alpha_vals)
    gamma_sorted = gamma_vals[sorted_indices]
    alpha_sorted = alpha_vals[sorted_indices]
    unique_alpha, unique_indices = np.unique(alpha_sorted, return_index=True)
    unique_gamma = gamma_sorted[unique_indices]
    lut_a2g = CubicSpline(unique_alpha, unique_gamma)
    return lut_a2g, lut_g2a


def alpha_to_gamma(alpha: np.ndarray, lut: CubicSpline) -> np.ndarray:
    return np.clip(lut(alpha), 0.0, 1.0)


def d_alpha_to_gamma(alpha: np.ndarray, lut: CubicSpline) -> np.ndarray:
    return np.asarray(lut.derivative()(alpha))


@st.cache_resource(show_spinner=False)
def build_luts_cached(vocab_size: int, lut_points: int):
    return build_luts_like_repo(vocab_size=vocab_size, n_points=lut_points)


def standardized_progress_to_accuracy(tau: np.ndarray, vocab_size: int) -> np.ndarray:
    base_accuracy = 1.0 / vocab_size
    return base_accuracy + (1.0 - base_accuracy) * tau


def standardized_progress_to_error(tau: np.ndarray, vocab_size: int) -> np.ndarray:
    return (1.0 - 1.0 / vocab_size) * (1.0 - tau)


def map_tau_to_physical_t(
    tau: np.ndarray,
    tau_min: float,
    tau_max: float,
    warp_mix: float,
    lut_a2g,
) -> dict[str, np.ndarray | float]:
    t_warp = alpha_to_gamma(tau, lut_a2g)
    dt_dtau_warp = d_alpha_to_gamma(tau, lut_a2g)

    tau_endpoints = np.array([tau_min, tau_max], dtype=np.float64)
    t_endpoints = alpha_to_gamma(tau_endpoints, lut_a2g)
    t_phys_min = float(t_endpoints[0])
    t_phys_max = float(t_endpoints[1])
    linear_slope = (t_phys_max - t_phys_min) / (tau_max - tau_min)
    t_linear = t_phys_min + linear_slope * (tau - tau_min)
    dt_dtau_linear = np.full_like(tau, linear_slope)

    t_mixed = (1.0 - warp_mix) * t_warp + warp_mix * t_linear
    dt_dtau_mixed = (1.0 - warp_mix) * dt_dtau_warp + warp_mix * dt_dtau_linear

    return {
        "t_warp": t_warp,
        "dt_dtau_warp": dt_dtau_warp,
        "t_linear": t_linear,
        "dt_dtau_linear": dt_dtau_linear,
        "t_mixed": t_mixed,
        "dt_dtau_mixed": dt_dtau_mixed,
        "t_phys_min": t_phys_min,
        "t_phys_max": t_phys_max,
        "linear_slope": linear_slope,
    }


@st.cache_data(show_spinner=False)
def compute_schedule(params: ScheduleParams) -> dict[str, np.ndarray | float]:
    lut_a2g, _ = build_luts_cached(params.vocab_size, params.lut_points)

    tau = np.linspace(params.tau_min, params.tau_max, params.plot_points, dtype=np.float64)
    mapped = map_tau_to_physical_t(
        tau=tau,
        tau_min=params.tau_min,
        tau_max=params.tau_max,
        warp_mix=params.warp_mix,
        lut_a2g=lut_a2g,
    )
    t = np.asarray(mapped["t_mixed"])
    dt_dtau = np.asarray(mapped["dt_dtau_mixed"])

    gamma = params.gamma_max + (params.gamma_min - params.gamma_max) * t
    snr = np.exp(-gamma)
    log_snr = -gamma
    nsr = np.exp(gamma)
    alpha = np.sqrt(expit(-gamma))
    sigma = np.sqrt(expit(gamma))
    snr_prime_t = (params.gamma_max - params.gamma_min) * snr
    loss_weight = snr_prime_t * dt_dtau
    accuracy = standardized_progress_to_accuracy(tau, params.vocab_size)
    decoding_error = standardized_progress_to_error(tau, params.vocab_size)
    raw_uniform_t = np.linspace(params.tau_min, params.tau_max, params.plot_points, dtype=np.float64)

    return {
        "tau": tau,
        "t": t,
        "t_warp": np.asarray(mapped["t_warp"]),
        "t_linear": np.asarray(mapped["t_linear"]),
        "dt_dtau": dt_dtau,
        "dt_dtau_warp": np.asarray(mapped["dt_dtau_warp"]),
        "dt_dtau_linear": np.asarray(mapped["dt_dtau_linear"]),
        "t_phys_min": float(mapped["t_phys_min"]),
        "t_phys_max": float(mapped["t_phys_max"]),
        "gamma": gamma,
        "snr": snr,
        "log_snr": log_snr,
        "nsr": nsr,
        "alpha": alpha,
        "sigma": sigma,
        "snr_prime_t": snr_prime_t,
        "loss_weight": loss_weight,
        "accuracy": accuracy,
        "decoding_error": decoding_error,
        "raw_uniform_t": raw_uniform_t,
    }


def sample_spacing(
    params: ScheduleParams,
    num_steps: int,
) -> dict[str, np.ndarray]:
    lut_a2g, _ = build_luts_cached(params.vocab_size, params.lut_points)
    tau_steps = np.linspace(params.tau_min, params.tau_max, num_steps, dtype=np.float64)
    mapped = map_tau_to_physical_t(
        tau=tau_steps,
        tau_min=params.tau_min,
        tau_max=params.tau_max,
        warp_mix=params.warp_mix,
        lut_a2g=lut_a2g,
    )
    raw_uniform_t = np.linspace(params.tau_min, params.tau_max, num_steps, dtype=np.float64)

    return {
        "step_idx": np.arange(num_steps),
        "tau_steps": tau_steps,
        "current_t": np.asarray(mapped["t_mixed"]),
        "warp_t": np.asarray(mapped["t_warp"]),
        "linear_t": np.asarray(mapped["t_linear"]),
        "uniform_t_raw": raw_uniform_t,
    }


def build_warp_figure(schedule: dict[str, np.ndarray | float], spacing: dict[str, np.ndarray]) -> go.Figure:
    tau = schedule["tau"]
    t = schedule["t"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "t(tau): Current vs Reference Curves",
            "tau(t): Inverse View",
            "dt/dtau",
            "Sampling Spacing Across Step Index",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.10,
    )

    fig.add_trace(
        go.Scatter(x=tau, y=schedule["t_warp"], name="pure FLM warp"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tau,
            y=schedule["t_linear"],
            name="same-endpoint linear",
            line={"dash": "dash"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tau,
            y=t,
            name="current warp_mix",
            line={"width": 4},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=t, y=tau, name="tau(t)", showlegend=False),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(x=tau, y=schedule["dt_dtau_warp"], name="dt/dtau pure warp"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tau,
            y=schedule["dt_dtau_linear"],
            name="dt/dtau linear",
            line={"dash": "dash"},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tau,
            y=schedule["dt_dtau"],
            name="dt/dtau current",
            line={"width": 4},
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=spacing["step_idx"],
            y=spacing["current_t"],
            mode="lines+markers",
            name="current warped_tau",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=spacing["step_idx"],
            y=spacing["linear_t"],
            mode="lines+markers",
            name="same-endpoint linear",
            line={"dash": "dash"},
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=spacing["step_idx"],
            y=spacing["uniform_t_raw"],
            mode="lines+markers",
            name="raw uniform_t",
            line={"dash": "dot"},
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="tau", row=1, col=1)
    fig.update_yaxes(title_text="physical t", row=1, col=1)
    fig.update_xaxes(title_text="physical t", row=1, col=2)
    fig.update_yaxes(title_text="tau", row=1, col=2)
    fig.update_xaxes(title_text="tau", row=2, col=1)
    fig.update_yaxes(title_text="dt/dtau", row=2, col=1)
    fig.update_xaxes(title_text="step index", row=2, col=2)
    fig.update_yaxes(title_text="physical t", row=2, col=2)
    fig.update_layout(height=820, legend={"orientation": "h", "y": -0.08})
    return fig


def build_noise_figure(schedule: dict[str, np.ndarray | float], log_scale: bool) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Physical Gamma and Log-SNR vs t",
            "SNR / NSR vs t",
            "alpha(t) and sigma(t)",
            "Weight Factors vs tau",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.10,
    )

    fig.add_trace(go.Scatter(x=schedule["t"], y=schedule["gamma"], name="gamma(t)"), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=schedule["t"],
            y=schedule["log_snr"],
            name="log SNR(t)",
            line={"dash": "dash"},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=schedule["t"], y=schedule["snr"], name="SNR(t)"), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=schedule["t"],
            y=schedule["nsr"],
            name="NSR(t)",
            line={"dash": "dash"},
        ),
        row=1,
        col=2,
    )

    fig.add_trace(go.Scatter(x=schedule["t"], y=schedule["alpha"], name="alpha(t)"), row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=schedule["t"],
            y=schedule["sigma"],
            name="sigma(t)",
            line={"dash": "dash"},
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=schedule["tau"], y=schedule["dt_dtau"], name="dt/dtau"),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=schedule["tau"],
            y=schedule["snr_prime_t"],
            name="snr_prime(t)",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=schedule["tau"],
            y=schedule["loss_weight"],
            name="loss weight",
            line={"width": 4},
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="physical t", row=1, col=1)
    fig.update_yaxes(title_text="value", row=1, col=1)
    fig.update_xaxes(title_text="physical t", row=1, col=2)
    fig.update_yaxes(title_text="value", row=1, col=2, type="log" if log_scale else "linear")
    fig.update_xaxes(title_text="physical t", row=2, col=1)
    fig.update_yaxes(title_text="value", row=2, col=1)
    fig.update_xaxes(title_text="tau", row=2, col=2)
    fig.update_yaxes(title_text="value", row=2, col=2, type="log" if log_scale else "linear")
    fig.update_layout(height=820, legend={"orientation": "h", "y": -0.08})
    return fig


def build_vocab_figure(
    params: ScheduleParams,
    compare_vocab_sizes: list[int],
    log_scale: bool,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("tau(t) by vocab size", "loss weight(tau) by vocab size"),
        horizontal_spacing=0.10,
    )

    for vocab_size in compare_vocab_sizes:
        compare_params = ScheduleParams(
            tau_min=params.tau_min,
            tau_max=params.tau_max,
            warp_mix=params.warp_mix,
            gamma_min=params.gamma_min,
            gamma_max=params.gamma_max,
            vocab_size=vocab_size,
            lut_points=params.lut_points,
            plot_points=params.plot_points,
        )
        schedule = compute_schedule(compare_params)
        label = f"K={vocab_size}"
        fig.add_trace(go.Scatter(x=schedule["t"], y=schedule["tau"], name=label), row=1, col=1)
        fig.add_trace(go.Scatter(x=schedule["tau"], y=schedule["loss_weight"], name=label), row=1, col=2)

    fig.update_xaxes(title_text="physical t", row=1, col=1)
    fig.update_yaxes(title_text="tau", row=1, col=1)
    fig.update_xaxes(title_text="tau", row=1, col=2)
    fig.update_yaxes(title_text="loss weight", row=1, col=2, type="log" if log_scale else "linear")
    fig.update_layout(height=460, legend={"orientation": "h", "y": -0.20})
    return fig


def hydra_override_snippet(params: ScheduleParams) -> str:
    return "\n".join(
        [
            f"algo.t_min={params.tau_min}",
            f"algo.t_max={params.tau_max}",
            f"algo.warp_mix={params.warp_mix}",
            f"algo.gamma_min={params.gamma_min}",
            f"algo.gamma_max={params.gamma_max}",
        ]
    )


def main() -> None:
    defaults = load_algo_defaults()
    st.set_page_config(page_title="FLM VDM Schedule Explorer", layout="wide")
    st.title("FLM VDM Schedule Explorer")
    st.caption(
        "This app uses the same LUT-based tau <-> t mapping as the repo in "
        "`utils.build_luts(...)`, so the plots follow the current FLM/VDM code path."
    )

    with st.sidebar:
        st.header("Controls")
        st.caption(f"Defaults loaded from `{CONFIG_PATH.relative_to(REPO_ROOT)}`.")
        tau_min = st.number_input("t_min", min_value=0.0, max_value=1.0, value=defaults["t_min"], step=0.01)
        tau_max = st.number_input("t_max", min_value=0.0, max_value=1.0, value=defaults["t_max"], step=0.01)
        warp_mix = st.slider("warp_mix", min_value=0.0, max_value=1.0, value=defaults["warp_mix"], step=0.01)
        gamma_min = st.number_input("gamma_min", min_value=-20.0, max_value=20.0, value=defaults["gamma_min"], step=0.25)
        gamma_max = st.number_input("gamma_max", min_value=-20.0, max_value=20.0, value=defaults["gamma_max"], step=0.25)

        preset_to_vocab = {
            "LM1B / bert-base-uncased (30522)": 30522,
            "OWT / gpt2 + mask slot (50258)": 50258,
            "Small demo (1024)": 1024,
            "Large demo (100000)": 100000,
            "Custom": None,
        }
        preset_label = st.selectbox("Vocab preset", list(preset_to_vocab.keys()), index=0)
        default_vocab = preset_to_vocab[preset_label] or 30522
        vocab_size = st.number_input(
            "vocab_size",
            min_value=2,
            value=int(default_vocab),
            step=1,
            help="This is the model vocabulary size used for the LUT. For GPT-2 in this repo, that includes the extra mask slot.",
        )

        num_steps = st.slider("Sampling steps to visualize", min_value=4, max_value=128, value=32, step=1)
        log_scale = st.checkbox("Use log scale for SNR and weight plots", value=True)

        with st.expander("Advanced"):
            lut_points = st.slider("LUT resolution", min_value=1000, max_value=10000, value=10000, step=500)
            plot_points = st.slider("Plot resolution", min_value=256, max_value=4096, value=2048, step=256)
            compare_vocab_sizes = st.multiselect(
                "Vocab sizes to compare",
                options=[256, 1024, 8192, 30522, 50258, 100000, int(vocab_size)],
                default=[1024, 30522, int(vocab_size)],
            )

    if tau_min >= tau_max:
        st.error("`t_min` must be strictly smaller than `t_max`.")
        st.stop()
    if gamma_min >= gamma_max:
        st.error("`gamma_min` must be strictly smaller than `gamma_max`.")
        st.stop()

    params = ScheduleParams(
        tau_min=float(tau_min),
        tau_max=float(tau_max),
        warp_mix=float(warp_mix),
        gamma_min=float(gamma_min),
        gamma_max=float(gamma_max),
        vocab_size=int(vocab_size),
        lut_points=int(lut_points),
        plot_points=int(plot_points),
    )
    schedule = compute_schedule(params)
    spacing = sample_spacing(params, num_steps=num_steps)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Physical t range", f"{schedule['t_phys_min']:.4f} -> {schedule['t_phys_max']:.4f}")
    col2.metric("SNR range", f"{schedule['snr'].min():.3e} -> {schedule['snr'].max():.3e}")
    col3.metric("Peak loss weight", f"{schedule['loss_weight'].max():.3e}")
    col4.metric("Mean loss weight", f"{schedule['loss_weight'].mean():.3e}")

    st.info(
        "Equation 25 is the standardized decoding-progress time. In the repo implementation, "
        "tau behaves like standardized correct-decoding progress: "
        "tau(t) = (p_correct(t) - 1/K) / (1 - 1/K), so decoding_error(t) = (1 - 1/K) * (1 - tau(t))."
    )

    warp_tab, noise_tab, vocab_tab, export_tab = st.tabs(
        ["Warp and Spacing", "Noise and Weight", "Vocab Comparison", "Config Snippet"]
    )

    with warp_tab:
        st.plotly_chart(build_warp_figure(schedule, spacing), use_container_width=True)
        st.caption(
            "With `warped_tau`, the config values `t_min` and `t_max` are tau endpoints. "
            "The app also shows the same-endpoint linear reference used by `warp_mix=1`."
        )

    with noise_tab:
        st.plotly_chart(build_noise_figure(schedule, log_scale=log_scale), use_container_width=True)
        stats_col1, stats_col2 = st.columns(2)
        stats_col1.write(
            {
                "accuracy_start": float(schedule["accuracy"][0]),
                "accuracy_end": float(schedule["accuracy"][-1]),
                "decoding_error_start": float(schedule["decoding_error"][0]),
                "decoding_error_end": float(schedule["decoding_error"][-1]),
            }
        )
        stats_col2.write(
            {
                "dt_dtau_min": float(np.min(schedule["dt_dtau"])),
                "dt_dtau_max": float(np.max(schedule["dt_dtau"])),
                "loss_weight_min": float(np.min(schedule["loss_weight"])),
                "loss_weight_max": float(np.max(schedule["loss_weight"])),
            }
        )

    with vocab_tab:
        compare_vocab_sizes = sorted(set(int(v) for v in compare_vocab_sizes + [int(vocab_size)]))
        st.plotly_chart(
            build_vocab_figure(params, compare_vocab_sizes=compare_vocab_sizes, log_scale=log_scale),
            use_container_width=True,
        )
        st.caption(
            "Larger vocabularies make the pure FLM warp more concentrated. The current curve also reflects `warp_mix`, "
            "so you can see how much linearization flattens that vocab-size effect."
        )

    with export_tab:
        st.code(hydra_override_snippet(params), language="bash")
        st.write(
            {
                "vocab_size": int(vocab_size),
                "tau_interval": [float(tau_min), float(tau_max)],
                "physical_t_interval_under_current_warp": [
                    float(schedule["t_phys_min"]),
                    float(schedule["t_phys_max"]),
                ],
            }
        )


if __name__ == "__main__":
    main()
