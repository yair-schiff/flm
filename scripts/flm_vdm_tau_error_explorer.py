#!/usr/bin/env python3
"""Compare FLM and VP/VDM decoding-progress curves.

Run from the repo root with:

    streamlit run scripts/flm_vdm_tau_error_explorer.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from numpy.polynomial.hermite import hermgauss
from omegaconf import OmegaConf
from plotly.subplots import make_subplots
from scipy.special import log_ndtr

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "algo" / "flm_vdm.yaml"


@dataclass(frozen=True)
class CompareParams:
    vocab_size: int
    gamma_min: float
    gamma_max: float
    t_plot_max: float
    plot_points: int
    n_gh: int


def load_defaults() -> dict[str, float | int]:
    cfg = OmegaConf.load(CONFIG_PATH)
    return {
        "gamma_min": float(cfg.gamma_min),
        "gamma_max": float(cfg.gamma_max),
        "t_plot_max": float(getattr(cfg, "t_max", 1.0)),
    }


def q_correct_from_margin(margin: np.ndarray, vocab_size: int, n_gh: int) -> np.ndarray:
    margin = np.asarray(margin, dtype=np.float64)
    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x

    log_cdf = log_ndtr(z_nodes[None, :] + margin[:, None])
    log_prod = (vocab_size - 1) * log_cdf
    return np.sum(w * np.exp(log_prod), axis=-1)


def tau_from_q_correct(q_correct: np.ndarray, vocab_size: int) -> np.ndarray:
    tau = vocab_size / (vocab_size - 1.0) * (q_correct - 1.0 / vocab_size)
    return np.clip(tau, 0.0, 1.0)


def error_from_q_correct(q_correct: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - q_correct, 0.0, 1.0)


@st.cache_data(show_spinner=False)
def compute_comparison(params: CompareParams) -> dict[str, np.ndarray | float]:
    t = np.linspace(0.0, params.t_plot_max, params.plot_points, dtype=np.float64)
    flm_margin = t / np.maximum(1.0 - t, 1e-12)
    q_correct_flm = q_correct_from_margin(flm_margin, params.vocab_size, params.n_gh)
    tau_flm = tau_from_q_correct(q_correct_flm, params.vocab_size)
    p_error_flm = error_from_q_correct(q_correct_flm)

    snr_min = float(np.exp(-params.gamma_max))
    snr_max = float(np.exp(-params.gamma_min))
    snr = np.geomspace(snr_min, snr_max, params.plot_points, dtype=np.float64)
    vdm_margin = np.sqrt(snr)
    q_correct_vdm = q_correct_from_margin(vdm_margin, params.vocab_size, params.n_gh)
    tau_vdm = tau_from_q_correct(q_correct_vdm, params.vocab_size)
    p_error_vdm = error_from_q_correct(q_correct_vdm)

    # The standardized-progress formula is linear in error once q_correct is known.
    p_error_from_tau_flm = (1.0 - 1.0 / params.vocab_size) * (1.0 - tau_flm)
    p_error_from_tau_vdm = (1.0 - 1.0 / params.vocab_size) * (1.0 - tau_vdm)

    return {
        "t": t,
        "tau_flm": tau_flm,
        "q_correct_flm": q_correct_flm,
        "p_error_flm": p_error_flm,
        "p_error_from_tau_flm": p_error_from_tau_flm,
        "snr": snr,
        "tau_vdm": tau_vdm,
        "q_correct_vdm": q_correct_vdm,
        "p_error_vdm": p_error_vdm,
        "p_error_from_tau_vdm": p_error_from_tau_vdm,
        "snr_min": snr_min,
        "snr_max": snr_max,
    }


def build_tau_figure(data: dict[str, np.ndarray | float]) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("FLM: tau(t)", "VP/VDM: tau(SNR)"),
        horizontal_spacing=0.10,
    )

    fig.add_trace(
        go.Scatter(
            x=np.asarray(data["t"]),
            y=np.asarray(data["tau_flm"]),
            name="tau(t)",
            line={"width": 4},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.asarray(data["snr"]),
            y=np.asarray(data["tau_vdm"]),
            name="tau(SNR)",
            line={"width": 4},
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="physical t", row=1, col=1)
    fig.update_yaxes(title_text="tau", row=1, col=1)
    fig.update_xaxes(title_text="SNR", type="log", row=1, col=2)
    fig.update_yaxes(title_text="tau", row=1, col=2)
    fig.update_layout(height=420, legend={"orientation": "h", "y": -0.20})
    return fig


def build_error_figure(data: dict[str, np.ndarray | float]) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "FLM: P_e(t)",
            "VP/VDM: P_e(SNR)",
            "FLM: q_correct(t) and P_e(t)",
            "VP/VDM: q_correct(SNR) and P_e(SNR)",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    fig.add_trace(
        go.Scatter(
            x=np.asarray(data["t"]),
            y=np.asarray(data["p_error_flm"]),
            name="P_e(t) from q_correct",
            line={"width": 4},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.asarray(data["t"]),
            y=np.asarray(data["p_error_from_tau_flm"]),
            name="(1-1/K)(1-tau)",
            line={"dash": "dash"},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=np.asarray(data["snr"]),
            y=np.asarray(data["p_error_vdm"]),
            name="P_e(SNR) from q_correct",
            line={"width": 4},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.asarray(data["snr"]),
            y=np.asarray(data["p_error_from_tau_vdm"]),
            name="(1-1/K)(1-tau)",
            line={"dash": "dash"},
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=np.asarray(data["t"]),
            y=np.asarray(data["q_correct_flm"]),
            name="q_correct(t)",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.asarray(data["t"]),
            y=np.asarray(data["p_error_flm"]),
            name="P_e(t)",
            line={"dash": "dash"},
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=np.asarray(data["snr"]),
            y=np.asarray(data["q_correct_vdm"]),
            name="q_correct(SNR)",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.asarray(data["snr"]),
            y=np.asarray(data["p_error_vdm"]),
            name="P_e(SNR)",
            line={"dash": "dash"},
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="physical t", row=1, col=1)
    fig.update_yaxes(title_text="probability", row=1, col=1, type="log")
    fig.update_xaxes(title_text="SNR", type="log", row=1, col=2)
    fig.update_yaxes(title_text="probability", row=1, col=2, type="log")
    fig.update_xaxes(title_text="physical t", row=2, col=1)
    fig.update_yaxes(title_text="probability", row=2, col=1)
    fig.update_xaxes(title_text="SNR", type="log", row=2, col=2)
    fig.update_yaxes(title_text="probability", row=2, col=2)
    fig.update_layout(height=760, legend={"orientation": "h", "y": -0.12})
    return fig


def main() -> None:
    defaults = load_defaults()

    st.set_page_config(page_title="FLM vs VDM Tau/Error Explorer", layout="wide")
    st.title("FLM vs VDM Tau/Error Explorer")
    st.caption(
        "Compare the repo's FLM decoding-progress curve tau(t) against the VP/VDM analogue "
        "tau(SNR), and visualize the corresponding decoding-error quantity P_e."
    )

    with st.sidebar:
        st.header("Controls")
        vocab_size = st.number_input(
            "vocab_size",
            min_value=2,
            value=30522,
            step=1,
            help="Vocabulary size K used in the Gauss-Hermite approximation.",
        )
        gamma_min = st.number_input(
            "gamma_min",
            min_value=-20.0,
            max_value=20.0,
            value=float(defaults["gamma_min"]),
            step=0.01,
            format="%.4f",
        )
        gamma_max = st.number_input(
            "gamma_max",
            min_value=-20.0,
            max_value=20.0,
            value=float(defaults["gamma_max"]),
            step=0.01,
            format="%.4f",
        )
        t_plot_max = st.number_input(
            "FLM t_max for plotting",
            min_value=0.0,
            max_value=0.999999,
            value=float(defaults["t_plot_max"]),
            step=1e-4,
            format="%.6f",
            help="Use values like 0.999 or 0.9999 to inspect the clean-end tail without hitting the singular t=1 endpoint.",
        )

        with st.expander("Advanced"):
            plot_points = st.slider("plot_points", min_value=256, max_value=4096, value=2048, step=256)
            n_gh = st.slider("Gauss-Hermite nodes", min_value=20, max_value=200, value=100, step=10)

    if gamma_min >= gamma_max:
        st.error("`gamma_min` must be strictly smaller than `gamma_max`.")
        st.stop()
    if not (0.0 <= t_plot_max < 1.0):
        st.error("`FLM t_max for plotting` must satisfy 0 <= t_max < 1.")
        st.stop()

    params = CompareParams(
        vocab_size=int(vocab_size),
        gamma_min=float(gamma_min),
        gamma_max=float(gamma_max),
        t_plot_max=float(t_plot_max),
        plot_points=int(plot_points),
        n_gh=int(n_gh),
    )
    data = compute_comparison(params)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("VDM SNR range", f"{float(data['snr_min']):.3e} -> {float(data['snr_max']):.3e}")
    col2.metric("FLM P_e(t_max)", f"{float(np.asarray(data['p_error_flm'])[-1]):.3e}")
    col3.metric("VDM P_e(SNR_max)", f"{float(np.asarray(data['p_error_vdm'])[-1]):.3e}")
    col4.metric("Chance error", f"{1.0 - 1.0 / int(vocab_size):.6f}")

    st.plotly_chart(build_tau_figure(data), use_container_width=True)
    st.caption(
        "Left: the repo's FLM implementation maps physical interpolation time t to standardized "
        "decoding progress tau. Right: the VP/VDM analogue maps SNR to the same standardized tau."
    )

    st.plotly_chart(build_error_figure(data), use_container_width=True)
    st.caption(
        "For both worlds, the analogue of Equation 25 is driven by q_correct, with "
        "P_e = 1 - q_correct and tau = (q_correct - 1/K) / (1 - 1/K). The dashed curves "
        "show the equivalent linear relation P_e = (1 - 1/K)(1 - tau)."
    )


if __name__ == "__main__":
    main()
