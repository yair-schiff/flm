#!/usr/bin/env python3
"""Interactive explorer for FLM and VP-native FLMVDM time warps.

This is the main combined explorer for schedule, warp, and tau/error views.

Run from the repo root with:

    streamlit run scripts/flm_vdm_schedule_explorer.py
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
from scipy.interpolate import CubicSpline
from scipy.special import expit, log_ndtr
from scripts.flm_vdm_tau_error_explorer import (
    CompareParams,
    NOTES_PATH,
    build_derivative_figure as build_tau_error_derivative_figure,
    build_error_figure as build_tau_error_error_figure,
    build_sampling_comparison_figure as build_tau_error_sampling_figure,
    build_tau_figure as build_tau_error_tau_figure,
    compute_comparison as compute_tau_error_comparison,
    enrich_comparison as enrich_tau_error_comparison,
    load_notes_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "algo" / "flm_vdm.yaml"


@dataclass(frozen=True)
class ScheduleParams:
    interpolant_type: str
    tau_min: float
    tau_max: float
    gamma_min: float
    gamma_max: float
    vocab_size: int
    lut_points: int
    plot_points: int


def load_algo_defaults() -> dict[str, float | str]:
    cfg = OmegaConf.load(CONFIG_PATH)
    return {
        "interpolant_type": str(getattr(cfg, "interpolant_type", "vp_vdm")),
        "t_min": float(cfg.t_min),
        "t_max": float(cfg.t_max),
        "gamma_min": float(cfg.gamma_min),
        "gamma_max": float(cfg.gamma_max),
    }


def standardized_progress_to_accuracy(tau: np.ndarray, vocab_size: int) -> np.ndarray:
    base_accuracy = 1.0 / vocab_size
    return base_accuracy + (1.0 - base_accuracy) * tau


def standardized_progress_to_error(tau: np.ndarray, vocab_size: int) -> np.ndarray:
    return (1.0 - 1.0 / vocab_size) * (1.0 - tau)


def compute_flm_tau_exact(t: np.ndarray, vocab_size: int, n_gh: int = 100) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    sigma = np.maximum(1.0 - t, 1e-12)
    m_c = t / sigma

    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x
    log_cdf = log_ndtr(z_nodes[None, :] + m_c[:, None])
    q_c = np.sum(w * np.exp((vocab_size - 1) * log_cdf), axis=-1)

    tau = vocab_size / (vocab_size - 1.0) * (q_c - 1.0 / vocab_size)
    tau += (t - 1.0) * 1e-10
    return np.clip(tau, 0.0, 1.0)


def compute_vp_tau_exact(gamma: np.ndarray, vocab_size: int, n_gh: int = 100) -> np.ndarray:
    gamma = np.asarray(gamma, dtype=np.float64)
    m_c = np.exp(-0.5 * gamma)

    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x
    log_cdf = log_ndtr(z_nodes[None, :] + m_c[:, None])
    q_c = np.sum(w * np.exp((vocab_size - 1) * log_cdf), axis=-1)

    tau = vocab_size / (vocab_size - 1.0) * (q_c - 1.0 / vocab_size)
    tau -= gamma * 1e-10
    return np.clip(tau, 0.0, 1.0)


def build_flm_luts(vocab_size: int, n_points: int = 10000) -> tuple[CubicSpline, CubicSpline]:
    t_vals = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    tau_vals = compute_flm_tau_exact(t_vals, vocab_size=vocab_size)

    lut_t2tau = CubicSpline(t_vals, tau_vals)

    sorted_indices = np.argsort(tau_vals)
    tau_sorted = tau_vals[sorted_indices]
    t_sorted = t_vals[sorted_indices]
    unique_tau, unique_indices = np.unique(tau_sorted, return_index=True)
    unique_t = t_sorted[unique_indices]
    lut_tau2t = CubicSpline(unique_tau, unique_t)
    return lut_tau2t, lut_t2tau


def build_vp_luts(
    vocab_size: int,
    gamma_min: float,
    gamma_max: float,
    n_points: int = 10000,
) -> tuple[CubicSpline, CubicSpline]:
    gamma_vals = np.linspace(gamma_min, gamma_max, n_points, dtype=np.float64)
    tau_vals = compute_vp_tau_exact(gamma_vals, vocab_size=vocab_size)

    lut_gamma2tau = CubicSpline(gamma_vals, tau_vals)

    sorted_indices = np.argsort(tau_vals)
    tau_sorted = tau_vals[sorted_indices]
    gamma_sorted = gamma_vals[sorted_indices]
    unique_tau, unique_indices = np.unique(tau_sorted, return_index=True)
    unique_gamma = gamma_sorted[unique_indices]
    tau_augmented = np.concatenate(([0.0], unique_tau, [1.0]))
    gamma_augmented = np.concatenate(([gamma_max], unique_gamma, [gamma_min]))
    tau_augmented, unique_indices = np.unique(tau_augmented, return_index=True)
    gamma_augmented = gamma_augmented[unique_indices]
    lut_tau2gamma = CubicSpline(tau_augmented, gamma_augmented)
    return lut_tau2gamma, lut_gamma2tau


def tau_to_coord(tau: np.ndarray, lut: CubicSpline, clip_unit_interval: bool) -> np.ndarray:
    tau = np.asarray(tau)
    if not clip_unit_interval:
        tau = np.clip(tau, 0.0, 1.0)
    values = np.asarray(lut(tau))
    if clip_unit_interval:
        return np.clip(values, 0.0, 1.0)
    return values


def d_tau_to_coord(tau: np.ndarray, lut: CubicSpline) -> np.ndarray:
    tau = np.clip(np.asarray(tau), 0.0, 1.0)
    return np.asarray(lut.derivative()(tau))


@st.cache_resource(show_spinner=False)
def build_luts_cached(
    interpolant_type: str,
    vocab_size: int,
    gamma_min: float,
    gamma_max: float,
    lut_points: int,
):
    if interpolant_type == "flm_linear":
        return build_flm_luts(vocab_size=vocab_size, n_points=lut_points)
    if interpolant_type == "vp_vdm":
        return build_vp_luts(
            vocab_size=vocab_size,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_points=lut_points,
        )
    raise ValueError(f"Unknown interpolant_type: {interpolant_type}")


@st.cache_data(show_spinner=False)
def compute_schedule(params: ScheduleParams) -> dict[str, np.ndarray | float | str]:
    tau = np.linspace(params.tau_min, params.tau_max, params.plot_points, dtype=np.float64)
    accuracy = standardized_progress_to_accuracy(tau, params.vocab_size)
    decoding_error = standardized_progress_to_error(tau, params.vocab_size)

    if params.interpolant_type == "flm_linear":
        lut_tau2coord, _ = build_luts_cached(
            params.interpolant_type,
            params.vocab_size,
            params.gamma_min,
            params.gamma_max,
            params.lut_points,
        )
        t = tau_to_coord(tau, lut_tau2coord, clip_unit_interval=True)
        dt_dtau = d_tau_to_coord(tau, lut_tau2coord)
        gamma = params.gamma_max + (params.gamma_min - params.gamma_max) * t
        alpha = np.sqrt(expit(-gamma))
        sigma = np.sqrt(expit(gamma))
        snr = np.exp(-gamma)
        nsr = np.exp(gamma)
        log_snr = -gamma
        snr_prime_t = snr * (params.gamma_max - params.gamma_min)
        loss_weight = snr_prime_t * dt_dtau
        reweight_factor = 2.0 * t * dt_dtau / np.maximum((1.0 - t) ** 3, 1e-12)
        return {
            "interpolant_type": params.interpolant_type,
            "tau": tau,
            "coord": t,
            "coord_name": "physical t",
            "coord_symbol": "t",
            "inverse_x": t,
            "inverse_y": tau,
            "derivative": dt_dtau,
            "derivative_name": "dt/dtau",
            "gamma": gamma,
            "alpha": alpha,
            "sigma": sigma,
            "snr": snr,
            "nsr": nsr,
            "log_snr": log_snr,
            "loss_weight": loss_weight,
            "weight_factor": snr_prime_t,
            "weight_factor_name": "snr_prime(t)",
            "weight_compare": reweight_factor,
            "weight_compare_name": "(2 * t * dt/dtau) / (1 - t)^3",
            "accuracy": accuracy,
            "decoding_error": decoding_error,
        }

    lut_tau2coord, _ = build_luts_cached(
        params.interpolant_type,
        params.vocab_size,
        params.gamma_min,
        params.gamma_max,
        params.lut_points,
    )
    gamma = tau_to_coord(tau, lut_tau2coord, clip_unit_interval=False)
    dgamma_dtau = d_tau_to_coord(tau, lut_tau2coord)
    alpha = np.sqrt(expit(-gamma))
    sigma = np.sqrt(expit(gamma))
    snr = np.exp(-gamma)
    nsr = np.exp(gamma)
    log_snr = -gamma
    loss_weight = snr * (-dgamma_dtau)
    return {
        "interpolant_type": params.interpolant_type,
        "tau": tau,
        "coord": gamma,
        "coord_name": "gamma",
        "coord_symbol": "gamma",
        "inverse_x": gamma,
        "inverse_y": tau,
        "derivative": dgamma_dtau,
        "derivative_name": "dgamma/dtau",
        "gamma": gamma,
        "alpha": alpha,
        "sigma": sigma,
        "snr": snr,
        "nsr": nsr,
        "log_snr": log_snr,
        "loss_weight": loss_weight,
        "weight_factor": -dgamma_dtau,
        "weight_factor_name": "-dgamma/dtau",
        "weight_compare": loss_weight,
        "weight_compare_name": "exp(-gamma) * (-dgamma/dtau)",
        "accuracy": accuracy,
        "decoding_error": decoding_error,
    }


def sample_spacing(params: ScheduleParams, num_steps: int) -> dict[str, np.ndarray]:
    tau_steps = np.linspace(params.tau_min, params.tau_max, num_steps, dtype=np.float64)
    schedule = compute_schedule(
        ScheduleParams(
            interpolant_type=params.interpolant_type,
            tau_min=params.tau_min,
            tau_max=params.tau_max,
            gamma_min=params.gamma_min,
            gamma_max=params.gamma_max,
            vocab_size=params.vocab_size,
            lut_points=params.lut_points,
            plot_points=num_steps,
        )
    )
    return {
        "step_idx": np.arange(num_steps),
        "tau_steps": tau_steps,
        "coord_steps": np.asarray(schedule["coord"]),
        "derivative_steps": np.asarray(schedule["derivative"]),
        "gamma_steps": np.asarray(schedule["gamma"]),
        "log_snr_steps": np.asarray(schedule["log_snr"]),
        "snr_steps": np.asarray(schedule["snr"]),
        "nsr_steps": np.asarray(schedule["nsr"]),
        "alpha_steps": np.asarray(schedule["alpha"]),
        "sigma_steps": np.asarray(schedule["sigma"]),
        "weight_factor_steps": np.asarray(schedule["weight_factor"]),
        "loss_weight_steps": np.asarray(schedule["loss_weight"]),
        "weight_compare_steps": np.asarray(schedule["weight_compare"]),
    }


def build_warp_figure(schedule: dict[str, np.ndarray | float | str], spacing: dict[str, np.ndarray]) -> go.Figure:
    tau = np.asarray(schedule["tau"])
    coord = np.asarray(schedule["coord"])
    marker_style = {
        "mode": "markers",
        "marker": {"size": 7, "symbol": "diamond"},
        "showlegend": False,
    }

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"{schedule['coord_symbol']}(tau)",
            f"tau({schedule['coord_symbol']})",
            str(schedule["derivative_name"]),
            "Sampling Spacing Across Step Index",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.10,
    )

    fig.add_trace(go.Scatter(x=tau, y=coord, name=f"{schedule['coord_symbol']}(tau)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.asarray(schedule["inverse_x"]), y=np.asarray(schedule["inverse_y"]), name="inverse", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=tau, y=np.asarray(schedule["derivative"]), name=str(schedule["derivative_name"])), row=2, col=1)
    fig.add_trace(go.Scatter(x=spacing["step_idx"], y=spacing["coord_steps"], mode="lines+markers", name=schedule["coord_name"]), row=2, col=2)
    fig.add_trace(go.Scatter(x=spacing["step_idx"], y=spacing["alpha_steps"], mode="lines+markers", name="alpha", line={"dash": "dash"}), row=2, col=2)
    fig.add_trace(go.Scatter(x=spacing["step_idx"], y=spacing["sigma_steps"], mode="lines+markers", name="sigma", line={"dash": "dot"}), row=2, col=2)
    fig.add_trace(go.Scatter(x=spacing["tau_steps"], y=spacing["coord_steps"], **marker_style), row=1, col=1)
    fig.add_trace(go.Scatter(x=spacing["coord_steps"], y=spacing["tau_steps"], **marker_style), row=1, col=2)
    fig.add_trace(go.Scatter(x=spacing["tau_steps"], y=spacing["derivative_steps"], **marker_style), row=2, col=1)

    fig.update_xaxes(title_text="tau", row=1, col=1)
    fig.update_yaxes(title_text=str(schedule["coord_name"]), row=1, col=1)
    fig.update_xaxes(title_text=str(schedule["coord_name"]), row=1, col=2)
    fig.update_yaxes(title_text="tau", row=1, col=2)
    fig.update_xaxes(title_text="tau", row=2, col=1)
    fig.update_yaxes(title_text=str(schedule["derivative_name"]), row=2, col=1)
    fig.update_xaxes(title_text="step index", row=2, col=2)
    fig.update_yaxes(title_text="value", row=2, col=2)
    fig.update_layout(height=820, legend={"orientation": "h", "y": -0.08})
    return fig


def build_noise_figure(
    schedule: dict[str, np.ndarray | float | str],
    spacing: dict[str, np.ndarray],
    log_scale: bool,
) -> go.Figure:
    marker_style = {
        "mode": "markers",
        "marker": {"size": 7, "symbol": "diamond"},
        "showlegend": False,
    }
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "gamma(tau) and log SNR(tau)",
            "SNR / NSR vs tau",
            "alpha(tau) and sigma(tau)",
            "Weight Factors vs tau",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.10,
    )

    fig.add_trace(go.Scatter(x=schedule["tau"], y=schedule["gamma"], name="gamma(tau)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=schedule["tau"], y=schedule["log_snr"], name="log SNR(tau)", line={"dash": "dash"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=schedule["tau"], y=schedule["snr"], name="SNR(tau)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=schedule["tau"], y=schedule["nsr"], name="NSR(tau)", line={"dash": "dash"}), row=1, col=2)
    fig.add_trace(go.Scatter(x=schedule["tau"], y=schedule["alpha"], name="alpha(tau)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=schedule["tau"], y=schedule["sigma"], name="sigma(tau)", line={"dash": "dash"}), row=2, col=1)
    fig.add_trace(go.Scatter(x=schedule["tau"], y=schedule["weight_factor"], name=str(schedule["weight_factor_name"])), row=2, col=2)
    fig.add_trace(go.Scatter(x=schedule["tau"], y=schedule["loss_weight"], name="loss weight", line={"width": 4}), row=2, col=2)
    fig.add_trace(go.Scatter(x=spacing["tau_steps"], y=spacing["gamma_steps"], **marker_style), row=1, col=1)
    fig.add_trace(go.Scatter(x=spacing["tau_steps"], y=spacing["log_snr_steps"], **marker_style), row=1, col=1)
    fig.add_trace(go.Scatter(x=spacing["tau_steps"], y=spacing["snr_steps"], **marker_style), row=1, col=2)
    fig.add_trace(go.Scatter(x=spacing["tau_steps"], y=spacing["nsr_steps"], **marker_style), row=1, col=2)
    fig.add_trace(go.Scatter(x=spacing["tau_steps"], y=spacing["alpha_steps"], **marker_style), row=2, col=1)
    fig.add_trace(go.Scatter(x=spacing["tau_steps"], y=spacing["sigma_steps"], **marker_style), row=2, col=1)
    fig.add_trace(go.Scatter(x=spacing["tau_steps"], y=spacing["weight_factor_steps"], **marker_style), row=2, col=2)
    fig.add_trace(go.Scatter(x=spacing["tau_steps"], y=spacing["loss_weight_steps"], **marker_style), row=2, col=2)

    fig.update_xaxes(title_text="tau", row=1, col=1)
    fig.update_yaxes(title_text="value", row=1, col=1)
    fig.update_xaxes(title_text="tau", row=1, col=2)
    fig.update_yaxes(title_text="value", row=1, col=2, type="log" if log_scale else "linear")
    fig.update_xaxes(title_text="tau", row=2, col=1)
    fig.update_yaxes(title_text="value", row=2, col=1)
    fig.update_xaxes(title_text="tau", row=2, col=2)
    fig.update_yaxes(title_text="value", row=2, col=2, type="log" if log_scale else "linear")
    fig.update_layout(height=820, legend={"orientation": "h", "y": -0.08})
    return fig


def build_weight_compare_figure(
    schedule: dict[str, np.ndarray | float | str],
    spacing: dict[str, np.ndarray],
    log_scale: bool,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Linear Scale", "Log Scale"),
        horizontal_spacing=0.10,
    )

    trace_specs = (
        (str(schedule["weight_factor_name"]), np.asarray(schedule["weight_factor"]), {"width": 4}),
        (str(schedule["weight_compare_name"]), np.asarray(schedule["weight_compare"]), {}),
    )

    for name, y, line in trace_specs:
        fig.add_trace(go.Scatter(x=schedule["tau"], y=y, name=name, line=line), row=1, col=1)
        fig.add_trace(go.Scatter(x=schedule["tau"], y=y, name=name, line=line, showlegend=False), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=spacing["tau_steps"],
            y=spacing["weight_factor_steps"],
            mode="markers",
            marker={"size": 7, "symbol": "diamond"},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=spacing["tau_steps"],
            y=spacing["weight_compare_steps"],
            mode="markers",
            marker={"size": 7, "symbol": "diamond"},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=spacing["tau_steps"],
            y=spacing["weight_factor_steps"],
            mode="markers",
            marker={"size": 7, "symbol": "diamond"},
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=spacing["tau_steps"],
            y=spacing["weight_compare_steps"],
            mode="markers",
            marker={"size": 7, "symbol": "diamond"},
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="tau", row=1, col=1)
    fig.update_yaxes(title_text="value", row=1, col=1)
    fig.update_xaxes(title_text="tau", row=1, col=2)
    fig.update_yaxes(title_text="value", row=1, col=2, type="log" if log_scale else "linear")
    fig.update_layout(height=460, legend={"orientation": "h", "y": -0.20})
    return fig


def build_vocab_figure(params: ScheduleParams, compare_vocab_sizes: list[int], log_scale: bool) -> go.Figure:
    left_title = "tau(t) by vocab size" if params.interpolant_type == "flm_linear" else "tau(gamma) by vocab size"
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(left_title, "loss weight(tau) by vocab size"),
        horizontal_spacing=0.10,
    )

    for vocab_size in compare_vocab_sizes:
        compare_params = ScheduleParams(
            interpolant_type=params.interpolant_type,
            tau_min=params.tau_min,
            tau_max=params.tau_max,
            gamma_min=params.gamma_min,
            gamma_max=params.gamma_max,
            vocab_size=vocab_size,
            lut_points=params.lut_points,
            plot_points=params.plot_points,
        )
        schedule = compute_schedule(compare_params)
        label = f"K={vocab_size}"
        fig.add_trace(go.Scatter(x=schedule["coord"], y=schedule["tau"], name=label), row=1, col=1)
        fig.add_trace(go.Scatter(x=schedule["tau"], y=schedule["loss_weight"], name=label), row=1, col=2)

    fig.update_xaxes(title_text=str(compute_schedule(params)["coord_name"]), row=1, col=1)
    fig.update_yaxes(title_text="tau", row=1, col=1)
    fig.update_xaxes(title_text="tau", row=1, col=2)
    fig.update_yaxes(title_text="loss weight", row=1, col=2, type="log" if log_scale else "linear")
    fig.update_layout(height=460, legend={"orientation": "h", "y": -0.20})
    return fig


def hydra_override_snippet(params: ScheduleParams) -> str:
    return "\n".join(
        [
            f"algo.interpolant_type={params.interpolant_type}",
            f"algo.t_min={params.tau_min}",
            f"algo.t_max={params.tau_max}",
            f"algo.gamma_min={params.gamma_min}",
            f"algo.gamma_max={params.gamma_max}",
        ]
    )


def main() -> None:
    defaults = load_algo_defaults()
    st.set_page_config(page_title="FLM VDM Schedule Explorer", layout="wide")
    st.title("FLM VDM Schedule Explorer")
    st.caption(
        "Explore the legacy FLM tau<->t warp and the VP-native FLMVDM tau<->gamma warp "
        "with the same decoding-progress definition from Equation 25. This combined app now "
        "also includes the former standalone tau/error comparison views."
    )

    with st.sidebar:
        st.header("Controls")
        st.caption(f"Defaults loaded from `{CONFIG_PATH.relative_to(REPO_ROOT)}`.")
        interpolant_type = st.selectbox(
            "interpolant_type",
            options=["vp_vdm", "flm_linear"],
            index=0 if defaults["interpolant_type"] == "vp_vdm" else 1,
        )
        tau_min = st.number_input(
            "t_min",
            min_value=0.0,
            max_value=1.0,
            value=float(defaults["t_min"]),
            step=1e-4,
            format="%.6f",
            help="For FLMVDM this is really the lower tau endpoint, kept under the legacy t_min name.",
        )
        tau_max = st.number_input(
            "t_max",
            min_value=0.0,
            max_value=1.0,
            value=float(defaults["t_max"]),
            step=1e-4,
            format="%.6f",
            help="For FLMVDM this is really the upper tau endpoint, so values like 0.9990 or 0.9999 are meaningful.",
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
            tau_error_n_gh = st.slider(
                "Tau/error Gauss-Hermite nodes", min_value=20, max_value=200, value=100, step=10
            )
            flm_plot_t_max = st.number_input(
                "FLM t_max for tau/error plots",
                min_value=0.0,
                max_value=0.999999,
                value=min(float(defaults["t_max"]), 0.999999),
                step=1e-4,
                format="%.6f",
                help="Use values like 0.999 or 0.9999 to inspect the FLM clean-end tail in the shared tau/error views.",
            )
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
    if not (0.0 <= flm_plot_t_max < 1.0):
        st.error("`FLM t_max for tau/error plots` must satisfy 0 <= t_max < 1.")
        st.stop()

    params = ScheduleParams(
        interpolant_type=str(interpolant_type),
        tau_min=float(tau_min),
        tau_max=float(tau_max),
        gamma_min=float(gamma_min),
        gamma_max=float(gamma_max),
        vocab_size=int(vocab_size),
        lut_points=int(lut_points),
        plot_points=int(plot_points),
    )
    schedule = compute_schedule(params)
    spacing = sample_spacing(params, num_steps=num_steps)
    compare_params = CompareParams(
        vocab_size=int(vocab_size),
        gamma_min=float(gamma_min),
        gamma_max=float(gamma_max),
        t_plot_max=float(flm_plot_t_max),
        plot_points=int(plot_points),
        n_gh=int(tau_error_n_gh),
        tau_sample_points=int(num_steps),
        cache_version=2,
    )
    compare_data = enrich_tau_error_comparison(
        compute_tau_error_comparison(compare_params), compare_params
    )

    col1, col2, col3, col4 = st.columns(4)
    if params.interpolant_type == "vp_vdm":
        col1.metric("Gamma range", f"{float(np.min(schedule['gamma'])):.4f} -> {float(np.max(schedule['gamma'])):.4f}")
    else:
        col1.metric("Physical t range", f"{float(np.min(schedule['coord'])):.4f} -> {float(np.max(schedule['coord'])):.4f}")
    col2.metric("SNR range", f"{float(np.min(schedule['snr'])):.3e} -> {float(np.max(schedule['snr'])):.3e}")
    col3.metric("Peak loss weight", f"{float(np.max(schedule['loss_weight'])):.3e}")
    col4.metric("Mean loss weight", f"{float(np.mean(schedule['loss_weight'])):.3e}")

    if params.interpolant_type == "vp_vdm":
        st.info(
            "VP-native mode: tau is sampled uniformly, the LUT inverts tau(gamma), and all corruption "
            "quantities are derived directly from gamma(tau)."
        )
    else:
        st.info(
            "Legacy FLM mode: tau behaves like standardized correct-decoding progress under linear "
            "Gaussian interpolation, and the explorer maps tau back to physical t."
        )

    warp_tab, noise_tab, compare_tab, notes_tab, vocab_tab, export_tab = st.tabs(
        [
            "Warp and Spacing",
            "Noise and Weight",
            "Tau/Error Comparison",
            "Notes",
            "Vocab Comparison",
            "Config Snippet",
        ]
    )

    with warp_tab:
        st.plotly_chart(build_warp_figure(schedule, spacing), use_container_width=True)
        st.caption(
            "The selected interpolant determines which inverse LUT is visualized: tau<->t for legacy FLM "
            "and tau<->gamma for VP-native FLMVDM. Diamond markers show the finite set of uniformly "
            "spaced tau samples mapped through the same runtime warp."
        )

    with noise_tab:
        st.plotly_chart(build_noise_figure(schedule, spacing, log_scale=log_scale), use_container_width=True)
        st.plotly_chart(
            build_weight_compare_figure(schedule, spacing, log_scale=log_scale),
            use_container_width=True,
        )
        st.caption(
            "The same uniform tau sample overlay is shown on the noise, coefficient, and weighting "
            "curves so you can see exactly where a finite-step schedule lands in each coordinate system."
        )
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
                str(schedule["derivative_name"]) + "_min": float(np.min(schedule["derivative"])),
                str(schedule["derivative_name"]) + "_max": float(np.max(schedule["derivative"])),
                "loss_weight_min": float(np.min(schedule["loss_weight"])),
                "loss_weight_max": float(np.max(schedule["loss_weight"])),
            }
        )

    with compare_tab:
        st.plotly_chart(build_tau_error_tau_figure(compare_data), use_container_width=True)
        st.caption(
            "This is the former tau/error explorer folded into the schedule explorer. It compares "
            "FLM tau(t) against the VP/VDM tau(SNR) and tau(gamma) views using the same vocabulary "
            "size and endpoint controls."
        )
        st.plotly_chart(build_tau_error_derivative_figure(compare_data), use_container_width=True)
        st.plotly_chart(build_tau_error_sampling_figure(compare_data), use_container_width=True)
        st.plotly_chart(build_tau_error_error_figure(compare_data), use_container_width=True)

    with notes_tab:
        st.caption(f"Rendered from `{NOTES_PATH.relative_to(REPO_ROOT)}`.")
        st.markdown(load_notes_markdown())

    with vocab_tab:
        compare_vocab_sizes = sorted(set(int(v) for v in compare_vocab_sizes + [int(vocab_size)]))
        st.plotly_chart(
            build_vocab_figure(params, compare_vocab_sizes=compare_vocab_sizes, log_scale=log_scale),
            use_container_width=True,
        )
        st.caption(
            "Larger vocabularies make the decoding-progress warp more concentrated, both for the legacy FLM "
            "path and for the VP-native tau<->gamma construction."
        )

    with export_tab:
        st.code(hydra_override_snippet(params), language="bash")
        st.write(
            {
                "interpolant_type": params.interpolant_type,
                "vocab_size": int(vocab_size),
                "tau_interval": [float(tau_min), float(tau_max)],
                "coord_interval": [float(np.min(schedule["coord"])), float(np.max(schedule["coord"]))],
            }
        )


if __name__ == "__main__":
    main()
