#!/usr/bin/env python3
"""Interactive explorer for FLMVDM gamma schedules.

Run from the repo root with:

    streamlit run scripts/flm_vdm_schedule_explorer.py

This app is meant to make the mental model concrete:

    tau ~ Uniform(0, 1)
    gamma = gamma(tau)
    z_t = alpha(gamma) x + sigma(gamma) noise
    weight = exp(-gamma) / q(gamma)

It supports a pure truncated Gumbel schedule, a truncated Gumbel over fixed
gamma endpoints, and an optional uniform mixture floor for comparison.
"""

from __future__ import annotations

import csv
import io
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.flm_vdm_mixed_schedule_viz import (  # noqa: E402
    alpha_sigma_from_gamma,
    build_figure,
    gumbel_cdf,
    midpoint_samples_from_cdf,
    mixed_cdf,
    truncated_gumbel_cdf,
    truncated_gumbel_pdf,
)


def gumbel_icdf(p: np.ndarray | float, loc: float, scale: float) -> np.ndarray:
    p_arr = np.asarray(p, dtype=np.float64)
    p_arr = np.clip(p_arr, 1e-300, 1.0 - 1e-15)
    return loc - scale * np.log(-np.log(p_arr))


def csv_text(rows: dict[str, np.ndarray]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    keys = list(rows.keys())
    writer.writerow(keys)
    for values in zip(*(rows[k] for k in keys)):
        writer.writerow(values)
    return buffer.getvalue()


def interpolate_from_cdf(
    gamma_grid: np.ndarray,
    cdf_grid: np.ndarray,
    tau_grid: np.ndarray,
) -> np.ndarray:
    cdf_unique, unique_idx = np.unique(cdf_grid, return_index=True)
    gamma_unique = gamma_grid[unique_idx]
    return np.interp(tau_grid, cdf_unique, gamma_unique)


def proposal_mass_below(
    threshold: float,
    gamma_min: float,
    gamma_max: float,
    q_gumbel_cdf_at_threshold: float,
    mixture_floor: float,
) -> float:
    if threshold <= gamma_min:
        return 0.0
    if threshold >= gamma_max:
        return 1.0
    uniform_cdf = (threshold - gamma_min) / (gamma_max - gamma_min)
    return ((1.0 - mixture_floor) * q_gumbel_cdf_at_threshold
            + mixture_floor * uniform_cdf)


def snr_mass_below(
    threshold: float,
    gamma_min: float,
    gamma_max: float,
) -> float:
    total = math.exp(-gamma_min) - math.exp(-gamma_max)
    if total <= 0:
        return 0.0
    if threshold <= gamma_min:
        return 0.0
    if threshold >= gamma_max:
        return 1.0
    return (math.exp(-gamma_min) - math.exp(-threshold)) / total


def build_tau_figure(
    tau_grid: np.ndarray,
    gamma_tau: np.ndarray,
    alpha_tau: np.ndarray,
    sigma_tau: np.ndarray,
    weight_tau: np.ndarray,
    q_tau: np.ndarray,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "gamma(tau)",
            "Noising Coefficients Along tau",
            "Proposal Density Along tau",
            "ELBO Weight Along tau",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.1,
    )
    fig.add_trace(go.Scatter(x=tau_grid, y=gamma_tau, name="gamma(tau)",
                             line=dict(width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=tau_grid, y=alpha_tau, name="alpha",
                             line=dict(width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=tau_grid, y=sigma_tau, name="sigma",
                             line=dict(width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=tau_grid, y=q_tau, name="q(gamma(tau))",
                             line=dict(width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=tau_grid, y=weight_tau,
                             name="exp(-gamma) / q",
                             line=dict(width=3)), row=2, col=2)
    fig.update_xaxes(title_text="tau", row=1, col=1)
    fig.update_xaxes(title_text="tau", row=1, col=2)
    fig.update_xaxes(title_text="tau", row=2, col=1)
    fig.update_xaxes(title_text="tau", row=2, col=2)
    fig.update_yaxes(title_text="gamma", row=1, col=1)
    fig.update_yaxes(title_text="coefficient", row=1, col=2)
    fig.update_yaxes(title_text="density", row=2, col=1)
    fig.update_yaxes(title_text="weight", type="log", row=2, col=2)
    fig.update_layout(
        template="plotly_white",
        height=720,
        margin=dict(t=80, b=60, l=70, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1.0),
    )
    return fig


def build_sample_figure(
    tau_samples: np.ndarray,
    gamma_samples: np.ndarray,
    weight_samples: np.ndarray,
    alpha_samples: np.ndarray,
    sigma_samples: np.ndarray,
) -> go.Figure:
    order = np.argsort(tau_samples)
    tau_sorted = tau_samples[order]
    gamma_sorted = gamma_samples[order]
    weight_sorted = weight_samples[order]
    alpha_sorted = alpha_samples[order]
    sigma_sorted = sigma_samples[order]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Sampled gamma vs tau",
            "Sampled ELBO weights vs tau",
            "Sampled gamma histogram",
            "Sampled alpha/sigma vs tau",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.1,
    )
    fig.add_trace(go.Scatter(x=tau_sorted, y=gamma_sorted,
                             mode="markers", name="sampled gamma",
                             marker=dict(size=6, opacity=0.75)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=tau_sorted, y=weight_sorted,
                             mode="markers", name="sampled weight",
                             marker=dict(size=6, opacity=0.75)),
                  row=1, col=2)
    fig.add_trace(go.Histogram(x=gamma_samples, name="gamma histogram",
                               nbinsx=60), row=2, col=1)
    fig.add_trace(go.Scatter(x=tau_sorted, y=alpha_sorted,
                             mode="markers", name="sampled alpha",
                             marker=dict(size=5, opacity=0.7)),
                  row=2, col=2)
    fig.add_trace(go.Scatter(x=tau_sorted, y=sigma_sorted,
                             mode="markers", name="sampled sigma",
                             marker=dict(size=5, opacity=0.7)),
                  row=2, col=2)

    fig.update_xaxes(title_text="sampled tau", row=1, col=1)
    fig.update_xaxes(title_text="sampled tau", row=1, col=2)
    fig.update_xaxes(title_text="sampled gamma", row=2, col=1)
    fig.update_xaxes(title_text="sampled tau", row=2, col=2)
    fig.update_yaxes(title_text="gamma", row=1, col=1)
    fig.update_yaxes(title_text="weight", type="log", row=1, col=2)
    fig.update_yaxes(title_text="count", row=2, col=1)
    fig.update_yaxes(title_text="coefficient", row=2, col=2)
    fig.update_layout(
        template="plotly_white",
        height=720,
        margin=dict(t=80, b=60, l=70, r=30),
        bargap=0.05,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1.0),
    )
    return fig


def main() -> None:
    st.set_page_config(
        page_title="FLMVDM Schedule Explorer",
        layout="wide",
    )
    st.title("FLMVDM Schedule Explorer")

    with st.sidebar:
        st.header("Gumbel")
        loc = st.number_input("loc", value=4.723, step=0.05, format="%.6f")
        scale = st.number_input("scale", value=0.852, min_value=1e-6,
                                step=0.02, format="%.6f")

        support_mode = st.radio(
            "Support",
            ("Gumbel quantiles", "Fixed gamma endpoints"),
            index=0,
        )
        if support_mode == "Gumbel quantiles":
            lower_log10 = st.slider("log10 lower quantile", -300.0, -2.0,
                                    -5.0, step=1.0)
            upper_tail_log10 = st.slider(
                "log10 upper tail mass", -15.0, -2.0, -5.0, step=0.5)
            q_low = 10.0 ** lower_log10
            q_high = 1.0 - 10.0 ** upper_tail_log10
            gamma_min = float(gumbel_icdf(q_low, loc, scale))
            gamma_max = float(gumbel_icdf(q_high, loc, scale))
            st.caption(
                f"Support from quantiles: [{gamma_min:.6g}, "
                f"{gamma_max:.6g}]")
        else:
            gamma_min = st.number_input("gamma_min", value=-8.0, step=0.5,
                                        format="%.6f")
            gamma_max = st.number_input("gamma_max", value=15.0, step=0.5,
                                        format="%.6f")
            q_low = float(gumbel_cdf(np.asarray([gamma_min]), loc, scale)[0])
            q_high = float(gumbel_cdf(np.asarray([gamma_max]), loc, scale)[0])
            st.caption(
                f"Underlying Gumbel quantiles: [{q_low:.3e}, "
                f"{q_high:.12f}]")

        st.header("Proposal")
        mixture_floor = st.slider(
            "uniform mixture floor eps", 0.0, 0.5, 0.0, step=0.005,
            help="eps=0 is a pure truncated Gumbel. Larger eps adds uniform "
                 "coverage over the selected support.")
        latent_type = st.radio("latent type", ("vp", "linear"), index=0,
                               horizontal=True)
        points = st.slider("grid points", 512, 10000, 3000, step=256)
        samples = st.slider("midpoint samples", 16, 512, 128, step=16)
        st.header("Random samples")
        random_samples = st.slider("random tau samples", 16, 10000, 512,
                                   step=16)
        sample_seed = st.number_input("sample seed", value=0, step=1)

    if gamma_min >= gamma_max:
        st.error("gamma_min must be smaller than gamma_max.")
        return

    gamma = np.linspace(gamma_min, gamma_max, points, dtype=np.float64)
    width = gamma_max - gamma_min
    q_uniform = np.full_like(gamma, 1.0 / width)
    q_gumbel = truncated_gumbel_pdf(gamma, loc, scale, gamma_min, gamma_max)
    q_gumbel_cdf = truncated_gumbel_cdf(
        gamma, loc, scale, gamma_min, gamma_max)
    q_mix = (1.0 - mixture_floor) * q_gumbel + mixture_floor * q_uniform
    q_mix_cdf = mixed_cdf(
        gamma, gamma_min, gamma_max, q_gumbel_cdf, mixture_floor)
    elbo_weight = np.exp(-gamma) / np.maximum(q_mix, 1e-300)
    alpha, sigma = alpha_sigma_from_gamma(gamma, latent_type)
    snr = np.exp(-gamma)
    sample_u, gamma_samples_mix = midpoint_samples_from_cdf(
        gamma, q_mix_cdf, samples)
    _, gamma_samples_gumbel = midpoint_samples_from_cdf(
        gamma, q_gumbel_cdf, samples)

    tau_grid = np.linspace(0.0, 1.0, points, dtype=np.float64)
    gamma_tau = interpolate_from_cdf(gamma, q_mix_cdf, tau_grid)
    alpha_tau, sigma_tau = alpha_sigma_from_gamma(gamma_tau, latent_type)
    q_tau = np.interp(gamma_tau, gamma, q_mix)
    weight_tau = np.exp(-gamma_tau) / np.maximum(q_tau, 1e-300)

    rng = np.random.default_rng(int(sample_seed))
    tau_random = rng.random(random_samples, dtype=np.float64)
    gamma_random = interpolate_from_cdf(gamma, q_mix_cdf, tau_random)
    q_random = np.interp(gamma_random, gamma, q_mix)
    weight_random = np.exp(-gamma_random) / np.maximum(q_random, 1e-300)
    alpha_random, sigma_random = alpha_sigma_from_gamma(
        gamma_random, latent_type)

    snr_mass = math.exp(-gamma_min) - math.exp(-gamma_max)
    max_weight_grid = float(np.max(elbo_weight))
    median_gamma = float(np.median(gamma_samples_mix))
    st.subheader("Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("gamma support", f"[{gamma_min:.3g}, {gamma_max:.3g}]")
    col2.metric("SNR mass", f"{snr_mass:.4g}")
    col3.metric("sample median gamma", f"{median_gamma:.4g}")
    col4.metric("max grid weight", f"{max_weight_grid:.4g}")

    scol1, scol2, scol3, scol4 = st.columns(4)
    scol1.metric("random mean weight", f"{weight_random.mean():.4g}")
    scol2.metric("random median weight", f"{np.median(weight_random):.4g}")
    scol3.metric("random p99 weight",
                  f"{np.quantile(weight_random, 0.99):.4g}")
    scol4.metric("random max weight", f"{weight_random.max():.4g}")

    thresholds = [0.0, 2.641163255254888, loc]
    rows = []
    for threshold in thresholds:
        q_cdf_threshold = float(truncated_gumbel_cdf(
            np.asarray([threshold]), loc, scale, gamma_min, gamma_max)[0])
        rows.append({
            "threshold gamma": threshold,
            "proposal mass below": proposal_mass_below(
                threshold, gamma_min, gamma_max, q_cdf_threshold,
                mixture_floor),
            "ELBO/SNR mass below": snr_mass_below(
                threshold, gamma_min, gamma_max),
        })
    st.dataframe(rows, width="stretch", hide_index=True)

    st.subheader("Uniform tau -> gamma(tau)")
    tau_fig = build_tau_figure(
        tau_grid=tau_grid,
        gamma_tau=gamma_tau,
        alpha_tau=alpha_tau,
        sigma_tau=sigma_tau,
        weight_tau=weight_tau,
        q_tau=q_tau,
    )
    st.plotly_chart(tau_fig, width="stretch")

    st.subheader("Random Uniform tau Samples")
    sample_fig = build_sample_figure(
        tau_samples=tau_random,
        gamma_samples=gamma_random,
        weight_samples=weight_random,
        alpha_samples=alpha_random,
        sigma_samples=sigma_random,
    )
    st.plotly_chart(sample_fig, width="stretch")

    st.subheader("Gamma-Space View")
    fig_args = SimpleNamespace(
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        mixture_floor=mixture_floor,
        loc=loc,
        scale=scale,
        latent_type=latent_type,
    )
    gamma_fig = build_figure(
        gamma=gamma,
        q_uniform=q_uniform,
        q_gumbel=q_gumbel,
        q_mix=q_mix,
        q_gumbel_cdf=q_gumbel_cdf,
        q_mix_cdf=q_mix_cdf,
        elbo_weight=elbo_weight,
        alpha=alpha,
        sigma=sigma,
        snr=snr,
        gamma_samples_mix=gamma_samples_mix,
        gamma_samples_gumbel=gamma_samples_gumbel,
        sample_u=sample_u,
        args=fig_args,
    )
    st.plotly_chart(gamma_fig, width="stretch")

    rows_for_csv = {
        "gamma": gamma,
        "proposal": q_mix,
        "proposal_trunc_gumbel": q_gumbel,
        "proposal_uniform": q_uniform,
        "cdf": q_mix_cdf,
        "cdf_trunc_gumbel": q_gumbel_cdf,
        "elbo_weight": elbo_weight,
        "alpha": alpha,
        "sigma": sigma,
        "snr": snr,
        "nsr": np.exp(gamma),
    }
    sample_csv = {
        "tau": tau_random,
        "gamma": gamma_random,
        "proposal": q_random,
        "elbo_weight": weight_random,
        "alpha": alpha_random,
        "sigma": sigma_random,
        "snr": np.exp(-gamma_random),
        "nsr": np.exp(gamma_random),
    }
    st.download_button(
        "Download gamma grid CSV",
        data=csv_text(rows_for_csv),
        file_name="flm_vdm_schedule_grid.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download random samples CSV",
        data=csv_text(sample_csv),
        file_name="flm_vdm_schedule_samples.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
