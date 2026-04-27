#!/usr/bin/env python3
"""Visualize FLMVDM gamma proposals and noising coefficients.

Example:

    python scripts/flm_vdm_mixed_schedule_viz.py \
      --gamma-min -8 --gamma-max 15 \
      --mixture-floor 0.02 \
      --output outputs/viz/flm_vdm_mixed_schedule.html \
      --csv outputs/viz/flm_vdm_mixed_schedule.csv

The mixed proposal is

    q(gamma) = (1 - eps) * TruncGumbel(gamma; loc, scale, support)
             + eps * Uniform(gamma; support)

and the VP-ELBO importance weight plotted here is

    w(gamma) = exp(-gamma) / q(gamma).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def alpha_sigma_from_gamma(
    gamma: np.ndarray,
    latent_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    if latent_type == "vp":
        alpha = np.sqrt(sigmoid(-gamma))
        sigma = np.sqrt(sigmoid(gamma))
    elif latent_type == "linear":
        alpha = sigmoid(-0.5 * gamma)
        sigma = sigmoid(0.5 * gamma)
    else:
        raise ValueError(f"Unknown latent_type: {latent_type}")
    return alpha, sigma


def gumbel_cdf(gamma: np.ndarray, loc: float, scale: float) -> np.ndarray:
    z = (gamma - loc) / scale
    return np.exp(-np.exp(-z))


def gumbel_pdf(gamma: np.ndarray, loc: float, scale: float) -> np.ndarray:
    z = (gamma - loc) / scale
    return np.exp(-(z + np.exp(-z))) / scale


def truncated_gumbel_pdf(
    gamma: np.ndarray,
    loc: float,
    scale: float,
    gamma_min: float,
    gamma_max: float,
) -> np.ndarray:
    cdf_min = gumbel_cdf(np.asarray([gamma_min]), loc, scale)[0]
    cdf_max = gumbel_cdf(np.asarray([gamma_max]), loc, scale)[0]
    return gumbel_pdf(gamma, loc, scale) / max(cdf_max - cdf_min, 1e-300)


def truncated_gumbel_cdf(
    gamma: np.ndarray,
    loc: float,
    scale: float,
    gamma_min: float,
    gamma_max: float,
) -> np.ndarray:
    cdf_min = gumbel_cdf(np.asarray([gamma_min]), loc, scale)[0]
    cdf_max = gumbel_cdf(np.asarray([gamma_max]), loc, scale)[0]
    cdf = (gumbel_cdf(gamma, loc, scale) - cdf_min) / max(
        cdf_max - cdf_min, 1e-300)
    return np.clip(cdf, 0.0, 1.0)


def mixed_cdf(
    gamma: np.ndarray,
    gamma_min: float,
    gamma_max: float,
    q_gumbel_cdf: np.ndarray,
    mixture_floor: float,
) -> np.ndarray:
    uniform_cdf = (gamma - gamma_min) / max(gamma_max - gamma_min, 1e-300)
    return ((1.0 - mixture_floor) * q_gumbel_cdf
            + mixture_floor * np.clip(uniform_cdf, 0.0, 1.0))


def midpoint_samples_from_cdf(
    gamma_grid: np.ndarray,
    cdf_grid: np.ndarray,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    u = (np.arange(num_samples, dtype=np.float64) + 0.5) / num_samples
    cdf_unique, unique_idx = np.unique(cdf_grid, return_index=True)
    gamma_unique = gamma_grid[unique_idx]
    return u, np.interp(u, cdf_unique, gamma_unique)


def write_csv(path: Path, rows: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows.keys())
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for values in zip(*(rows[k] for k in keys)):
            writer.writerow(values)


def build_figure(
    gamma: np.ndarray,
    q_uniform: np.ndarray,
    q_gumbel: np.ndarray,
    q_mix: np.ndarray,
    q_gumbel_cdf: np.ndarray,
    q_mix_cdf: np.ndarray,
    elbo_weight: np.ndarray,
    alpha: np.ndarray,
    sigma: np.ndarray,
    snr: np.ndarray,
    gamma_samples_mix: np.ndarray,
    gamma_samples_gumbel: np.ndarray,
    sample_u: np.ndarray,
    args: argparse.Namespace,
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Proposal Density",
            "VP ELBO Weight",
            "Noising Coefficients",
            "SNR / NSR",
            "CDF",
            "Midpoint Quantile Samples",
        ),
        vertical_spacing=0.11,
        horizontal_spacing=0.09,
    )

    fig.add_trace(go.Scatter(x=gamma, y=q_mix, name="mixed q",
                             line=dict(width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=gamma, y=q_gumbel, name="trunc gumbel",
                             line=dict(dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=gamma, y=q_uniform, name="uniform floor",
                             line=dict(dash="dot")), row=1, col=1)

    fig.add_trace(go.Scatter(x=gamma, y=elbo_weight, name="exp(-gamma) / q",
                             line=dict(width=3)), row=1, col=2)

    fig.add_trace(go.Scatter(x=gamma, y=alpha, name="alpha",
                             line=dict(width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=gamma, y=sigma, name="sigma",
                             line=dict(width=3)), row=2, col=1)

    fig.add_trace(go.Scatter(x=gamma, y=snr, name="SNR = exp(-gamma)",
                             line=dict(width=3)), row=2, col=2)
    fig.add_trace(go.Scatter(x=gamma, y=np.exp(gamma), name="NSR = exp(gamma)",
                             line=dict(dash="dash")), row=2, col=2)

    fig.add_trace(go.Scatter(x=gamma, y=q_mix_cdf, name="mixed CDF",
                             line=dict(width=3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=gamma, y=q_gumbel_cdf, name="gumbel CDF",
                             line=dict(dash="dash")), row=3, col=1)

    fig.add_trace(go.Scatter(x=sample_u, y=gamma_samples_mix,
                             mode="lines+markers",
                             name="mixed samples"), row=3, col=2)
    fig.add_trace(go.Scatter(x=sample_u, y=gamma_samples_gumbel,
                             mode="lines+markers",
                             name="gumbel samples",
                             line=dict(dash="dash")), row=3, col=2)

    fig.update_xaxes(title_text="gamma = log NSR", row=1, col=1)
    fig.update_xaxes(title_text="gamma = log NSR", row=1, col=2)
    fig.update_xaxes(title_text="gamma = log NSR", row=2, col=1)
    fig.update_xaxes(title_text="gamma = log NSR", row=2, col=2)
    fig.update_xaxes(title_text="gamma = log NSR", row=3, col=1)
    fig.update_xaxes(title_text="u midpoint", row=3, col=2)

    fig.update_yaxes(title_text="density", row=1, col=1)
    fig.update_yaxes(title_text="weight", type="log", row=1, col=2)
    fig.update_yaxes(title_text="coefficient", row=2, col=1)
    fig.update_yaxes(title_text="value", type="log", row=2, col=2)
    fig.update_yaxes(title_text="CDF", row=3, col=1)
    fig.update_yaxes(title_text="gamma sample", row=3, col=2)

    snr_mass = math.exp(-args.gamma_min) - math.exp(-args.gamma_max)
    max_weight_bound = (
        (args.gamma_max - args.gamma_min) * math.exp(-args.gamma_min)
        / max(args.mixture_floor, 1e-300)
    )
    title = (
        f"FLMVDM mixed gamma proposal: eps={args.mixture_floor:g}, "
        f"loc={args.loc:g}, scale={args.scale:g}, "
        f"support=[{args.gamma_min:g}, {args.gamma_max:g}], "
        f"latent={args.latent_type}<br>"
        f"finite-support SNR mass = {snr_mass:.6g}; "
        f"uniform-floor weight bound = {max_weight_bound:.6g}"
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.985,
                   yanchor="top"),
        template="plotly_white",
        height=1120,
        margin=dict(t=135, b=135, l=80, r=40),
        legend=dict(orientation="h", yanchor="top", y=-0.08,
                    xanchor="center", x=0.5),
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma-min", type=float, default=-8.0)
    parser.add_argument("--gamma-max", type=float, default=15.0)
    parser.add_argument("--loc", type=float, default=4.723)
    parser.add_argument("--scale", type=float, default=0.852)
    parser.add_argument("--mixture-floor", type=float, default=0.02,
                        help="Uniform mixture probability eps.")
    parser.add_argument("--latent-type", choices=("vp", "linear"),
                        default="vp")
    parser.add_argument("--points", type=int, default=4000)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/viz/flm_vdm_mixed_schedule.html"))
    parser.add_argument("--csv", type=Path,
                        default=Path("outputs/viz/flm_vdm_mixed_schedule.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gamma_min >= args.gamma_max:
        raise ValueError("--gamma-min must be smaller than --gamma-max")
    if args.scale <= 0:
        raise ValueError("--scale must be positive")
    if not 0.0 <= args.mixture_floor <= 1.0:
        raise ValueError("--mixture-floor must be in [0, 1]")
    if args.points < 16:
        raise ValueError("--points must be at least 16")
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    gamma = np.linspace(args.gamma_min, args.gamma_max, args.points,
                        dtype=np.float64)
    width = args.gamma_max - args.gamma_min
    q_uniform = np.full_like(gamma, 1.0 / width)
    q_gumbel = truncated_gumbel_pdf(
        gamma, args.loc, args.scale, args.gamma_min, args.gamma_max)
    q_gumbel_cdf = truncated_gumbel_cdf(
        gamma, args.loc, args.scale, args.gamma_min, args.gamma_max)
    q_mix = (1.0 - args.mixture_floor) * q_gumbel + (
        args.mixture_floor * q_uniform)
    q_mix_cdf = mixed_cdf(
        gamma, args.gamma_min, args.gamma_max,
        q_gumbel_cdf, args.mixture_floor)
    elbo_weight = np.exp(-gamma) / np.maximum(q_mix, 1e-300)
    alpha, sigma = alpha_sigma_from_gamma(gamma, args.latent_type)
    snr = np.exp(-gamma)

    sample_u, gamma_samples_mix = midpoint_samples_from_cdf(
        gamma, q_mix_cdf, args.samples)
    _, gamma_samples_gumbel = midpoint_samples_from_cdf(
        gamma, q_gumbel_cdf, args.samples)

    fig = build_figure(
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
        args=args,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs="cdn")

    write_csv(args.csv, {
        "gamma": gamma,
        "proposal_mixed": q_mix,
        "proposal_trunc_gumbel": q_gumbel,
        "proposal_uniform": q_uniform,
        "cdf_mixed": q_mix_cdf,
        "cdf_trunc_gumbel": q_gumbel_cdf,
        "elbo_weight": elbo_weight,
        "alpha": alpha,
        "sigma": sigma,
        "snr": snr,
        "nsr": np.exp(gamma),
    })

    snr_mass = math.exp(-args.gamma_min) - math.exp(-args.gamma_max)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.csv}")
    print(f"finite_support_snr_mass={snr_mass:.12g}")
    print(f"elbo_weight_mean_under_q_should_equal={snr_mass:.12g}")
    print(f"mixed_gamma_sample_min={gamma_samples_mix.min():.6g}")
    print(f"mixed_gamma_sample_max={gamma_samples_mix.max():.6g}")


if __name__ == "__main__":
    main()
