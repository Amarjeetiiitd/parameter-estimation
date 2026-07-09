"""
plotting.py
===========

All figure-generation routines. Each function saves a single PNG to the
given output directory and returns the saved path, so the CLI can log and
list every artifact produced.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / reproducible rendering
import matplotlib.pyplot as plt
import numpy as np

from . import model
from .optimize import FitResult, OptimizationHistory

logger = logging.getLogger(__name__)


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_path = out_dir / name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", out_path)
    return out_path


def plot_raw_data(x: np.ndarray, y: np.ndarray, out_dir: Path) -> Path:
    """Scatter plot of the raw (x, y) dataset."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=8, alpha=0.6, color="tab:blue")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Raw Dataset (xy_data.csv)")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "01_raw_data.png")


def plot_fit_overlay(
    x: np.ndarray, y: np.ndarray, fit: FitResult, out_dir: Path
) -> Path:
    """Overlay the recovered continuous curve on top of the raw data points."""
    t_dense = np.linspace(model.T_MIN, model.T_MAX, 3000)
    params = model.CurveParams(fit.theta, fit.M, fit.X)
    cx, cy = model.forward(t_dense, params)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=10, alpha=0.5, color="tab:blue", label="Observed data")
    ax.plot(cx, cy, color="tab:red", linewidth=1.8, label="Recovered curve")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Recovered Fit  (θ={fit.theta_deg:.4f}°, M={fit.M:.6f}, X={fit.X:.4f})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "02_fit_overlay.png")


def plot_residuals(
    x: np.ndarray, y: np.ndarray, fit: FitResult, out_dir: Path
) -> Path:
    """Residual (v - v_hat) plotted against recovered t for every data point."""
    t_hat = model.recovered_t(x, y, fit.theta, fit.X)
    u, v = model.inverse_rotate(x, y, fit.theta, fit.X)
    v_hat = np.exp(fit.M * np.abs(u)) * np.sin(model.OMEGA * u)
    resid = v - v_hat

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(t_hat, resid, s=8, alpha=0.6, color="tab:green")
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Recovered t")
    ax.set_ylabel("Residual (v - v_hat)")
    ax.set_title("Residuals vs Recovered Parameter t")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "03_residuals_vs_t.png")


def plot_residual_histogram(
    x: np.ndarray, y: np.ndarray, fit: FitResult, out_dir: Path
) -> Path:
    """Histogram of residuals to visually assess noise structure/normality."""
    u, v = model.inverse_rotate(x, y, fit.theta, fit.X)
    v_hat = np.exp(fit.M * np.abs(u)) * np.sin(model.OMEGA * u)
    resid = v - v_hat

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(resid, bins=40, color="tab:purple", alpha=0.75, edgecolor="black")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Residuals")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "04_residual_histogram.png")


def plot_convergence(history: OptimizationHistory, out_dir: Path) -> Path:
    """Best-so-far objective value vs evaluation index for DE and local refinement."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    de_vals = np.array(history.de_history)
    de_best = np.minimum.accumulate(de_vals)
    axes[0].plot(de_best, color="tab:orange")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Differential Evolution function evaluation")
    axes[0].set_ylabel("Best SSE so far (log scale)")
    axes[0].set_title("Global Search Convergence")
    axes[0].grid(alpha=0.3)

    ls_vals = np.array(history.ls_history)
    ls_best = np.minimum.accumulate(ls_vals)
    axes[1].plot(ls_best, color="tab:red")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Local refinement function evaluation")
    axes[1].set_ylabel("Best SSE so far (log scale)")
    axes[1].set_title("Local Refinement Convergence")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, out_dir, "05_convergence.png")


def plot_multistart_spread(
    solutions: np.ndarray, out_dir: Path
) -> Path:
    """Scatter of (theta, M, X) recovered from each independent random restart."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    labels = [r"$\theta$ (rad)", "$M$", "$X$"]
    for i, ax in enumerate(axes):
        ax.plot(solutions[:, i], "o-", color="tab:blue")
        ax.set_title(f"Multi-start recovered {labels[i]}")
        ax.set_xlabel("Restart index")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir, "06_multistart_spread.png")
