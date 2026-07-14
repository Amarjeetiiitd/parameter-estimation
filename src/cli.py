"""
cli.py
======

Command-line entry point for the full parameter-estimation pipeline:

    python -m src.cli --data data/xy_data.csv --outdir outputs

Runs, in order: dataset loading & validation, EDA, global optimization
(Differential Evolution), local refinement (Trust-Region-Reflective least
squares), multi-start robustness checks, confidence-interval computation,
independent KDTree/Chamfer cross-validation, figure generation, and result
serialization to JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from . import data_io, model, plotting
from .optimize import run_full_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate theta, M, X for the parametric curve from xy_data.csv"
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/xy_data.csv"), help="Path to xy_data.csv"
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("outputs"), help="Directory for results/figures"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--loss", type=str, choices=["l1", "l2"], default="l2", help="Loss function to optimize: l1 or l2"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig_dir = args.outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load & validate dataset -------------------------------------
    df = data_io.load_dataset(args.data)
    report = data_io.analyze_dataset(df)
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    # ---- 2. Visualize raw dataset ----------------------------------------
    plotting.plot_raw_data(x, y, fig_dir)

    # ---- 3-6. Optimization, refinement, multi-start, CIs ------------------
    fit = run_full_pipeline(x, y, loss=args.loss)

    # ---- 7. Final plots ----------------------------------------------------
    plotting.plot_fit_overlay(x, y, fit, fig_dir)
    plotting.plot_residuals(x, y, fit, fig_dir)
    plotting.plot_residual_histogram(x, y, fit, fig_dir)
    plotting.plot_convergence(fit.history, fig_dir)

    # Re-run multistart just for the plot (cheap; pipeline already validated it)
    from .optimize import multistart_refine

    solutions, _ = multistart_refine(x, y, np.array([fit.theta, fit.M, fit.X]))
    plotting.plot_multistart_spread(solutions, fig_dir)

    # ---- 8. Result printing --------------------------------------------
    print("\n" + "=" * 60)
    print("RECOVERED PARAMETERS")
    print("=" * 60)
    print(f"theta   = {fit.theta:.18f} rad  ({fit.theta_deg:.12f} deg)")
    print(f"M       = {fit.M:.18f}")
    print(f"X       = {fit.X:.18f}")
    print(f"Final SSE cost        : {fit.cost * 2:.18e}")
    print(f"RMSE (closed-form)     : {fit.rmse:.18e}")
    print(f"R-squared (R2 score)   : {fit.r2:.18f}")
    print(f"Max |residual|         : {fit.max_abs_residual:.18e}")
    print(f"Chamfer/KDTree RMSE    : {fit.chamfer_rmse_validation:.18e} (cross-check)")
    print(f"Std errors (theta,M,X) : [{fit.param_std_errors[0]:.10e} {fit.param_std_errors[1]:.10e} {fit.param_std_errors[2]:.10e}]")
    print(f"L1 Distance (residual) : {fit.l1_residual:.18f}  ({fit.l1_residual:.18e})")
    print(f"Median Absolute Error  : {fit.median_error:.18e}")
    print(f"90th Percentile Error  : {fit.p90_error:.18e}")
    print(f"95th Percentile Error  : {fit.p95_error:.18e}")
    print(f"99th Percentile Error  : {fit.p99_error:.18e}")
    print(f"Coverage within 10um   : {fit.pct_within_1e5:.10f}%")
    print(f"Coverage within 5um    : {fit.pct_within_5e6:.10f}%")
    print(f"Multi-start runs       : {fit.n_multistart_runs}")
    print(f"Multi-start std dev    : [{fit.multistart_spread[0]:.10e} {fit.multistart_spread[1]:.10e} {fit.multistart_spread[2]:.10e}]")
    print(f"Wall time              : {fit.wall_time_seconds:.2f} s")
    print("=" * 60)

    # ---- 9/10. Save figures (already saved) & recovered parameters --------
    result_payload = {
        "loss_optimized": args.loss,
        "theta_rad": fit.theta,
        "theta_deg": fit.theta_deg,
        "M": fit.M,
        "X": fit.X,
        "sse_cost": fit.cost * 2,
        "rmse": fit.rmse,
        "r2_score": fit.r2,
        "max_abs_residual": fit.max_abs_residual,
        "l1_residual": fit.l1_residual,
        "chamfer_rmse_validation": fit.chamfer_rmse_validation,
        "param_std_errors": fit.param_std_errors.tolist(),
        "covariance_matrix": fit.covariance.tolist(),
        "n_multistart_runs": fit.n_multistart_runs,
        "multistart_std": fit.multistart_spread.tolist(),
        "wall_time_seconds": fit.wall_time_seconds,
        "precision_coverage": {
            "median_error": fit.median_error,
            "p90_error": fit.p90_error,
            "p95_error": fit.p95_error,
            "p99_error": fit.p99_error,
            "pct_within_10um": fit.pct_within_1e5,
            "pct_within_5um": fit.pct_within_5e6,
        },
        "dataset_report": {
            "n_samples": report.n_samples,
            "n_missing": report.n_missing,
            "n_duplicates": report.n_duplicates,
            "x_range": report.x_range,
            "y_range": report.y_range,
        },
        "desmos_latex": (
            f"\\left(t*\\cos({fit.theta:.6f})-e^{{{fit.M:.6f}\\left|t\\right|}}"
            f"\\cdot\\sin(0.3t)\\sin({fit.theta:.6f})+{fit.X:.6f},"
            f"42+t*\\sin({fit.theta:.6f})+e^{{{fit.M:.6f}\\left|t\\right|}}"
            f"\\cdot\\sin(0.3t)\\cos({fit.theta:.6f})\\right)"
        ),
    }
    result_path = args.outdir / "recovered_parameters.json"
    with open(result_path, "w") as f:
        json.dump(result_payload, f, indent=2)
    logger.info("Saved recovered parameters to %s", result_path)
    print(f"\nSaved: {result_path}")
    print(f"Figures saved under: {fig_dir}")


if __name__ == "__main__":
    main()
