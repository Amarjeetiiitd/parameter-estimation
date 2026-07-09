"""
optimize.py
===========

Optimization pipeline combining a global optimizer (Differential Evolution)
with local refinement (Levenberg-Marquardt / Trust-Region-Reflective via
``scipy.optimize.least_squares``), multi-start robustness checks, and
Jacobian-based confidence intervals.

Two objective functions are provided:

1. **Closed-form rotation-inversion residual** (:mod:`model`,
   ``residuals_closed_form``) -- exact, O(N) per evaluation, exploits the
   rigid-isometry structure of the forward model. This is the *primary*
   objective used for the production fit.
2. **Generic KDTree / Chamfer-distance objective** (this module,
   :func:`chamfer_sse`) -- does **not** assume any closed-form invertibility;
   it densely samples the candidate curve and measures the nearest-neighbour
   distance from every data point to that sampled curve using
   ``scipy.spatial.cKDTree``. This is slower (curve sampling + tree build
   per evaluation) but fully general, and is used here purely as an
   *independent cross-validation* of the closed-form result -- if a
   completely different algorithm agrees with the closed-form fit to high
   precision, that is strong evidence the recovered parameters are correct
   rather than an artifact of one particular objective's structure.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution, least_squares
from scipy.spatial import cKDTree

from . import model

logger = logging.getLogger(__name__)

BOUNDS = [
    (model.THETA_MIN, model.THETA_MAX),
    (model.M_MIN, model.M_MAX),
    (model.X_MIN, model.X_MAX),
]
LOWER = np.array([b[0] for b in BOUNDS])
UPPER = np.array([b[1] for b in BOUNDS])


@dataclass
class OptimizationHistory:
    """Records objective value at every evaluation for convergence plots."""

    de_history: List[float] = field(default_factory=list)
    ls_history: List[float] = field(default_factory=list)

    def record_de(self, val: float) -> None:
        self.de_history.append(val)

    def record_ls(self, val: float) -> None:
        self.ls_history.append(val)


@dataclass
class FitResult:
    """Final packaged output of the estimation pipeline."""

    theta: float
    M: float
    X: float
    theta_deg: float
    cost: float
    rmse: float
    max_abs_residual: float
    param_std_errors: np.ndarray
    covariance: np.ndarray
    n_multistart_runs: int
    multistart_spread: np.ndarray
    history: OptimizationHistory
    wall_time_seconds: float
    chamfer_rmse_validation: float


def _make_de_objective(
    x: np.ndarray, y: np.ndarray, history: OptimizationHistory
) -> Callable[[np.ndarray], float]:
    def objective(params: np.ndarray) -> float:
        val = model.sse_closed_form(params, x, y)
        history.record_de(val)
        return val

    return objective


def global_search(
    x: np.ndarray,
    y: np.ndarray,
    history: OptimizationHistory,
    seed: int = 42,
) -> OptimizeResult:
    """Run Differential Evolution over the full bounded parameter space.

    Differential Evolution is population-based and derivative-free, which
    makes it robust to the mild non-convexity introduced by the
    ``sin(0.3 t)`` oscillation and the ``|t|`` kink in the exponential --
    both of which can create shallow local minima that would trap a purely
    local, gradient-based method started from a poor initial guess.

    Parameters
    ----------
    x, y : np.ndarray
        Observed data.
    history : OptimizationHistory
        Mutable container that will be filled with the objective value at
        every candidate evaluation (used later for the "error vs iteration"
        diagnostic plot).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Result of the differential evolution run.
    """
    objective = _make_de_objective(x, y, history)
    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        seed=seed,
        strategy="best1bin",
        maxiter=500,
        popsize=25,
        tol=1e-14,
        mutation=(0.3, 1.7),
        recombination=0.9,
        polish=False,  # we do our own, more carefully-tuned local refinement
        workers=1,  # deterministic, avoids sandbox multiprocessing issues
        updating="deferred",
    )
    logger.info(
        "Differential Evolution finished: best SSE=%.6e at theta=%.6f, M=%.6f, X=%.6f "
        "after %d generations",
        result.fun,
        result.x[0],
        result.x[1],
        result.x[2],
        result.nit,
    )
    return result


def local_refine(
    x0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    history: OptimizationHistory | None = None,
) -> OptimizeResult:
    """Refine a global-search estimate with Trust-Region-Reflective least squares.

    ``scipy.optimize.least_squares`` with ``method='trf'`` is used (rather
    than unconstrained Levenberg-Marquardt) because it natively supports the
    box constraints on ``theta``, ``M``, and ``X`` given in the assignment,
    while still exhibiting the fast local (super-linear near the optimum)
    convergence characteristic of Gauss-Newton-type methods on a
    least-squares residual structure.

    Parameters
    ----------
    x0 : np.ndarray
        Initial guess ``[theta, M, X]`` (typically the Differential
        Evolution result).
    x, y : np.ndarray
        Observed data.
    history : OptimizationHistory, optional
        If given, the cost at each internal residual evaluation is recorded.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Result of the local refinement.
    """

    def residual_fn(params: np.ndarray) -> np.ndarray:
        r = model.residuals_closed_form(params, x, y)
        if history is not None:
            history.record_ls(float(np.sum(r**2)))
        return r

    result = least_squares(
        residual_fn,
        x0=x0,
        bounds=(LOWER, UPPER),
        method="trf",
        xtol=1e-15,
        ftol=1e-15,
        gtol=1e-15,
        max_nfev=20000,
    )
    logger.info(
        "Local refinement finished: cost=%.6e at theta=%.6f, M=%.6f, X=%.6f",
        result.cost,
        result.x[0],
        result.x[1],
        result.x[2],
    )
    return result


def multistart_refine(
    x: np.ndarray,
    y: np.ndarray,
    best_x0: np.ndarray,
    n_restarts: int = 12,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run local refinement from multiple random initial points as a robustness check.

    If the objective surface has multiple basins of attraction, independent
    restarts from random points across the full bounded domain (plus the
    global-search result itself) should converge to noticeably different
    optima. Tight clustering of all restart results is direct empirical
    evidence of a single, well-identified global optimum -- this is the
    practical, data-driven substitute for a formal identifiability proof.

    Parameters
    ----------
    x, y : np.ndarray
        Observed data.
    best_x0 : np.ndarray
        The best point found by global search (always included as one of
        the restarts).
    n_restarts : int
        Number of random restarts.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    tuple
        ``(all_solutions, best_solution)`` where ``all_solutions`` has shape
        ``(n_restarts + 1, 3)``.
    """
    rng = np.random.default_rng(seed)
    starts = [best_x0]
    for _ in range(n_restarts):
        starts.append(rng.uniform(LOWER, UPPER))

    solutions = []
    costs = []
    for x0 in starts:
        res = local_refine(np.asarray(x0), x, y)
        solutions.append(res.x)
        costs.append(res.cost)

    solutions = np.array(solutions)
    best_idx = int(np.argmin(costs))
    logger.info(
        "Multi-start refinement: %d restarts, best cost=%.6e, "
        "solution std dev (theta,M,X)=%s",
        len(starts),
        costs[best_idx],
        solutions.std(axis=0),
    )
    return solutions, solutions[best_idx]


def confidence_intervals(
    ls_result: OptimizeResult, n_points: int, n_params: int = 3, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute asymptotic parameter standard errors and covariance from the Jacobian.

    For a nonlinear least-squares fit :math:`\\hat\\beta` minimizing
    :math:`\\sum r_i(\\beta)^2`, the classical (Gauss-Newton / delta-method)
    approximation to the parameter covariance matrix is

    .. math::
        \\widehat{\\mathrm{Cov}}(\\hat\\beta) = \\hat\\sigma^2 (J^\\top J)^{-1}

    where :math:`J` is the Jacobian of the residual vector evaluated at
    :math:`\\hat\\beta`, and :math:`\\hat\\sigma^2 = \\mathrm{RSS}/(n-p)` is the
    residual-variance estimate (:math:`n` = number of data points, :math:`p`
    = number of free parameters). Standard errors are the square roots of
    the diagonal of this covariance matrix. This is a *local, asymptotic*
    approximation (valid when residuals are small and the model is
    well-approximated by its linearization near the optimum) -- exactly the
    regime we are in here given the sub-1e-4 residuals achieved.

    Parameters
    ----------
    ls_result : scipy.optimize.OptimizeResult
        Result object from ``least_squares`` (must expose ``.jac`` and
        ``.fun``).
    n_points : int
        Number of data points ``n``.
    n_params : int
        Number of free parameters ``p`` (default 3: theta, M, X).
    alpha : float
        Significance level for future interval construction (stored for
        reference; not used to alter the covariance itself).

    Returns
    -------
    tuple
        ``(std_errors, covariance)``.
    """
    J = ls_result.jac
    resid = ls_result.fun
    dof = max(n_points - n_params, 1)
    sigma2 = float(np.sum(resid**2)) / dof
    try:
        JTJ_inv = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        logger.warning("J^T J is singular; using pseudo-inverse for covariance.")
        JTJ_inv = np.linalg.pinv(J.T @ J)
    covariance = sigma2 * JTJ_inv
    std_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    return std_errors, covariance


def chamfer_sse(
    params: np.ndarray, x: np.ndarray, y: np.ndarray, n_curve_samples: int = 4000
) -> float:
    """Generic, correspondence-free objective: mean squared nearest-neighbour distance.

    This function makes **no** use of the closed-form rotation-inversion
    shortcut. It densely samples the candidate curve at ``n_curve_samples``
    uniformly spaced ``t`` values, builds a ``scipy.spatial.cKDTree`` over
    those samples, and for every observed data point finds the Euclidean
    distance to its nearest sampled curve point (a one-sided Chamfer
    distance from data to model curve). It is included as an
    algorithm-independent cross-check on the closed-form fit and as the
    fallback strategy that would be required if the forward model did not
    happen to be a rigid isometry (see ``report.md`` Part 7-8).

    Parameters
    ----------
    params : np.ndarray
        ``[theta, M, X]``.
    x, y : np.ndarray
        Observed data.
    n_curve_samples : int
        Number of uniformly spaced ``t`` samples used to approximate the
        continuous curve.

    Returns
    -------
    float
        Mean squared nearest-neighbour distance (data -> curve).
    """
    theta, M, X = params
    t_grid = np.linspace(model.T_MIN, model.T_MAX, n_curve_samples)
    cx, cy = model.forward(t_grid, model.CurveParams(theta, M, X))
    tree = cKDTree(np.column_stack([cx, cy]))
    data_pts = np.column_stack([x, y])
    dist, _ = tree.query(data_pts, k=1)
    return float(np.mean(dist**2))


def run_full_pipeline(x: np.ndarray, y: np.ndarray) -> FitResult:
    """Execute the complete global-then-local, multi-start optimization pipeline.

    Steps
    -----
    1. Global search via Differential Evolution over the closed-form
       objective (fast, exact, avoids curve sampling).
    2. Local refinement via Trust-Region-Reflective least squares, which
       converges to machine-precision residuals given the (near-)noiseless
       data and the excellent DE starting point.
    3. Multi-start robustness check: refine from several random points to
       confirm the optimum is unique (not one of several local minima).
    4. Jacobian-based confidence intervals on the final refined estimate.
    5. Independent cross-validation via the generic KDTree/Chamfer objective,
       evaluated (not re-optimized) at the final parameter estimate, to
       confirm agreement between two structurally different formulations.

    Parameters
    ----------
    x, y : np.ndarray
        Observed data coordinates.

    Returns
    -------
    FitResult
        Complete, packaged result of the estimation procedure.
    """
    t_start = time.time()
    history = OptimizationHistory()

    de_result = global_search(x, y, history)
    ls_result = local_refine(de_result.x, x, y, history)

    multistart_solutions, best_solution = multistart_refine(x, y, ls_result.x)
    # Use the best multi-start solution as the final estimate (should match
    # ls_result to high precision if the optimum is unique).
    final_ls = local_refine(best_solution, x, y, history)

    std_errors, covariance = confidence_intervals(final_ls, n_points=len(x))

    resid = final_ls.fun
    rmse = float(np.sqrt(np.mean(resid**2)))
    max_abs_resid = float(np.max(np.abs(resid)))

    chamfer_val = chamfer_sse(final_ls.x, x, y)
    chamfer_rmse = float(np.sqrt(chamfer_val))

    wall_time = time.time() - t_start

    theta_hat, M_hat, X_hat = final_ls.x
    result = FitResult(
        theta=float(theta_hat),
        M=float(M_hat),
        X=float(X_hat),
        theta_deg=float(np.rad2deg(theta_hat)),
        cost=float(final_ls.cost),
        rmse=rmse,
        max_abs_residual=max_abs_resid,
        param_std_errors=std_errors,
        covariance=covariance,
        n_multistart_runs=len(multistart_solutions),
        multistart_spread=multistart_solutions.std(axis=0),
        history=history,
        wall_time_seconds=wall_time,
        chamfer_rmse_validation=chamfer_rmse,
    )
    return result
