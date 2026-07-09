"""
model.py
========

Forward model of the parametric curve and the closed-form inverse
transform that is the mathematical core of this solution.

Forward model
-------------
.. math::
    x(t) = t\\cos\\theta - e^{M|t|}\\sin(0.3t)\\sin\\theta + X

    y(t) = 42 + t\\sin\\theta + e^{M|t|}\\sin(0.3t)\\cos\\theta

Key structural observation
---------------------------
Define the *base curve* in an unrotated, untranslated frame:

.. math::
    u(t) = t, \\qquad v(t; M) = e^{M|t|}\\sin(0.3t)

Then the forward model is **exactly** a rigid-body transform (rotation by
:math:`\\theta` followed by translation by :math:`(X, 42)`) applied to the
point :math:`(u, v)`:

.. math::
    \\begin{pmatrix} x \\\\ y \\end{pmatrix}
    =
    \\begin{pmatrix} \\cos\\theta & -\\sin\\theta \\\\
                      \\sin\\theta & \\cos\\theta \\end{pmatrix}
    \\begin{pmatrix} u \\\\ v \\end{pmatrix}
    +
    \\begin{pmatrix} X \\\\ 42 \\end{pmatrix}

Because the 2x2 rotation matrix :math:`R(\\theta)` is **orthogonal**
(:math:`R^{-1} = R^{\\top}`), this transform is exactly invertible in closed
form for *any* candidate :math:`(\\theta, X)` -- no root finding, curve
sampling, or nearest-neighbour search is required to recover the
corresponding :math:`(u, v)`:

.. math::
    \\begin{pmatrix} u \\\\ v \\end{pmatrix}
    = R(\\theta)^{\\top}
    \\left[
    \\begin{pmatrix} x \\\\ y \\end{pmatrix} - \\begin{pmatrix} X \\\\ 42 \\end{pmatrix}
    \\right]

If :math:`(\\theta, X)` are the *true* generating parameters, then the
recovered :math:`u` equals the true latent parameter :math:`t` for that
sample **exactly** (up to data noise), and the recovered :math:`v` must
satisfy :math:`v = e^{M|u|}\\sin(0.3u)`. This lets us build a per-point
residual that is a function of the three unknowns *only*, with zero
additional latent variables -- avoiding the generic (and much more
expensive) curve-matching machinery (Chamfer/Hausdorff/KDTree) that would
otherwise be needed when correspondence between data points and ``t`` is
unknown. See ``report.md`` Part 8 for the full derivation and discussion of
when this shortcut is (and is not) applicable.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

T_MIN: float = 6.0
T_MAX: float = 60.0
THETA_MIN: float = 0.0
THETA_MAX: float = np.deg2rad(50.0)
M_MIN: float = -0.05
M_MAX: float = 0.05
X_MIN: float = 0.0
X_MAX: float = 100.0
Y_OFFSET: float = 42.0
OMEGA: float = 0.3  # angular frequency inside sin(0.3 t)


class CurveParams(NamedTuple):
    """Container for the three unknown curve parameters."""

    theta: float  # radians
    M: float
    X: float


def base_curve(t: np.ndarray, M: float) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the unrotated base curve ``(u, v) = (t, e^{M|t|} sin(0.3 t))``.

    Parameters
    ----------
    t : np.ndarray
        Parameter values.
    M : float
        Exponential growth/decay rate.

    Returns
    -------
    tuple of np.ndarray
        ``(u, v)`` arrays, same shape as ``t``.
    """
    u = np.asarray(t, dtype=np.float64)
    v = np.exp(M * np.abs(u)) * np.sin(OMEGA * u)
    return u, v


def forward(t: np.ndarray, params: CurveParams) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the full forward model ``x(t), y(t)`` for given parameters.

    Parameters
    ----------
    t : np.ndarray
        Parameter values (should lie within ``[T_MIN, T_MAX]``).
    params : CurveParams
        ``(theta, M, X)`` in radians / dimensionless / length units.

    Returns
    -------
    tuple of np.ndarray
        ``(x, y)`` arrays, same shape as ``t``.
    """
    theta, M, X = params
    u, v = base_curve(t, M)
    ct, st = np.cos(theta), np.sin(theta)
    x = u * ct - v * st + X
    y = Y_OFFSET + u * st + v * ct
    return x, y


def inverse_rotate(
    x: np.ndarray, y: np.ndarray, theta: float, X: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the closed-form inverse rigid transform to recover ``(u, v)``.

    This is the adjoint/transpose rotation applied to the translated data,
    exact for an orthogonal rotation matrix -- no optimization or search is
    involved in this step itself; only ``theta`` and ``X`` are needed
    (``M`` is not required to undo the rotation/translation).

    Parameters
    ----------
    x, y : np.ndarray
        Observed data coordinates.
    theta : float
        Candidate rotation angle (radians).
    X : float
        Candidate x-translation.

    Returns
    -------
    tuple of np.ndarray
        Candidate ``(u, v)`` = ``(t_hat, e^{M|t|} sin(0.3 t)_hat)``.
    """
    ct, st = np.cos(theta), np.sin(theta)
    xs = x - X
    ys = y - Y_OFFSET
    u = xs * ct + ys * st
    v = -xs * st + ys * ct
    return u, v


def residuals_closed_form(
    params: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Per-point residual vector using the closed-form rotation-inversion trick.

    For candidate parameters, recovers ``(u, v)`` via :func:`inverse_rotate`
    and compares ``v`` against the model prediction
    ``e^{M|u|} sin(0.3 u)``. This residual is used directly by
    ``scipy.optimize.least_squares`` (local refinement) and its sum of
    squares is used by ``scipy.optimize.differential_evolution`` (global
    search).

    Parameters
    ----------
    params : np.ndarray
        Array ``[theta, M, X]``.
    x, y : np.ndarray
        Observed data.

    Returns
    -------
    np.ndarray
        Residual vector, one entry per data point.
    """
    theta, M, X = params
    u, v = inverse_rotate(x, y, theta, X)
    v_hat = np.exp(M * np.abs(u)) * np.sin(OMEGA * u)
    return v - v_hat


def sse_closed_form(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Sum of squared residuals -- scalar objective for global optimizers."""
    r = residuals_closed_form(params, x, y)
    return float(np.sum(r**2))


def recovered_t(x: np.ndarray, y: np.ndarray, theta: float, X: float) -> np.ndarray:
    """Return the latent ``t`` recovered for each data point under given params."""
    u, _ = inverse_rotate(x, y, theta, X)
    return u
