# Parametric Curve Parameter Estimation

Recovers the unknown parameters `theta`, `M`, `X` of the parametric curve

```
x(t) = t*cos(theta) - e^(M|t|) * sin(0.3t) * sin(theta) + X
y(t) = 42 + t*sin(theta) + e^(M|t|) * sin(0.3t) * cos(theta)
```
for `t in [6, 60]`, given only noisy/plain `(x, y)` sample points in
`xy_data.csv`, with bounds `0 < theta < 50 deg`, `-0.05 < M < 0.05`, `0 < X < 100`.

## Headline Result

```
theta = 0.5235983032 rad  = 29.999973 deg   (true value: 30 deg = pi/6)
M     = 0.0299999969                          (true value: 0.03)
X     = 54.9999982128                         (true value: 55)
```

Recovered with RMSE ≈ 3.5e-6 on the core model residual — effectively exact
recovery, confirmed by an independent KDTree/Chamfer cross-check
(RMSE ≈ 4.7e-3, dominated by the coarse curve-sampling resolution rather than
parameter error) and by 12/13 multi-start random restarts converging to the
identical optimum. See `report.md` for the full mathematical derivation and
methodology (17-part write-up as required by the assignment).

Desmos-ready parametric expression (for pasting into the calculator):
```
\left(t*\cos(0.523598)-e^{0.030000\left|t\right|}\cdot\sin(0.3t)\sin(0.523598)+54.999998,42+t*\sin(0.523598)+e^{0.030000\left|t\right|}\cdot\sin(0.3t)\cos(0.523598)\right)
```

## Folder Structure

```
project/
├── README.md                      <- this file
├── report.md                      <- full 17-part mathematical/technical write-up
├── requirements.txt
├── data/
│   └── xy_data.csv                <- input dataset
├── src/
│   ├── __init__.py
│   ├── data_io.py                 <- loading + EDA/diagnostics
│   ├── model.py                   <- forward model + closed-form inverse trick
│   ├── optimize.py                <- global (DE) + local (TRF/LM) + multistart + CIs
│   ├── plotting.py                <- all diagnostic/result figures
│   └── cli.py                     <- CLI entry point / pipeline orchestration
└── outputs/
    ├── recovered_parameters.json  <- final parameters, uncertainties, metadata
    └── figures/
        ├── 01_raw_data.png
        ├── 02_fit_overlay.png
        ├── 03_residuals_vs_t.png
        ├── 04_residual_histogram.png
        ├── 05_convergence.png
        └── 06_multistart_spread.png
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Execution

```bash
python -m src.cli --data data/xy_data.csv --outdir outputs --seed 42
```

Optional flags:
- `--data PATH` — path to the CSV (default `data/xy_data.csv`)
- `--outdir PATH` — output directory for figures/JSON (default `outputs`)
- `--seed INT` — RNG seed for reproducibility (default `42`)

## Expected Output

- Console printout of recovered `theta`/`M`/`X`, residual statistics,
  cross-validation RMSE, parameter standard errors, and multi-start spread.
- `outputs/recovered_parameters.json` — machine-readable results.
- Six PNG figures under `outputs/figures/` (raw data, fit overlay, residuals
  vs. recovered `t`, residual histogram, convergence curves, multi-start
  spread).

Total runtime on the provided 1500-point dataset: **well under 10 seconds**
on a single CPU core (no GPU/parallelism required).

## Method Summary

The forward model is recognized as a rigid 2D rotation (by `theta`) plus
translation (`X`, `42`) applied to the base curve `(t, e^(M|t|) sin(0.3t))`.
Because a rotation matrix is orthogonal, this transform is **exactly
invertible in closed form** for any candidate `(theta, X)` — so instead of
generic, expensive curve-matching (nearest-neighbour/Chamfer/Hausdorff), each
data point's residual is computed directly and cheaply. `theta`, `M`, `X` are
then estimated with Differential Evolution (global search) followed by
Trust-Region-Reflective least squares (local refinement to machine
precision), validated with multi-start restarts, Jacobian-based confidence
intervals, and an independent KDTree-based cross-check. Full justification
in `report.md`.
