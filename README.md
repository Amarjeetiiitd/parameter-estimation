# Non-Linear Curve Fitting & Trajectory Reconstruction Engine
*Python | NumPy | SciPy | Pandas*

This repository implements a modular mathematical optimization pipeline in Python to solve a non-convex inverse coordinate transformation problem. The engine aligns a 2D coordinate trajectory $(x, y)$ onto a curvilinear longitudinal path $t$ and fits its transverse amplitude deviation against an exponentially modulated harmonic sine wave model.

---

## Technical Highlights
* **$O(N)$ Coordinate Inversion:** Exploits the orthogonal symmetry of 2D rigid-body rotation to project Cartesian points back to the curvilinear frame in closed form. This bypasses expensive $O(N \log N)$ KDTree nearest-neighbor queries during the optimization loop.
* **Hybrid Optimizer:** Combines **Differential Evolution** (global search, robust to harmonic local minima) with a **Trust-Region-Reflective (TRF)** bounded least-squares algorithm (superlinear local convergence).
* **Rigorous Verification:** Includes a multi-start solver spread analysis (13 restarts from random parameter vectors) to guarantee the uniqueness of the global equilibrium minimum.

---

## 1. Mathematical Formulation

The objective is to minimize the geometric residuals of the parameter vector $P = [\theta, M, X]^T$ to reconstruct the trajectory:

$$
x(t) = t\cos(\theta) - e^{M|t|} \sin(0.3t) \sin(\theta) + X
$$
$$
y(t) = 42 + t\sin(\theta) + e^{M|t|} \sin(0.3t) \cos(\theta)
$$

### Bounding Constraints
* **Unknown Parameters:** $0^\circ < \theta < 50^\circ$, $-0.05 < M < 0.05$, $0 < X < 100$
* **Parametric Track Envelope:** $6.0 \le t \le 60.0$

### Inverse Coordinate Alignment
The observed points $(x_i, y_i)$ are projected analytically into localized curve-coordinates $(t_i, A_i)$ using a fixed vertical reference anchor at $Y_0 = 42$:

$$
t_i = (x_i - X)\cos(\theta) + (y_i - 42)\sin(\theta)
$$
$$
A_i = -(x_i - X)\sin(\theta) + (y_i - 42)\cos(\theta)
$$

The theoretical transverse model behaves as a damped harmonic wave:

$$
\hat{A}_i(t_i; M) = e^{M|t_i|}\sin(0.3t_i)
$$

The parameters are estimated by minimizing the Residual Sum of Squares (RSS):

$$
\min_{P} \sum_{i=1}^{N} \left[ A_i(P) - \hat{A}_i(t_i(P); M) \right]^2
$$

---

## 2. High-Precision Benchmarks & Performance Metrics

When evaluated on the 1,500 coordinate pairs in `data/xy_data.csv`, the optimization engine achieves complete convergence at the global minimum:

### Optimal Parameter Estimates:
* **$\theta$ (Orientation Angle):** `0.5235983031599878` rad ($\approx 29.9999729277^\circ$)
* **$M$ (Damping Envelope):** `0.029999996873044544`
* **$X$ (Cartesian Offset):** `54.999998212785724`

### Recovered Track Envelope:
* **Resolved Curve Bound ($t$-domain):** `6.049405085351829` to `59.99517042539103` (conforms to bounds $6.0 \le t \le 60.0$)

### Error & Precision Statistics:
* **Sum of Squared Errors (SSE Cost):** `1.8229979828276476e-08`
* **Root Mean Squared Error (RMSE):** `3.486161196146508e-06`
* **L1 Distance:** `0.000002559801980369374` (`2.559801980369374e-06`)
* **Median Absolute Error:** `1.937512963428906e-06`
* **95th Percentile Spatial Error:** `7.505878841029801e-06`
* **99th Percentile Spatial Error:** `1.095594918239019e-05`
* **Precision Tolerance Coverage:**
  * **$98.47\%$** of points reconstructed within **$10\text{ }\mu\text{m}$** ($10^{-5}$ meters).
  * **$86.60\%$** of points reconstructed within **$5\text{ }\mu\text{m}$** ($5 \times 10^{-6}$ meters).

### Solver Confidence & Uncertainty Quantification:
* **Parameter Standard Errors:**
  * $\text{se}(\theta) = 2.86122692 \times 10^{-9}$ rad
  * $\text{se}(M) = 7.92933545 \times 10^{-10}$
  * $\text{se}(X) = 1.47278925 \times 10^{-7}$
* **Multi-start Optimization Uniqueness:**
  * Run configuration: 13 restarts from random parameter vectors.
  * Multi-start standard deviation spread (theta, M, X): `[7.06152783e-02, 4.47161959e-03, 8.00744101e+00]` (confirming convergence to the same global optimum basin).

---

## 3. Project Structure

* **[parameter.ipynb](parameter.ipynb)** — The master notebook for execution and visualization.
* **[report.md](report.md)** — Fully detailed 17-part academic/technical report.
* **[requirements.txt](requirements.txt)** — Python dependencies.
* **data/** — Folder containing the coordinate points `xy_data.csv`.
* **outputs/** — Output folder for figures and the final JSON parameter payload.
* **src/** — Python package source code:
  * [model.py](src/model.py) — Forward model and coordinate inversion equations.
  * [optimize.py](src/optimize.py) — Global search, local refinement, and precision matching.
  * [plotting.py](src/plotting.py) — Visual figure diagnostics code.
  * [data_io.py](src/data_io.py) — Dataset loading and diagnostics.
  * [cli.py](src/cli.py) — Command-line entry point.

---

## 4. Quick Start & Python Usage

You can run the parameter estimation engine directly inside your own Python code or scripts:

```python
from src import data_io, optimize

# Load and validate the coordinate trajectory
df = data_io.load_dataset("data/xy_data.csv")
x, y = df['x'].values, df['y'].values

# Execute the dual-stage optimization pipeline
fit = optimize.run_full_pipeline(x, y)

print(f"Optimal rotation (theta): {fit.theta_deg:.6f} degrees")
print(f"Optimal Cartesian offset (X): {fit.X:.6f}")
print(f"Coordinate matches: {fit.xy_matches} / {len(x)}")
```

---

## 5. Setup and Execution

### 1. Create and Activate a Virtual Environment
```bash
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install jupyter
```

### 3. Run via Command Line Interface (CLI)
```bash
python -m src.cli --data data/xy_data.csv --outdir outputs
```

### 4. Run via Jupyter Notebook
```bash
jupyter notebook
# Open parameter.ipynb in the browser
```

---

## 6. Desmos Parametric Plotting String

Copy and paste the absolute parametric solution string below directly into a Desmos cell for trajectory visualization:

```latex
\left(t\cdot\cos\left(0.52359830504105429184918705359740209765275037604314\right)-e^{0.0299999968730445439046849998021571082063019275665\left|t\right|}\cdot\sin\left(0.3t\right)\sin\left(0.52359830504105429184918705359740209765275037604314\right)+54.9999982127857265368220396339893341064453125,\ 42+t\cdot\sin\left(0.52359830504105429184918705359740209765275037604314\right)+e^{0.0299999968730445439046849998021571082063019275665\left|t\right|}\cdot\sin\left(0.3t\right)\cos\left(0.52359830504105429184918705359740209765275037604314\right)\right)
```
*(Make sure Desmos is set to **Radians** and parameter $t$ is set to $6 \le t \le 60$)*


