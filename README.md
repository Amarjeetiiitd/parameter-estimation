# Parametric Curve Parameter Estimation

This project recovers the unknown parameters $\theta$ (rotation), $M$ (exponential envelope rate), and $X$ (horizontal translation) of the parametric curve:

$$
x(t) = t\cos(\theta) - e^{M|t|} \sin(0.3t) \sin(\theta) + X
$$
$$
y(t) = 42 + t\sin(\theta) + e^{M|t|} \sin(0.3t) \cos(\theta)
$$

for $t \in [6, 60]$, given only noisy/plain $(x, y)$ sample points in `data/xy_data.csv`, with bounds $0 < \theta < 50^\circ$, $-0.05 < M < 0.05$, $0 < X < 100$.

## Key Results
- **$\theta$ (Rotation)** = $0.5235983032\text{ rad} \approx 29.999973^\circ$ (True value: $30^\circ$)
- **$M$ (Envelope rate)** = $0.0299999969$ (True value: $0.03$)
- **$X$ (Translation)** = $54.9999982128$ (True value: $55$)
- **Residual RMSE** = $3.486 \times 10^{-6}$ (effectively exact recovery)
- **Chamfer RMSE** = $4.660 \times 10^{-3}$ (independent spatial cross-validation)

---

## Project Structure

The project has been consolidated into a single, interactive Jupyter Notebook for ease of submission, evaluation, and interview presentation:

* **[parameter.ipynb](file:///D:/parameter%20estimation/parameter.ipynb)** — The **primary master file** containing:
  * LaTeX mathematical derivations (coordinate inversion trick).
  * Data diagnostics and Exploratory Data Analysis (EDA).
  * Model equations and optimization routines.
  * Multi-start validation checks and confidence interval calculations.
  * Matplotlib inline figures.
  * An Interview Cheatsheet with target discussion points.
* **[requirements.txt](file:///D:/parameter%20estimation/requirements.txt)** — Python dependencies.
* **[report.md](file:///D:/parameter%20estimation/report.md)** — Fully detailed 17-part academic/technical report.
* **data/** — Folder containing the coordinate points `xy_data.csv`.
* **outputs/** — Output folder for figures and the final JSON parameter payload.

---

## Setup and Execution

Follow these steps to set up a virtual environment and run the notebook:

### 1. Create a Virtual Environment
Navigate to the root directory and create a virtual environment:
```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment
Activate the environment based on your operating system:
* **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **Windows (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **macOS/Linux (Bash/Zsh):**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies
Install the required packages, including Jupyter:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install jupyter
```

### 4. Run the Jupyter Notebook
Start the notebook server:
```bash
jupyter notebook
```
In the browser, open **[parameter.ipynb](file:///D:/parameter%20estimation/parameter.ipynb)** and select the **Python 3** (or **.venv**) kernel to execute the cells.
