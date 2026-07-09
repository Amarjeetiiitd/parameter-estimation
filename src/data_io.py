"""
data_io.py
==========

Dataset loading, validation, and exploratory-data-analysis (EDA) utilities.

The assignment guarantees that ``xy_data.csv`` contains points that lie on the
parametric curve

.. math::
    x(t) = t\\cos\\theta - e^{M|t|}\\sin(0.3t)\\sin\\theta + X

    y(t) = 42 + t\\sin\\theta + e^{M|t|}\\sin(0.3t)\\cos\\theta,\\qquad t\\in[6,60]

but it does **not** state whether the data is noiseless, whether ``t`` is
sampled uniformly, whether duplicates exist, or whether outliers are present.
All of these must be checked empirically rather than assumed -- this module
performs exactly that verification and logs a structured report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetReport:
    """Structured summary of dataset diagnostics produced by :func:`analyze_dataset`."""

    n_samples: int
    n_missing: int
    n_duplicates: int
    x_range: tuple
    y_range: tuple
    x_mean_std: tuple
    y_mean_std: tuple
    outlier_indices: np.ndarray
    notes: Dict[str, Any] = field(default_factory=dict)

    def summary_text(self) -> str:
        lines = [
            "===== Dataset Diagnostic Report =====",
            f"Samples                : {self.n_samples}",
            f"Missing values          : {self.n_missing}",
            f"Duplicate rows          : {self.n_duplicates}",
            f"x range                 : [{self.x_range[0]:.6f}, {self.x_range[1]:.6f}]",
            f"y range                 : [{self.y_range[0]:.6f}, {self.y_range[1]:.6f}]",
            f"x mean/std              : {self.x_mean_std[0]:.6f} / {self.x_mean_std[1]:.6f}",
            f"y mean/std              : {self.y_mean_std[0]:.6f} / {self.y_mean_std[1]:.6f}",
            f"Outliers (IQR rule)     : {len(self.outlier_indices)}",
        ]
        for key, val in self.notes.items():
            lines.append(f"{key:24s}: {val}")
        return "\n".join(lines)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the (x, y) dataset from a CSV file.

    Parameters
    ----------
    path : Path
        Path to a CSV file with columns ``x`` and ``y``.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe with float64 columns ``x`` and ``y``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the required columns are missing or the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    required = {"x", "y"}
    if not required.issubset(set(df.columns.str.lower())):
        # Handle possible casing/whitespace differences defensively.
        df.columns = [c.strip().lower() for c in df.columns]
        if not required.issubset(set(df.columns)):
            raise ValueError(
                f"Dataset must contain columns {required}, found {list(df.columns)}"
            )
    if df.empty:
        raise ValueError("Dataset is empty.")

    df = df[["x", "y"]].astype(np.float64)
    logger.info("Loaded dataset with %d rows from %s", len(df), path)
    return df


def analyze_dataset(df: pd.DataFrame) -> DatasetReport:
    """Compute EDA diagnostics required to justify downstream modelling choices.

    We explicitly check (rather than assume):

    * Missing values -- would require imputation or row removal.
    * Exact duplicate rows -- would bias a least-squares objective by giving
      repeated points extra implicit weight.
    * Outliers via the 1.5*IQR rule on both coordinates -- points that do not
      lie near the manifold would indicate measurement noise or corruption.
    * Basic range/scale statistics used to set sane optimizer bounds/initial
      guesses and sanity-check recovered parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with columns ``x`` and ``y``.

    Returns
    -------
    DatasetReport
        Structured diagnostic report.
    """
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    n_missing = int(df.isna().sum().sum())
    n_duplicates = int(df.duplicated().sum())

    def iqr_outliers(arr: np.ndarray) -> np.ndarray:
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return np.where((arr < lower) | (arr > upper))[0]

    out_x = iqr_outliers(x)
    out_y = iqr_outliers(y)
    outlier_idx = np.union1d(out_x, out_y)

    notes: Dict[str, Any] = {}

    report = DatasetReport(
        n_samples=len(df),
        n_missing=n_missing,
        n_duplicates=n_duplicates,
        x_range=(float(x.min()), float(x.max())),
        y_range=(float(y.min()), float(y.max())),
        x_mean_std=(float(x.mean()), float(x.std())),
        y_mean_std=(float(y.mean()), float(y.std())),
        outlier_indices=outlier_idx,
        notes=notes,
    )
    logger.info("\n%s", report.summary_text())
    return report
