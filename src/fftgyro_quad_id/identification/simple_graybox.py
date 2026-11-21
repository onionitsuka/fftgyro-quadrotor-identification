"""Small, explicit gray-box model inspired by Sun, de Visser, and Chu (2019).

The paper uses dimensionless airspeed, body rates, rotor-speed control inputs,
and wrench coefficients.  This module keeps only the leading physical terms;
it deliberately does not perform polynomial expansion or stepwise selection.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

OUTPUTS = ("Cx", "Cy", "Cz", "Cl", "Cm", "Cn")
ROLE_ALIASES = {
    "T": "training",
    "D": "model_selection",
    "H": "development_holdout",
    "training": "training",
    "model_selection": "model_selection",
    "development_holdout": "development_holdout",
}

STATIC_TERMS = {
    "Cx": ("mu_x",),
    "Cy": ("mu_y",),
    "Cz": ("mu_h2", "inflow_z"),
    "Cl": ("up",),
    "Cm": ("uq",),
    "Cn": ("ur",),
}

DYNAMIC_TERMS = {
    "Cx": ("mu_x",),
    "Cy": ("mu_y",),
    "Cz": ("mu_h2", "inflow_z"),
    "Cl": ("up", "p_bar"),
    "Cm": ("uq", "q_bar"),
    "Cn": ("ur", "r_bar"),
}


@dataclass(frozen=True)
class LinearOutputModel:
    output: str
    terms: tuple[str, ...]
    coefficients: tuple[float, ...]
    design_rank: int
    condition_number: float

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return design_matrix(rows, self.terms) @ np.asarray(self.coefficients)


def derived_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return one row with the two composite inflow variables made explicit."""

    result = dict(row)
    result["mu_h2"] = float(row["mu_x"]) ** 2 + float(row["mu_y"]) ** 2
    result["inflow_z"] = float(row["nu_in"]) - float(row["mu_z"])
    return result


def design_matrix(rows: Sequence[Mapping[str, Any]], terms: Sequence[str]) -> np.ndarray:
    """Build ``[1, term_1, ...]`` without hidden feature generation."""

    prepared = [derived_row(row) for row in rows]
    columns = [np.ones(len(prepared))]
    columns.extend(np.asarray([float(row[name]) for row in prepared]) for name in terms)
    matrix = np.column_stack(columns)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("gray-box design contains non-finite values")
    return matrix


def canonical_role(row: Mapping[str, Any]) -> str | None:
    value = str(row.get("analysis_role", row.get("role", "")))
    return ROLE_ALIASES.get(value)


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    residual = observed - predicted
    tss = float(np.sum((observed - np.mean(observed)) ** 2))
    return {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": 1.0 - float(np.sum(residual**2)) / tss if tss > 0 else math.nan,
    }


def fit_output(
    rows: Sequence[Mapping[str, Any]], output: str, terms: tuple[str, ...], target_suffix: str
) -> LinearOutputModel:
    matrix = design_matrix(rows, terms)
    target = np.asarray([float(row[f"{output}_{target_suffix}"]) for row in rows])
    coefficients = np.linalg.lstsq(matrix, target, rcond=None)[0]
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    condition = (
        float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else math.inf
    )
    return LinearOutputModel(
        output=output,
        terms=terms,
        coefficients=tuple(float(value) for value in coefficients),
        design_rank=int(np.linalg.matrix_rank(matrix)),
        condition_number=condition,
    )


def identify_simple_graybox(
    rows: Sequence[Mapping[str, Any]], *, target_suffix: str, dynamic: bool = False
) -> dict[str, Any]:
    """Fit fixed model terms on T and report D/H without selecting on H."""

    prepared = [dict(row) for row in rows]
    ignored_roles = Counter(
        str(row.get("analysis_role", row.get("role", "")))
        for row in prepared
        if canonical_role(row) is None
    )
    grouped = {
        role: [row for row in prepared if canonical_role(row) == role]
        for role in ("training", "model_selection", "development_holdout")
    }
    if len(grouped["training"]) < 8:
        raise ValueError("simple gray-box identification requires at least eight T rows")
    terms_by_output = DYNAMIC_TERMS if dynamic else STATIC_TERMS
    result: dict[str, Any] = {
        "method": "fixed leading-term Sun-style dimensionless gray-box",
        "dynamic": dynamic,
        "target_suffix": target_suffix,
        "selection_policy": "model structure fixed before fitting; coefficients use T only",
        "ignored_role_counts": dict(ignored_roles),
        "models": {},
    }
    for output in OUTPUTS:
        model = fit_output(grouped["training"], output, terms_by_output[output], target_suffix)
        metrics = {}
        for role, role_rows in grouped.items():
            if not role_rows:
                continue
            observed = np.asarray([float(row[f"{output}_{target_suffix}"]) for row in role_rows])
            metrics[role] = regression_metrics(observed, model.predict(role_rows))
        result["models"][output] = {
            "identification_status": (
                "identified"
                if model.design_rank == len(model.terms) + 1
                else "not_identifiable_rank_deficient"
            ),
            "feature_names": ["1", *model.terms],
            "coefficients": dict(zip(["1", *model.terms], model.coefficients, strict=True)),
            "design_rank": model.design_rank,
            "condition_number": model.condition_number,
            "metrics": metrics,
        }
    return result
