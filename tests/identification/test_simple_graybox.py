from __future__ import annotations

import numpy as np

from fftgyro_quad_id.identification.simple_graybox import (
    DYNAMIC_TERMS,
    design_matrix,
    identify_simple_graybox,
)


def rows() -> list[dict[str, float | str]]:
    result = []
    for index in range(18):
        role = "T" if index < 10 else "D" if index < 14 else "H"
        row: dict[str, float | str] = {
            "role": role,
            "mu_x": -0.2 + 0.025 * index,
            "mu_y": -0.1 + 0.013 * index,
            "mu_z": -0.05 + 0.006 * index,
            "nu_in": 0.08 + 0.002 * index,
            "up": (-1) ** index * 0.02 * (index + 1),
            "uq": (-1) ** (index // 2) * 0.015 * (index + 1),
            "ur": -0.08 + 0.01 * index,
            "p_bar": -0.04 + 0.006 * index,
            "q_bar": 0.03 - 0.004 * index,
            "r_bar": -0.02 + 0.0004 * index**2,
        }
        for output, terms in DYNAMIC_TERMS.items():
            matrix = design_matrix([row], terms)[0]
            beta = np.arange(1, len(matrix) + 1, dtype=float)
            row[f"{output}_measured"] = float(matrix @ beta)
        result.append(row)
    return result


def test_dynamic_model_recovers_fixed_synthetic_structure() -> None:
    result = identify_simple_graybox(rows(), target_suffix="measured", dynamic=True)
    for output, terms in DYNAMIC_TERMS.items():
        model = result["models"][output]
        assert model["identification_status"] == "identified"
        assert model["feature_names"] == ["1", *terms]
        assert model["metrics"]["development_holdout"]["rmse"] < 1e-10


def test_static_design_has_only_declared_columns() -> None:
    matrix = design_matrix(rows()[:3], ("mu_x", "mu_h2", "inflow_z"))
    assert matrix.shape == (3, 4)
    np.testing.assert_allclose(matrix[:, 0], 1.0)


def test_non_analysis_role_is_reported_and_ignored() -> None:
    bad = rows()
    bad[-1]["role"] = "needs_review"
    result = identify_simple_graybox(bad, target_suffix="measured")
    assert result["ignored_role_counts"] == {"needs_review": 1}
    assert result["models"]["Cx"]["metrics"]["development_holdout"]["n"] == 3
