"""Sun-style static gray-box identification for the FFTGYRO campaign.

Sun et al. identify dimensionless aerodynamic coefficients from free-flight
airspeed.  A fixed wind-tunnel test has the same independent variable because
``V_air = V_ground - V_wind`` and ``V_ground = 0``.  This module implements
that specialization and keeps the original paper's axis-specific polynomial
candidate sets separate from an optional protected-physics variant.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


SUN_OUTPUTS = ("Cx", "Cy", "Cz", "Cl", "Cm", "Cn")
PHYSICAL_OUTPUTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


def rpm_acquisition_to_x500_motor_order(values: Sequence[float]) -> np.ndarray:
    """Reorder the campaign RPM acquisition channels into X500 motor order.

    Logged channels were wired as RPM1<-Motor2, RPM2<-Motor4,
    RPM3<-Motor3, RPM4<-Motor1.  ``sun_motor_inputs`` expects Motor1..Motor4.
    """

    rpm = np.asarray(values, dtype=float)
    if rpm.shape != (4,):
        raise ValueError(f"expected four RPM acquisition channels, got {rpm.shape}")
    return rpm[[3, 0, 2, 1]]


@dataclass(frozen=True)
class SunStaticConfig:
    """Geometry and normalization constants for the as-tested X500."""

    rotor_radius_m: float = 0.127
    reference_length_m: float = 0.1768
    rotor_count: int = 4
    air_density_kg_m3: float = 1.225
    max_stepwise_terms: int = 30
    f_out: float = 4.0
    pse_tolerance: float = 1.0e-6
    minimum_residual_df: int = 5


def fixed_test_airspeed(
    wind_speed_mps: float,
    alpha_rad: float,
    beta_fixture_rad: float,
) -> tuple[float, float, float, float]:
    """Return Sun FRD airspeed for the fixed rig and the aerodynamic beta.

    The campaign convention calls a clockwise apparatus rotation positive.
    The operator confirmed that positive fixture beta makes the wind hit the
    aircraft's left side.  Sun's body ``+y`` points right, so this is negative
    aerodynamic sideslip.  Positive alpha makes the wind hit the underside,
    which is positive ``w`` in Sun's forward-right-down frame.
    """

    beta_sun = -float(beta_fixture_rad)
    speed = float(wind_speed_mps)
    alpha = float(alpha_rad)
    horizontal = speed * math.cos(alpha)
    u = horizontal * math.cos(beta_sun)
    v = horizontal * math.sin(beta_sun)
    w = speed * math.sin(alpha)
    return u, v, w, beta_sun


def sun_motor_inputs(omega_rad_s: Sequence[float]) -> dict[str, float]:
    """Return Sun-normalized RPM and positive FRD moment-control inputs.

    X500 numbering is FR, RL, FL, RR.  Positive roll uses left minus right,
    positive pitch uses front minus aft.  Positive yaw uses Motors 1+2 minus
    Motors 3+4.  That Yaw sign is verified from the real-rig no-wind records:
    the opposite allocation had correlation -0.91 with positive Yaw angular
    acceleration in the high-acceleration samples.
    """

    omega = np.asarray(omega_rad_s, dtype=float)
    if omega.shape != (4,):
        raise ValueError(f"expected four rotor speeds, got {omega.shape}")
    if not np.all(np.isfinite(omega)) or np.any(omega < 0.0):
        raise ValueError("rotor speeds must be finite nonnegative magnitudes in rad/s")
    omega_bar = float(math.sqrt(float(np.mean(omega**2))))
    if omega_bar <= 0.0:
        raise ValueError("Sun normalization requires non-zero rotor speed")
    normalized = omega / omega_bar
    w1, w2, w3, w4 = normalized
    return {
        "omega_bar_rad_s": omega_bar,
        "omega_hat_1": float(w1),
        "omega_hat_2": float(w2),
        "omega_hat_3": float(w3),
        "omega_hat_4": float(w4),
        "up": float((w2**2 + w3**2) - (w1**2 + w4**2)),
        "uq": float((w1**2 + w3**2) - (w2**2 + w4**2)),
        "ur": float((w1**2 + w2**2) - (w3**2 + w4**2)),
    }


def wrench_scales(config: SunStaticConfig, omega_bar_rad_s: float) -> tuple[float, float]:
    disk_area = config.rotor_count * math.pi * config.rotor_radius_m**2
    force_scale = (
        config.air_density_kg_m3
        * disk_area
        * (config.rotor_radius_m * float(omega_bar_rad_s)) ** 2
    )
    return force_scale, config.reference_length_m * force_scale


def solve_normalized_induced_velocity(
    ct_hover: float,
    mu_x: float,
    mu_y: float,
    mu_z: float,
) -> float:
    """Solve Sun Eq. (34) by stable fixed-point iteration."""

    ct = max(float(ct_hover), 0.0)
    if ct == 0.0:
        return 0.0
    mu_h2 = float(mu_x) ** 2 + float(mu_y) ** 2
    value = math.sqrt(ct / 2.0)
    for _ in range(200):
        denominator = 2.0 * math.sqrt(mu_h2 + (-float(mu_z) + value) ** 2)
        updated = ct / max(denominator, 1.0e-12)
        relaxed = 0.5 * value + 0.5 * updated
        if abs(relaxed - value) <= 1.0e-12:
            return relaxed
        value = relaxed
    return value


def add_dimensionless_features(
    row: Mapping[str, Any],
    wrench_frd_cg: Sequence[float],
    config: SunStaticConfig,
    target_suffix: str,
) -> dict[str, Any]:
    """Copy one condition row and add Sun variables and coefficient targets."""

    result = dict(row)
    motor = sun_motor_inputs(
        [float(row[f"omega{index}_est_rad_s"]) for index in range(1, 5)]
    )
    alpha = float(row["alpha_rad"])
    beta_fixture = float(row["beta_rad"])
    u, v, w, beta_sun = fixed_test_airspeed(
        float(row["wind_speed_mps"]), alpha, beta_fixture
    )
    omega_bar = motor["omega_bar_rad_s"]
    denominator = omega_bar * config.rotor_radius_m
    force_scale, moment_scale = wrench_scales(config, omega_bar)
    wrench = np.asarray(wrench_frd_cg, dtype=float)
    if wrench.shape != (6,):
        raise ValueError(f"expected a six-axis wrench, got {wrench.shape}")
    result.update(motor)
    result.update(
        {
            "beta_fixture_rad": beta_fixture,
            "beta_sun_rad": beta_sun,
            "airspeed_u_mps": u,
            "airspeed_v_mps": v,
            "airspeed_w_mps": w,
            "mu_x": u / denominator,
            "mu_y": v / denominator,
            "mu_z": w / denominator,
            "abs_mu_x": abs(u / denominator),
            "abs_mu_y": abs(v / denominator),
            "abs_up": abs(motor["up"]),
            "abs_uq": abs(motor["uq"]),
            "abs_ur": abs(motor["ur"]),
            "force_scale_n": force_scale,
            "moment_scale_n_m": moment_scale,
        }
    )
    for index, (coefficient, physical) in enumerate(zip(SUN_OUTPUTS, PHYSICAL_OUTPUTS)):
        scale = force_scale if index < 3 else moment_scale
        result[f"{coefficient}_{target_suffix}"] = float(wrench[index] / scale)
        result[f"{physical}_{target_suffix}"] = float(wrench[index])
    return result


def _powers(variable_count: int, degree: int) -> list[tuple[int, ...]]:
    return [
        exponents
        for exponents in itertools.product(range(degree + 1), repeat=variable_count)
        if sum(exponents) <= degree
    ]


def _power_name(name: str, exponent: int) -> str:
    if exponent == 1:
        return name
    return f"{name}^{exponent}"


def _polynomial_terms(
    rows: Sequence[Mapping[str, Any]],
    variables: Sequence[str],
    degree: int,
) -> dict[str, np.ndarray]:
    values = {name: np.asarray([float(row[name]) for row in rows]) for name in variables}
    terms: dict[str, np.ndarray] = {}
    for exponents in _powers(len(variables), degree):
        parts = [
            _power_name(name, exponent)
            for name, exponent in zip(variables, exponents)
            if exponent
        ]
        name = "*".join(parts) if parts else "1"
        vector = np.ones(len(rows))
        for variable, exponent in zip(variables, exponents):
            if exponent:
                vector = vector * values[variable] ** exponent
        terms.setdefault(name, vector)
    return terms


def _multiply_term_sets(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for left_name, left_value in left.items():
        for right_name, right_value in right.items():
            parts = [name for name in (left_name, right_name) if name != "1"]
            name = "*".join(parts) if parts else "1"
            result.setdefault(name, left_value * right_value)
    return result


def sun_candidate_terms(
    rows: Sequence[Mapping[str, Any]], output: str
) -> tuple[dict[str, np.ndarray], list[str], dict[str, np.ndarray]]:
    """Return the exact static subset of Sun Eqs. (29), (30), (36), (39), (41), (43)."""

    if output == "Cx":
        candidates = _polynomial_terms(rows, ("mu_x", "abs_mu_y", "mu_z"), 3)
        priority = ["mu_x"]
        fixed: dict[str, np.ndarray] = {}
    elif output == "Cy":
        candidates = _polynomial_terms(rows, ("abs_mu_x", "mu_y", "mu_z"), 3)
        priority = ["mu_y"]
        fixed = {}
    elif output == "Cz":
        base = _polynomial_terms(rows, ("abs_mu_x", "abs_mu_y", "mu_z"), 4)
        multipliers = {
            "1": np.ones(len(rows)),
            "abs_up": np.asarray([float(row["abs_up"]) for row in rows]),
            "abs_uq": np.asarray([float(row["abs_uq"]) for row in rows]),
            "abs_ur": np.asarray([float(row["abs_ur"]) for row in rows]),
        }
        candidates = _multiply_term_sets(base, multipliers)
        priority = []
        fixed = {
            "mu_h^2": np.asarray(
                [float(row["mu_x"]) ** 2 + float(row["mu_y"]) ** 2 for row in rows]
            ),
            "nu_in-mu_z": np.asarray(
                [float(row["nu_in"]) - float(row["mu_z"]) for row in rows]
            ),
        }
    elif output == "Cm":
        longitudinal = _polynomial_terms(rows, ("mu_x", "mu_z"), 5)
        lateral = _polynomial_terms(rows, ("abs_mu_y",), 2)
        inputs = {
            "1": np.ones(len(rows)),
            "uq": np.asarray([float(row["uq"]) for row in rows]),
        }
        candidates = _multiply_term_sets(_multiply_term_sets(longitudinal, lateral), inputs)
        priority = []
        fixed = {}
    elif output == "Cl":
        lateral = _polynomial_terms(rows, ("mu_y", "mu_z"), 5)
        longitudinal = _polynomial_terms(rows, ("abs_mu_x",), 2)
        inputs = {
            "1": np.ones(len(rows)),
            "up": np.asarray([float(row["up"]) for row in rows]),
        }
        candidates = _multiply_term_sets(_multiply_term_sets(lateral, longitudinal), inputs)
        priority = []
        fixed = {}
    elif output == "Cn":
        state = _polynomial_terms(rows, ("mu_x", "mu_y", "mu_z"), 5)
        ur = np.asarray([float(row["ur"]) for row in rows])
        inputs = {
            "1": np.ones(len(rows)),
            "ur": ur,
            "ur*abs(ur)": ur * np.abs(ur),
            "ur^3": ur**3,
        }
        candidates = _multiply_term_sets(state, inputs)
        priority = []
        fixed = {}
    else:
        raise ValueError(f"unknown Sun output {output!r}")
    candidates.pop("1", None)
    for name in fixed:
        candidates.pop(name, None)
    return candidates, priority, fixed


def _design(
    fixed: Mapping[str, np.ndarray],
    candidates: Mapping[str, np.ndarray],
    selected: Sequence[str],
) -> np.ndarray:
    reference = next(iter(candidates.values()), None)
    if reference is None:
        reference = next(iter(fixed.values()), None)
    if reference is None:
        raise ValueError("design requires at least one fixed or candidate vector")
    columns = [np.ones(len(reference))]
    columns.extend(fixed.values())
    columns.extend(candidates[name] for name in selected)
    return np.column_stack(columns)


def _fit(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    return coefficients, residual, float(np.dot(residual, residual))


def forward_backward_stepwise(
    y: Sequence[float],
    candidates: Mapping[str, np.ndarray],
    *,
    fixed: Mapping[str, np.ndarray] | None = None,
    priority: Sequence[str] = (),
    config: SunStaticConfig | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Apply Sun Algorithm 1 with protected fixed regressors only.

    Candidate choice uses correlation between the current residual and each
    candidate residualized against the active design.  One backward F-test is
    performed after each addition, matching the paper.  The minimum-PSE state
    is returned when the next addition increases PSE.
    """

    cfg = config or SunStaticConfig()
    target = np.asarray(y, dtype=float)
    fixed_terms = dict(fixed or {})
    filtered: dict[str, np.ndarray] = {}
    for name, values in candidates.items():
        vector = np.asarray(values, dtype=float)
        if vector.shape != target.shape or not np.all(np.isfinite(vector)):
            continue
        if float(np.std(vector)) <= 1.0e-12:
            continue
        filtered[name] = vector
    candidates = filtered
    available = list(candidates)
    selected: list[str] = []
    history: list[dict[str, Any]] = []
    n = len(target)
    tss = float(np.sum((target - np.mean(target)) ** 2))
    baseline = _design(fixed_terms, candidates, selected)
    _, residual, sse = _fit(baseline, target)
    previous_pse = sse / n + (len(fixed_terms) / n**2) * tss
    best_pse = previous_pse
    best_selected: list[str] = []

    for step in range(cfg.max_stepwise_terms):
        if not available:
            break
        current_design = _design(fixed_terms, candidates, selected)
        _, current_residual, current_sse = _fit(current_design, target)
        trials: list[tuple[float, int, str]] = []
        priority_name = next((name for name in priority if name in available), None)
        names_to_test = [priority_name] if priority_name else available
        for name in names_to_test:
            assert name is not None
            vector = candidates[name]
            projection = current_design @ np.linalg.lstsq(current_design, vector, rcond=None)[0]
            residualized = vector - projection
            if float(np.std(residualized)) <= 1.0e-12:
                continue
            trial_design = np.column_stack((current_design, vector))
            if (
                n - trial_design.shape[1] < cfg.minimum_residual_df
                or np.linalg.matrix_rank(trial_design) < trial_design.shape[1]
            ):
                continue
            correlation = abs(float(np.corrcoef(residualized, current_residual)[0, 1]))
            if math.isfinite(correlation):
                trials.append((correlation, -available.index(name), name))
        if not trials:
            if priority_name:
                priority = [name for name in priority if name != priority_name]
                continue
            break
        _, _, added = max(trials)
        selected.append(added)
        available.remove(added)

        full_design = _design(fixed_terms, candidates, selected)
        _, _, full_sse = _fit(full_design, target)
        residual_df = n - full_design.shape[1]
        removed: str | None = None
        minimum_f = math.inf
        if residual_df > 0:
            variance = full_sse / residual_df
            f_values: list[tuple[float, str]] = []
            for name in selected:
                reduced = [item for item in selected if item != name]
                reduced_design = _design(fixed_terms, candidates, reduced)
                _, _, reduced_sse = _fit(reduced_design, target)
                f_value = (reduced_sse - full_sse) / variance if variance > 0.0 else math.inf
                f_values.append((float(f_value), name))
            minimum_f, least_significant = min(f_values, key=lambda item: item[0])
            if minimum_f < cfg.f_out:
                selected.remove(least_significant)
                available.append(least_significant)
                removed = least_significant

        fit_design = _design(fixed_terms, candidates, selected)
        _, _, fit_sse = _fit(fit_design, target)
        p = len(fixed_terms) + len(selected)
        pse = fit_sse / n + (p / n**2) * tss
        history.append(
            {
                "step": step + 1,
                "added": added,
                "removed": removed,
                "selected": list(selected),
                "fixed": list(fixed_terms),
                "minimum_f": minimum_f,
                "pse": pse,
                "previous_pse": previous_pse,
                "selection_correlation": max(item[0] for item in trials),
            }
        )
        if pse < best_pse:
            best_pse = pse
            best_selected = list(selected)
        if pse > previous_pse or pse <= cfg.pse_tolerance or removed == added:
            break
        previous_pse = pse
    return best_selected, history


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    residual = observed - predicted
    rmse = float(np.sqrt(np.mean(residual**2))) if len(observed) else math.nan
    tss = float(np.sum((observed - np.mean(observed)) ** 2)) if len(observed) else math.nan
    r2 = 1.0 - float(np.sum(residual**2)) / tss if len(observed) and tss > 0.0 else math.nan
    correlation = (
        float(np.corrcoef(observed, predicted)[0, 1])
        if len(observed) > 1 and np.std(observed) > 0.0 and np.std(predicted) > 0.0
        else math.nan
    )
    return {"n": len(observed), "rmse": rmse, "r2": r2, "correlation": correlation}


def identify_static_models(
    rows: Sequence[Mapping[str, Any]],
    target_suffix: str,
    config: SunStaticConfig | None = None,
) -> dict[str, Any]:
    """Identify benchmark, Sun-original, and protected-physics models."""

    cfg = config or SunStaticConfig()
    training = [row for row in rows if row.get("analysis_role") == "training"]
    if len(training) < 8:
        raise ValueError("Sun static identification requires at least eight training conditions")
    ct_hover_samples = [
        -float(row[f"Cz_{target_suffix}"])
        for row in training
        if abs(float(row["wind_speed_mps"])) <= 1.0e-12
        and abs(float(row["up"])) <= 1.0e-9
        and abs(float(row["uq"])) <= 1.0e-9
        and abs(float(row["ur"])) <= 1.0e-9
    ]
    ct_hover = float(np.median(ct_hover_samples)) if ct_hover_samples else 0.0
    augmented: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["nu_in"] = solve_normalized_induced_velocity(
            ct_hover, float(row["mu_x"]), float(row["mu_y"]), float(row["mu_z"])
        )
        augmented.append(row)
    training = [row for row in augmented if row.get("analysis_role") == "training"]
    result: dict[str, Any] = {
        "method": "Sun et al. axis-specific dimensionless candidates specialized to static fixed-rig data",
        "target_suffix": target_suffix,
        "ct_hover_median": ct_hover,
        "config": {
            "rotor_radius_m": cfg.rotor_radius_m,
            "reference_length_m": cfg.reference_length_m,
            "rotor_count": cfg.rotor_count,
            "air_density_kg_m3": cfg.air_density_kg_m3,
            "f_out": cfg.f_out,
            "pse_tolerance": cfg.pse_tolerance,
            "max_stepwise_terms": cfg.max_stepwise_terms,
            "minimum_residual_df": cfg.minimum_residual_df,
        },
        "models": {},
    }
    for output_index, output in enumerate(SUN_OUTPUTS):
        candidates_train, priority, sun_fixed_train = sun_candidate_terms(training, output)
        physical_fixed_names = {
            "Cx": ("mu_x",),
            "Cy": ("mu_y",),
            "Cz": ("mu_h^2", "nu_in-mu_z"),
            "Cl": ("up",),
            "Cm": ("uq",),
            "Cn": ("ur",),
        }[output]
        all_terms_train = dict(candidates_train)
        all_terms_train.update(sun_fixed_train)
        physics_fixed_requested = {
            name: all_terms_train[name]
            for name in physical_fixed_names
            if name in all_terms_train
        }
        physics_fixed_train = {
            name: values
            for name, values in physics_fixed_requested.items()
            if float(np.std(values)) > 1.0e-12
        }
        physics_unidentifiable = [
            name for name in physics_fixed_requested if name not in physics_fixed_train
        ]
        y_train = np.asarray([float(row[f"{output}_{target_suffix}"]) for row in training])
        sun_selected, sun_history = forward_backward_stepwise(
            y_train,
            candidates_train,
            fixed=sun_fixed_train,
            priority=priority,
            config=cfg,
        )
        physics_candidates_train = {
            name: values
            for name, values in candidates_train.items()
            if name not in physics_fixed_train
        }
        physics_selected, physics_history = forward_backward_stepwise(
            y_train,
            physics_candidates_train,
            fixed=physics_fixed_train,
            priority=(),
            config=cfg,
        )
        model_specs = {
            "benchmark": (physics_fixed_train, []),
            "sun_original_stepwise": (sun_fixed_train, sun_selected),
            "physics_forced_stepwise": (physics_fixed_train, physics_selected),
        }
        output_result: dict[str, Any] = {
            "candidate_count_training": len(candidates_train),
            "sun_priority": priority,
            "sun_fixed_regressors": list(sun_fixed_train),
            "physics_requested_regressors": list(physics_fixed_requested),
            "physics_fixed_regressors": list(physics_fixed_train),
            "physics_unidentifiable_regressors": physics_unidentifiable,
            "models": {},
        }
        for model_name, (fixed_train, selected) in model_specs.items():
            design_train = _design(fixed_train, candidates_train, selected)
            coefficients, _, _ = _fit(design_train, y_train)
            coefficient_names = ["1", *fixed_train, *selected]
            model_result: dict[str, Any] = {
                "feature_names": coefficient_names,
                "coefficients": {
                    name: float(value) for name, value in zip(coefficient_names, coefficients)
                },
                "design_rank": int(np.linalg.matrix_rank(design_train)),
                "selection_history": (
                    sun_history
                    if model_name == "sun_original_stepwise"
                    else physics_history if model_name == "physics_forced_stepwise" else []
                ),
                "metrics": {},
            }
            for role in ("training", "model_selection", "development_holdout"):
                role_rows = [row for row in augmented if row.get("analysis_role") == role]
                if not role_rows:
                    continue
                role_candidates, _, role_sun_fixed = sun_candidate_terms(role_rows, output)
                role_all = dict(role_candidates)
                role_all.update(role_sun_fixed)
                matrix = np.column_stack(
                    [
                        np.ones(len(role_rows)),
                        *[role_all[name] for name in coefficient_names[1:]],
                    ]
                )
                predicted_c = matrix @ coefficients
                observed_c = np.asarray(
                    [float(row[f"{output}_{target_suffix}"]) for row in role_rows]
                )
                scale_name = "force_scale_n" if output_index < 3 else "moment_scale_n_m"
                scales = np.asarray([float(row[scale_name]) for row in role_rows])
                model_result["metrics"][role] = {
                    "dimensionless": _metrics(observed_c, predicted_c),
                    "physical": _metrics(observed_c * scales, predicted_c * scales),
                }
            output_result["models"][model_name] = model_result
        result["models"][output] = output_result
    recommendations: dict[str, Any] = {}
    for output in SUN_OUTPUTS:
        output_result = result["models"][output]
        if output_result["physics_unidentifiable_regressors"]:
            recommendations[output] = {
                "status": "not_identifiable",
                "missing_required_regressors": output_result[
                    "physics_unidentifiable_regressors"
                ],
            }
            continue
        eligible = ("benchmark", "physics_forced_stepwise")
        selected_model = min(
            eligible,
            key=lambda name: output_result["models"][name]["metrics"]
            .get("model_selection", {})
            .get("physical", {})
            .get("rmse", math.inf),
        )
        recommendations[output] = {
            "status": "development_recommendation",
            "model": selected_model,
            "rule": "lowest model-selection physical RMSE among models retaining identifiable required physics",
        }
    result["recommended_model_by_output"] = recommendations
    result["rows"] = augmented
    return result
