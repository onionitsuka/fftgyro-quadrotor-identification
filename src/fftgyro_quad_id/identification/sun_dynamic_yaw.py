"""Sun-style dynamic Yaw moment identification for the fixed FFTGYRO rig."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from fftgyro_quad_id.identification.sun_dynamic import (
    ROLE_ALIASES,
    fixed_rig_airspeed_frd,
    sun_angles,
)
from fftgyro_quad_id.identification.sun_static import (
    SunStaticConfig,
    _design,
    _fit,
    _metrics,
    _multiply_term_sets,
    _polynomial_terms,
    forward_backward_stepwise,
    sun_motor_inputs,
    wrench_scales,
)


def dynamic_yaw_sample(
    *,
    rate_rad_s: float,
    acceleration_rad_s2: float,
    rotor_spin_acceleration_rad_s2: float,
    wind_speed_m_s: float,
    pitch_rad: float,
    heading_rad: float,
    roll_rad: float,
    rpm_motor_order: Sequence[float],
    support_moment_yaw_Nm: float,
    day: str,
    config: SunStaticConfig | None = None,
) -> dict[str, float | str]:
    """Build one Yaw moment row with free body and rotor inertia regressors.

    For pure Yaw, positive ``dOmega_reaction`` follows the airframe reaction-
    torque direction, so ``Cn = Izz*rdot/QM - Ip*dOmega_reaction/QM -
    Msupport/QM``.
    The regression target is the measured ``-Msupport/QM`` term; therefore
    the two inertia columns carry a minus sign and their fitted coefficients
    have inertia units.
    """

    cfg = config or SunStaticConfig()
    rotor = sun_motor_inputs(
        np.asarray(rpm_motor_order, dtype=float) * (2.0 * math.pi / 60.0)
    )
    omega_bar = float(rotor["omega_bar_rad_s"])
    _, moment_scale = wrench_scales(cfg, omega_bar)
    u, v, w = fixed_rig_airspeed_frd(
        wind_speed_m_s, pitch_rad, heading_rad, roll_rad
    )
    alpha, beta = sun_angles(u, v, w)
    denominator = omega_bar * cfg.rotor_radius_m
    return {
        "day": str(day),
        "alpha_rad": alpha,
        "beta_rad": beta,
        "mu_x": float(u / denominator),
        "mu_y": float(v / denominator),
        "mu_z": float(w / denominator),
        "mu": float(math.sqrt(u * u + v * v + w * w) / denominator),
        "r_bar": float(rate_rad_s * cfg.reference_length_m / denominator),
        "force_scale_N": wrench_scales(cfg, omega_bar)[0],
        "moment_scale_Nm": moment_scale,
        "inverse_moment_scale_per_Nm": 1.0 / moment_scale,
        "minus_acceleration_over_moment_scale": -float(acceleration_rad_s2)
        / moment_scale,
        "rotor_reaction_acceleration_over_moment_scale": float(
            rotor_spin_acceleration_rad_s2
        )
        / moment_scale,
        "Cn_observed": -float(support_moment_yaw_Nm) / moment_scale,
        **rotor,
    }


def dynamic_yaw_candidate_terms(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Return the Sun Cn state/input family augmented by dimensionless Yaw rate."""

    state = _polynomial_terms(rows, ("mu_x", "mu_y", "mu_z"), 5)
    ur = np.asarray([float(row["ur"]) for row in rows])
    dynamic = {
        "1": np.ones(len(rows)),
        "r_bar": np.asarray([float(row["r_bar"]) for row in rows]),
        "ur": ur,
        "ur*abs(ur)": ur * np.abs(ur),
        "ur^3": ur**3,
    }
    candidates = _multiply_term_sets(state, dynamic)
    candidates.pop("1", None)
    return candidates


def sun_eq43_cn_candidate_terms(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Return Sun Eq. (43): P5(mu_x,mu_y,mu_z) P3(r_bar) P3(ur)."""

    state = _polynomial_terms(rows, ("mu_x", "mu_y", "mu_z"), 5)
    yaw_rate = _polynomial_terms(rows, ("r_bar",), 3)
    yaw_input = _polynomial_terms(rows, ("ur",), 3)
    candidates = _multiply_term_sets(
        _multiply_term_sets(state, yaw_rate), yaw_input
    )
    candidates.pop("1", None)
    return candidates


def _canonical_role(row: Mapping[str, Any]) -> str | None:
    return ROLE_ALIASES.get(str(row.get("analysis_role", row.get("role", ""))))


def _nuisance_terms(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    days = sorted({str(row["day"]) for row in rows})
    terms = {
        f"bias_{day}": np.asarray(
            [
                float(row["inverse_moment_scale_per_Nm"])
                if str(row["day"]) == day
                else 0.0
                for row in rows
            ]
        )
        for day in days
    }
    terms["Izz_kg_m2"] = np.asarray(
        [float(row["minus_acceleration_over_moment_scale"]) for row in rows]
    )
    terms["Ip_kg_m2"] = np.asarray(
        [
            float(row["rotor_reaction_acceleration_over_moment_scale"])
            for row in rows
        ]
    )
    return terms


def identify_dynamic_yaw_model(
    rows: Sequence[Mapping[str, Any]], config: SunStaticConfig | None = None
) -> dict[str, Any]:
    """Fit original-selection and protected-physics Cn models on T runs."""

    cfg = config or SunStaticConfig()
    grouped = {
        role: [row for row in rows if _canonical_role(row) == role]
        for role in ("training", "model_selection", "development_holdout")
    }
    training = grouped["training"]
    if len(training) < 100:
        raise ValueError("dynamic Yaw identification requires at least 100 training samples")
    candidates = dynamic_yaw_candidate_terms(training)
    nuisance = _nuisance_terms(training)
    target = np.asarray([float(row["Cn_observed"]) for row in training])
    models: dict[str, Any] = {}
    for model_name, forced_names in (
        ("sun_original_stepwise", ()),
        ("physics_forced_stepwise", ("ur", "r_bar")),
    ):
        fixed = dict(nuisance)
        for name in forced_names:
            fixed[name] = candidates[name]
        available = {name: value for name, value in candidates.items() if name not in fixed}
        selected, history = forward_backward_stepwise(
            target, available, fixed=fixed, priority=[], config=cfg
        )
        design = _design(fixed, candidates, selected)
        coefficients, _, _ = _fit(design, target)
        names = ["1", *fixed, *selected]
        model: dict[str, Any] = {
            "feature_names": names,
            "coefficients": dict(zip(names, map(float, coefficients), strict=True)),
            "selected_terms": selected,
            "design_rank": int(np.linalg.matrix_rank(design)),
            "condition_number": float(np.linalg.cond(design)),
            "selection_history": history,
            "metrics": {},
        }
        for role, role_rows in grouped.items():
            if not role_rows:
                continue
            role_candidates = dynamic_yaw_candidate_terms(role_rows)
            role_nuisance = _nuisance_terms(role_rows)
            role_fixed = {
                name: role_nuisance.get(name, role_candidates.get(name, np.zeros(len(role_rows))))
                for name in fixed
            }
            role_design = _design(role_fixed, role_candidates, selected)
            observed = np.asarray([float(row["Cn_observed"]) for row in role_rows])
            model["metrics"][role] = _metrics(observed, role_design @ coefficients)
        models[model_name] = model
    return {
        "method": "Sun dynamic Yaw Cn gray-box with fixed-rig load-balance observation",
        "recommended_model": "physics_forced_stepwise",
        "rotor_inertia": {
            "status": "ESTIMATED_JOINTLY_NOT_FIXED_FROM_ARCHIVED_MOTOR_TEST",
            "archived_candidate_kg_m2": 5.492931503258379e-05,
            "archived_candidate_adoption": "DO_NOT_APPLY_AS_FIXED_IP_IN_GRAYBOX",
        },
        "models": {"Cn": models},
    }


def _sun_regime(row: Mapping[str, Any]) -> str | None:
    mu = float(row["mu"])
    if mu <= 0.05:
        return "mu_le_0p05"
    beta = abs(math.degrees(float(row["beta_rad"])))
    if not math.isfinite(beta):
        return None
    if beta < 30.0:
        return "mu_gt_0p05_beta_0_30"
    if beta < 60.0:
        return "mu_gt_0p05_beta_30_60"
    if beta <= 90.0 + 1.0e-9:
        return "mu_gt_0p05_beta_60_90"
    return None


def _cn_target(
    rows: Sequence[Mapping[str, Any]], body_inertia_kg_m2: float,
    rotor_inertia_kg_m2: float,
) -> np.ndarray:
    """Recover Cn using Sun's dynamics-first treatment of known inertias."""

    observed_support_term = np.asarray([float(row["Cn_observed"]) for row in rows])
    minus_body = np.asarray(
        [float(row["minus_acceleration_over_moment_scale"]) for row in rows]
    )
    rotor_reaction = np.asarray(
        [
            float(row["rotor_reaction_acceleration_over_moment_scale"])
            for row in rows
        ]
    )
    return (
        observed_support_term
        - body_inertia_kg_m2 * minus_body
        - rotor_inertia_kg_m2 * rotor_reaction
    )


def identify_sun_eq43_piecewise_yaw_models(
    rows: Sequence[Mapping[str, Any]], *, body_inertia_kg_m2: float,
    rotor_inertia_cases_kg_m2: Mapping[str, float],
    config: SunStaticConfig | None = None,
) -> dict[str, Any]:
    """Fit Sun Eq. (43) and a physics-protected variant in four regimes.

    The protected model always retains ``ur`` (the normalized rotor-speed-
    squared difference) and ``r_bar`` (Yaw-rate damping).  Stepwise selection
    is applied only to the remaining Sun Eq. (43) interaction terms.
    """

    cfg = config or SunStaticConfig()
    regime_names = (
        "mu_le_0p05",
        "mu_gt_0p05_beta_0_30",
        "mu_gt_0p05_beta_30_60",
        "mu_gt_0p05_beta_60_90",
    )
    result: dict[str, Any] = {
        "method": "Sun Eq. (43) piecewise Cn model",
        "candidate_set": "P5(mu_x,mu_y,mu_z) P3(r_bar) P3(ur)",
        "recommended_model": "physics_protected_stepwise",
        "physical_baseline": {
            "yaw_control": (
                "ur = ((Omega1^2 + Omega2^2) - (Omega3^2 + Omega4^2)) "
                "/ omega_bar^2"
            ),
            "yaw_damping": "r_bar = r*b/(omega_bar*R)",
            "policy": (
                "ur and r_bar are mandatory fitted terms; Sun Eq. (43) terms "
                "are selected only as corrections"
            ),
        },
        "regimes": list(regime_names),
        "body_inertia_kg_m2": float(body_inertia_kg_m2),
        "rotor_inertia_cases_kg_m2": dict(rotor_inertia_cases_kg_m2),
        "models": {},
    }
    for case, rotor_inertia in rotor_inertia_cases_kg_m2.items():
        case_result: dict[str, Any] = {}
        for regime in regime_names:
            grouped = {
                role: [
                    row for row in rows
                    if _canonical_role(row) == role and _sun_regime(row) == regime
                ]
                for role in ("training", "model_selection", "development_holdout")
            }
            training = grouped["training"]
            if len(training) < 100:
                raise ValueError(f"{regime}: fewer than 100 training samples")
            candidates = sun_eq43_cn_candidate_terms(training)
            nuisance = {
                name: values
                for name, values in _nuisance_terms(training).items()
                if name.startswith("bias_")
            }
            target = _cn_target(training, body_inertia_kg_m2, rotor_inertia)
            regime_models: dict[str, Any] = {}
            for model_name, protected_names in (
                ("sun_original_stepwise", ()),
                ("physics_protected_stepwise", ("ur", "r_bar")),
            ):
                fixed = dict(nuisance)
                for name in protected_names:
                    fixed[name] = candidates[name]
                available = {
                    name: values for name, values in candidates.items()
                    if name not in fixed
                }
                selected, history = forward_backward_stepwise(
                    target, available, fixed=fixed, priority=[], config=cfg
                )
                design = _design(fixed, candidates, selected)
                coefficients, _, _ = _fit(design, target)
                names = ["1", *fixed, *selected]
                model: dict[str, Any] = {
                    "feature_names": names,
                    "coefficients": dict(
                        zip(names, map(float, coefficients), strict=True)
                    ),
                    "protected_physics_terms": list(protected_names),
                    "selected_terms": selected,
                    "candidate_count": len(candidates),
                    "design_rank": int(np.linalg.matrix_rank(design)),
                    "condition_number": float(np.linalg.cond(design)),
                    "selection_history": history,
                    "metrics": {},
                    "counts": {
                        role: len(role_rows) for role, role_rows in grouped.items()
                    },
                }
                for role, role_rows in grouped.items():
                    if not role_rows:
                        continue
                    role_candidates = sun_eq43_cn_candidate_terms(role_rows)
                    role_nuisance_all = _nuisance_terms(role_rows)
                    role_fixed = {
                        name: role_nuisance_all.get(
                            name, role_candidates.get(name, np.zeros(len(role_rows)))
                        )
                        for name in fixed
                    }
                    role_design = _design(role_fixed, role_candidates, selected)
                    role_target = _cn_target(
                        role_rows, body_inertia_kg_m2, rotor_inertia
                    )
                    model["metrics"][role] = _metrics(
                        role_target, role_design @ coefficients
                    )
                regime_models[model_name] = model
            case_result[regime] = regime_models
        result["models"][case] = case_result
    return result


def _shared_yaw_design(
    rows: Sequence[Mapping[str, Any]], *, bias_names: Sequence[str],
    regimes: Sequence[str], selected_by_regime: Mapping[str, Sequence[str]],
    include_ur: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build a joint design with shared physics and regime-local corrections."""

    count = len(rows)
    nuisance_all = _nuisance_terms(rows)
    candidates = sun_eq43_cn_candidate_terms(rows)
    columns = [np.ones(count)]
    names = ["1"]
    for name in bias_names:
        columns.append(nuisance_all.get(name, np.zeros(count)))
        names.append(name)
    if include_ur:
        columns.append(np.asarray([float(row["ur"]) for row in rows]))
        names.append("ur_shared")
    columns.append(-np.asarray([float(row["r_bar"]) for row in rows]))
    names.append("minus_r_bar_shared")
    row_regimes = np.asarray([_sun_regime(row) for row in rows], dtype=object)
    # The first regime is the reference intercept; the other three receive an
    # explicit offset without duplicating the global intercept.
    for regime in regimes[1:]:
        columns.append((row_regimes == regime).astype(float))
        names.append(f"{regime}::offset")
    for regime in regimes:
        mask = (row_regimes == regime).astype(float)
        for term in selected_by_regime[regime]:
            columns.append(mask * candidates[term])
            names.append(f"{regime}::{term}")
    return np.column_stack(columns), names


def identify_shared_physics_piecewise_yaw_models(
    rows: Sequence[Mapping[str, Any]], *, body_inertia_kg_m2: float,
    rotor_inertia_cases_kg_m2: Mapping[str, float],
    fixed_kq_Nm_per_rad_s_squared: float | None = None,
    config: SunStaticConfig | None = None,
) -> dict[str, Any]:
    """Fit one global KQ and dissipative Yaw-rate term plus Sun corrections.

    ``ur_shared`` is constrained nonnegative. ``minus_r_bar_shared`` is also
    constrained nonnegative, which is equivalent to requiring Cn_r <= 0.
    Both coefficients are shared across all four Sun flight regimes.
    """

    cfg = config or SunStaticConfig()
    regimes = (
        "mu_le_0p05",
        "mu_gt_0p05_beta_0_30",
        "mu_gt_0p05_beta_30_60",
        "mu_gt_0p05_beta_60_90",
    )
    grouped = {
        role: [
            row for row in rows
            if _canonical_role(row) == role and _sun_regime(row) in regimes
        ]
        for role in ("training", "model_selection", "development_holdout")
    }
    training = grouped["training"]
    bias_names = tuple(
        name for name in _nuisance_terms(training) if name.startswith("bias_")
    )
    disk_area = cfg.rotor_count * math.pi * cfg.rotor_radius_m**2
    kq_scale = (
        cfg.reference_length_m * cfg.air_density_kg_m3 * disk_area
        * cfg.rotor_radius_m**2
    )
    result: dict[str, Any] = {
        "method": "shared physical Yaw baseline plus piecewise Sun Eq. (43) corrections",
        "recommended_model": "shared_kq_dissipative_yaw_damping",
        "equation": (
            "Cn = Cn_ur*ur + Cn_r*r_bar + regime Sun corrections; "
            "Cn_ur>=0 and Cn_r<=0"
        ),
        "correction_policy": (
            "all r_bar-dependent correction terms are excluded so the total "
            "modeled Yaw-rate contribution cannot bypass Cn_r<=0"
        ),
        "body_inertia_kg_m2": float(body_inertia_kg_m2),
        "rotor_inertia_cases_kg_m2": dict(rotor_inertia_cases_kg_m2),
        "fixed_kq_Nm_per_rad_s_squared": fixed_kq_Nm_per_rad_s_squared,
        "models": {},
    }
    fixed_cn_ur = (
        None if fixed_kq_Nm_per_rad_s_squared is None
        else float(fixed_kq_Nm_per_rad_s_squared) / kq_scale
    )
    for case, rotor_inertia in rotor_inertia_cases_kg_m2.items():
        target = _cn_target(training, body_inertia_kg_m2, rotor_inertia)
        if fixed_cn_ur is not None:
            target -= fixed_cn_ur * np.asarray([float(row["ur"]) for row in training])
        # Select only correction structure after removing a preliminary shared
        # physical baseline. The final coefficients are then refit jointly.
        empty_selection = {regime: () for regime in regimes}
        baseline_design, baseline_names = _shared_yaw_design(
            training, bias_names=bias_names, regimes=regimes,
            selected_by_regime=empty_selection,
            include_ur=fixed_cn_ur is None,
        )
        lower = np.full(baseline_design.shape[1], -np.inf)
        upper = np.full(baseline_design.shape[1], np.inf)
        if fixed_cn_ur is None:
            lower[baseline_names.index("ur_shared")] = 0.0
        lower[baseline_names.index("minus_r_bar_shared")] = 0.0
        baseline_fit = lsq_linear(baseline_design, target, bounds=(lower, upper))
        residual = target - baseline_design @ baseline_fit.x

        training_regimes = np.asarray(
            [_sun_regime(row) for row in training], dtype=object
        )
        selected_by_regime: dict[str, list[str]] = {}
        selection_history: dict[str, Any] = {}
        for regime in regimes:
            mask = training_regimes == regime
            regime_rows = [row for row, keep in zip(training, mask, strict=True) if keep]
            candidates = sun_eq43_cn_candidate_terms(regime_rows)
            # KQ remains the only pure linear ur term. All r_bar-dependent
            # corrections are excluded so no local term can create apparent
            # anti-damping behind the shared Cn_r <= 0 constraint.
            candidates = {
                name: values for name, values in candidates.items()
                if name != "ur" and "r_bar" not in name
            }
            selected, history = forward_backward_stepwise(
                residual[mask], candidates, fixed={}, priority=[], config=cfg
            )
            selected_by_regime[regime] = selected
            selection_history[regime] = history

        design, names = _shared_yaw_design(
            training, bias_names=bias_names, regimes=regimes,
            selected_by_regime=selected_by_regime,
            include_ur=fixed_cn_ur is None,
        )
        lower = np.full(design.shape[1], -np.inf)
        upper = np.full(design.shape[1], np.inf)
        damping_index = names.index("minus_r_bar_shared")
        if fixed_cn_ur is None:
            lower[names.index("ur_shared")] = 0.0
        lower[damping_index] = 0.0
        fit = lsq_linear(design, target, bounds=(lower, upper))
        coefficients = dict(zip(names, map(float, fit.x), strict=True))
        cn_ur = coefficients.get("ur_shared", fixed_cn_ur)
        if fixed_cn_ur is not None:
            coefficients["ur_shared_fixed"] = fixed_cn_ur
        damping_magnitude = coefficients["minus_r_bar_shared"]
        model: dict[str, Any] = {
            "feature_names": names,
            "coefficients": coefficients,
            "shared_physics": {
                "Cn_ur": cn_ur,
                "KQ_Nm_per_rad_s_squared": cn_ur * kq_scale,
                "Cn_r": -damping_magnitude,
                "damping_constraint_active": damping_magnitude <= 1.0e-12,
            },
            "selected_corrections_by_regime": selected_by_regime,
            "selection_history_by_regime": selection_history,
            "design_rank": int(np.linalg.matrix_rank(design)),
            "condition_number": float(np.linalg.cond(design)),
            "optimizer_success": bool(fit.success),
            "metrics": {},
            "regime_metrics": {},
        }
        for role, role_rows in grouped.items():
            if not role_rows:
                continue
            role_design, role_names = _shared_yaw_design(
                role_rows, bias_names=bias_names, regimes=regimes,
                selected_by_regime=selected_by_regime,
                include_ur=fixed_cn_ur is None,
            )
            if role_names != names:
                raise RuntimeError("shared Yaw design columns changed across roles")
            role_target = _cn_target(
                role_rows, body_inertia_kg_m2, rotor_inertia
            )
            fixed_prediction = np.zeros(len(role_rows))
            if fixed_cn_ur is not None:
                fixed_prediction = fixed_cn_ur * np.asarray([
                    float(row["ur"]) for row in role_rows
                ])
            predicted = role_design @ fit.x
            predicted += fixed_prediction
            model["metrics"][role] = _metrics(role_target, predicted)
            role_regimes = np.asarray(
                [_sun_regime(row) for row in role_rows], dtype=object
            )
            model["regime_metrics"][role] = {
                regime: _metrics(
                    role_target[role_regimes == regime],
                    predicted[role_regimes == regime],
                )
                for regime in regimes if np.any(role_regimes == regime)
            }
        result["models"][case] = model
    return result
