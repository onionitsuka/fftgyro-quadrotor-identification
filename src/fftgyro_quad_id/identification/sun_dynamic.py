"""Sun-style dynamic Pitch gray-box identification for the fixed FFTGYRO rig.

The aerodynamic regressors follow Sun, de Visser, and Chu (2019), while the
observation equation is specialized to a one-axis force-balance experiment.
Rotor-inertia moments are deliberately not applied until propeller inertia is
measured; the omission is explicit in every fit result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fftgyro_quad_id.identification.sun_static import (
    SunStaticConfig,
    _design,
    _fit,
    _metrics,
    _multiply_term_sets,
    _polynomial_terms,
    forward_backward_stepwise,
    solve_normalized_induced_velocity,
    sun_motor_inputs,
    wrench_scales,
)

DYNAMIC_PITCH_OUTPUTS = ("Cx", "Cy", "Cz", "Cm")
ROLE_ALIASES = {
    "T": "training",
    "D": "model_selection",
    "H": "development_holdout",
    "training": "training",
    "model_selection": "model_selection",
    "development_holdout": "development_holdout",
}


def canonical_role(row: Mapping[str, Any]) -> str | None:
    return ROLE_ALIASES.get(str(row.get("analysis_role", row.get("role", ""))))


def fixed_rig_airspeed_frd(
    speed_m_s: float, pitch_rad: float, heading_rad: float, roll_rad: float
) -> np.ndarray:
    """Return Sun FRD airspeed for the fixed rig.

    At zero attitude, positive tunnel speed gives ``[V, 0, 0]``.  The campaign
    convention has positive heading expose the aircraft's left side, hence
    negative Sun/FRD lateral airspeed.  Positive encoder Pitch gives positive
    Sun angle of attack.  Roll rotates the lateral/vertical components only.
    """
    speed = float(speed_m_s)
    pitch = float(pitch_rad)
    heading = float(heading_rad)
    roll = float(roll_rad)
    if not all(math.isfinite(value) for value in (speed, pitch, heading, roll)) or speed < 0:
        raise ValueError("finite nonnegative speed and finite attitude are required")
    unrolled = np.array(
        [
            speed * math.cos(heading) * math.cos(pitch),
            -speed * math.sin(heading),
            speed * math.cos(heading) * math.sin(pitch),
        ]
    )
    c, s = math.cos(roll), math.sin(roll)
    return np.array(
        [unrolled[0], c * unrolled[1] + s * unrolled[2], -s * unrolled[1] + c * unrolled[2]]
    )


def sun_angles(u: float, v: float, w: float) -> tuple[float, float]:
    """Return Sun Eq. (2) alpha and beta, or NaN at zero airspeed."""
    speed = float(np.linalg.norm([u, v, w]))
    horizontal = math.hypot(u, v)
    if speed <= 1.0e-12 or horizontal <= 1.0e-12:
        return math.nan, math.nan
    alpha = math.asin(float(np.clip(w / speed, -1.0, 1.0)))
    beta = math.asin(float(np.clip(v / horizontal, -1.0, 1.0)))
    return alpha, beta


def dynamic_pitch_sample(
    *,
    angle_rad: float,
    rate_rad_s: float,
    acceleration_rad_s2: float,
    wind_speed_m_s: float,
    heading_rad: float,
    roll_rad: float,
    rpm: Sequence[float],
    force_ap_flu_N: Sequence[float],
    support_moment_pitch_Nm: float,
    day: str,
    config: SunStaticConfig | None = None,
) -> dict[str, float | str]:
    """Build one dynamic Sun row without assuming aircraft Pitch inertia.

    ``support_moment_pitch_Nm`` is the measured support moment about FRD +Y at
    the CG.  From ``M_external = Iyy*qdot - M_support`` (with rotor inertia
    omitted), the regression observation is ``-M_support/Q_M`` and the design
    includes ``-qdot/Q_M`` whose fitted coefficient is ``Iyy``.
    """
    cfg = config or SunStaticConfig()
    rotor = sun_motor_inputs(np.asarray(rpm, dtype=float) * (2.0 * math.pi / 60.0))
    omega_bar = rotor["omega_bar_rad_s"]
    force_scale, moment_scale = wrench_scales(cfg, omega_bar)
    u, v, w = fixed_rig_airspeed_frd(
        wind_speed_m_s, angle_rad, heading_rad, roll_rad
    )
    alpha, beta = sun_angles(u, v, w)
    denominator = omega_bar * cfg.rotor_radius_m
    force_flu = np.asarray(force_ap_flu_N, dtype=float)
    if force_flu.shape != (3,) or not np.all(np.isfinite(force_flu)):
        raise ValueError("force_ap_flu_N must contain three finite values")
    # FLU -> Sun FRD: x unchanged, y and z reversed.
    force_frd = force_flu * np.array([1.0, -1.0, -1.0])
    result: dict[str, float | str] = {
        "day": str(day),
        "alpha_rad": alpha,
        "beta_rad": beta,
        "airspeed_u_m_s": float(u),
        "airspeed_v_m_s": float(v),
        "airspeed_w_m_s": float(w),
        "mu_x": float(u / denominator),
        "mu_y": float(v / denominator),
        "mu_z": float(w / denominator),
        "abs_mu_x": float(abs(u / denominator)),
        "abs_mu_y": float(abs(v / denominator)),
        "mu_h2": float((u * u + v * v) / denominator**2),
        "q_bar": float(rate_rad_s * cfg.reference_length_m / denominator),
        "abs_q_bar": float(abs(rate_rad_s) * cfg.reference_length_m / denominator),
        "force_scale_N": force_scale,
        "moment_scale_Nm": moment_scale,
        "inverse_force_scale_per_N": 1.0 / force_scale,
        "inverse_moment_scale_per_Nm": 1.0 / moment_scale,
        "minus_acceleration_over_moment_scale": -float(acceleration_rad_s2) / moment_scale,
        "Cx_observed": float(force_frd[0] / force_scale),
        "Cy_observed": float(force_frd[1] / force_scale),
        "Cz_observed": float(force_frd[2] / force_scale),
        "Cm_observed": -float(support_moment_pitch_Nm) / moment_scale,
        **rotor,
    }
    result.update(
        {
            "abs_up": abs(float(rotor["up"])),
            "abs_uq": abs(float(rotor["uq"])),
            "abs_ur": abs(float(rotor["ur"])),
        }
    )
    return result


def dynamic_pitch_candidate_terms(
    rows: Sequence[Mapping[str, Any]], output: str
) -> tuple[dict[str, np.ndarray], list[str], dict[str, np.ndarray]]:
    """Return the Pitch-relevant dynamic subsets of Sun Eqs. (29,30,36,39)."""
    if output == "Cx":
        return _polynomial_terms(rows, ("mu_x", "abs_mu_y", "mu_z"), 3), ["mu_x"], {}
    if output == "Cy":
        return _polynomial_terms(rows, ("abs_mu_x", "mu_y", "mu_z"), 3), ["mu_y"], {}
    if output == "Cz":
        base = _polynomial_terms(rows, ("abs_mu_x", "abs_mu_y", "mu_z"), 4)
        multipliers = {
            "1": np.ones(len(rows)),
            "abs_q_bar": np.asarray([float(row["abs_q_bar"]) for row in rows]),
            "abs_up": np.asarray([float(row["abs_up"]) for row in rows]),
            "abs_uq": np.asarray([float(row["abs_uq"]) for row in rows]),
            "abs_ur": np.asarray([float(row["abs_ur"]) for row in rows]),
        }
        candidates = _multiply_term_sets(base, multipliers)
        fixed = {
            "mu_h2": np.asarray([float(row["mu_h2"]) for row in rows]),
            "nu_in-mu_z": np.asarray(
                [float(row["nu_in"]) - float(row["mu_z"]) for row in rows]
            ),
        }
        return candidates, [], fixed
    if output == "Cm":
        longitudinal = _polynomial_terms(rows, ("mu_x", "mu_z"), 5)
        lateral = _polynomial_terms(rows, ("abs_mu_y",), 2)
        dynamic = {
            "1": np.ones(len(rows)),
            "q_bar": np.asarray([float(row["q_bar"]) for row in rows]),
            "uq": np.asarray([float(row["uq"]) for row in rows]),
        }
        return _multiply_term_sets(_multiply_term_sets(longitudinal, lateral), dynamic), [], {}
    raise ValueError(f"unsupported dynamic Pitch output {output!r}")


def _nuisance_terms(
    rows: Sequence[Mapping[str, Any]], output: str, *, fit_pitch_inertia: bool = True
) -> dict[str, np.ndarray]:
    scale = "inverse_moment_scale_per_Nm" if output == "Cm" else "inverse_force_scale_per_N"
    days = sorted({str(row["day"]) for row in rows})
    nuisance = {
        f"bias_{day}": np.asarray(
            [float(row[scale]) if str(row["day"]) == day else 0.0 for row in rows]
        )
        for day in days
    }
    if output == "Cm" and fit_pitch_inertia:
        nuisance["Iyy_kg_m2"] = np.asarray(
            [float(row["minus_acceleration_over_moment_scale"]) for row in rows]
        )
    return nuisance


def _augment_inflow(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    hover = [
        -float(row["Cz_observed"])
        for row in rows
        if abs(float(row["wind_speed_m_s"])) <= 1.0e-12
        and abs(float(row["uq"])) < 0.25
    ]
    ct_hover = float(np.median(hover)) if hover else 0.0
    output = []
    for source in rows:
        row = dict(source)
        row["nu_in"] = solve_normalized_induced_velocity(
            ct_hover, float(row["mu_x"]), float(row["mu_y"]), float(row["mu_z"])
        )
        output.append(row)
    return output, ct_hover


def identify_dynamic_pitch_models(
    rows: Sequence[Mapping[str, Any]], config: SunStaticConfig | None = None,
    *, pitch_inertia_kg_m2: float | None = None,
) -> dict[str, Any]:
    """Fit Sun-original and protected-physics dynamic Pitch models on T runs."""
    cfg = config or SunStaticConfig()
    augmented, ct_hover = _augment_inflow(rows)
    grouped = {
        role: [row for row in augmented if canonical_role(row) == role]
        for role in ("training", "model_selection", "development_holdout")
    }
    training = grouped["training"]
    if len(training) < 100:
        raise ValueError("dynamic Pitch identification requires at least 100 training samples")
    result: dict[str, Any] = {
        "method": "Sun dynamic Pitch gray-box with fixed-rig load-balance observation equation",
        "recommended_model": "physics_forced_stepwise",
        "physical_baseline": {
            "pitch_control": (
                "uq = ((Omega1^2 + Omega3^2) - (Omega2^2 + Omega4^2)) "
                "/ omega_bar^2"
            ),
            "pitch_damping": "q_bar = q*b/(omega_bar*R)",
            "policy": (
                "uq and q_bar are mandatory fitted terms; Sun polynomial terms "
                "are selected only as corrections"
            ),
        },
        "ct_hover_median": ct_hover,
        "rotor_inertia": {
            "status": "NO_DIRECT_IDEAL_ONE_AXIS_PITCH_TERM",
            "pitch_axis_direct_term": "zero for ideal one-axis Pitch when body roll rate p=0",
        },
        "pitch_inertia": {
            "mode": "fixed_independent_measurement" if pitch_inertia_kg_m2 is not None
            else "estimated_jointly",
            "Iyy_kg_m2": pitch_inertia_kg_m2,
        },
        "models": {},
    }
    forced_names = {
        "Cx": ("mu_x",),
        "Cy": ("mu_y",),
        "Cz": ("mu_h2", "nu_in-mu_z"),
        "Cm": ("uq", "q_bar"),
    }
    for output in DYNAMIC_PITCH_OUTPUTS:
        candidates, priority, sun_fixed = dynamic_pitch_candidate_terms(training, output)
        candidates.pop("1", None)
        fit_pitch_inertia = not (output == "Cm" and pitch_inertia_kg_m2 is not None)
        nuisance = _nuisance_terms(
            training, output, fit_pitch_inertia=fit_pitch_inertia
        )
        sun_fixed = {**nuisance, **sun_fixed}
        physics_fixed = dict(sun_fixed)
        for name in forced_names[output]:
            if name in candidates:
                physics_fixed[name] = candidates[name]
        target = np.asarray([float(row[f"{output}_observed"]) for row in training])
        if output == "Cm" and pitch_inertia_kg_m2 is not None:
            target -= pitch_inertia_kg_m2 * np.asarray([
                float(row["minus_acceleration_over_moment_scale"])
                for row in training
            ])
        models = {}
        for model_name, fixed in (
            ("sun_original_stepwise", sun_fixed),
            ("physics_forced_stepwise", physics_fixed),
        ):
            available = {name: values for name, values in candidates.items() if name not in fixed}
            selected, history = forward_backward_stepwise(
                target, available, fixed=fixed, priority=priority, config=cfg
            )
            design = _design(fixed, candidates, selected)
            coefficients, _, _ = _fit(design, target)
            names = ["1", *fixed, *selected]
            model = {
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
                role_candidates, _, role_sun_fixed = dynamic_pitch_candidate_terms(
                    role_rows, output
                )
                role_candidates.pop("1", None)
                available_fixed = {
                    **_nuisance_terms(
                        role_rows, output, fit_pitch_inertia=fit_pitch_inertia
                    ),
                    **role_sun_fixed,
                }
                available_fixed.update(
                    {
                        name: role_candidates[name]
                        for name in fixed
                        if name in role_candidates
                    }
                )
                # Preserve training columns; unseen days receive zero bias rather than leakage.
                role_fixed = {
                    name: available_fixed.get(name, np.zeros(len(role_rows)))
                    for name in fixed
                }
                role_design = _design(role_fixed, role_candidates, selected)
                observed = np.asarray(
                    [float(row[f"{output}_observed"]) for row in role_rows]
                )
                if output == "Cm" and pitch_inertia_kg_m2 is not None:
                    observed -= pitch_inertia_kg_m2 * np.asarray([
                        float(row["minus_acceleration_over_moment_scale"])
                        for row in role_rows
                    ])
                model["metrics"][role] = _metrics(observed, role_design @ coefficients)
            models[model_name] = model
        result["models"][output] = models
    return result
