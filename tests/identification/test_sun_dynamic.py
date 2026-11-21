import math

import numpy as np

from fftgyro_quad_id.identification.sun_dynamic import (
    dynamic_pitch_candidate_terms,
    dynamic_pitch_sample,
    fixed_rig_airspeed_frd,
    identify_dynamic_pitch_models,
    sun_angles,
)


def test_fixed_rig_angles_follow_sun_frd_signs():
    pitch, heading = np.deg2rad([20.0, 15.0])
    u, v, w = fixed_rig_airspeed_frd(10.0, pitch, heading, 0.0)
    alpha, beta = sun_angles(u, v, w)
    assert alpha == np.arcsin(np.cos(heading) * np.sin(pitch))
    assert beta < 0
    np.testing.assert_allclose(np.linalg.norm([u, v, w]), 10.0)


def test_dynamic_sample_uses_raw_support_observation_and_inertia_regressor():
    row = dynamic_pitch_sample(
        angle_rad=0.0,
        rate_rad_s=0.2,
        acceleration_rad_s2=3.0,
        wind_speed_m_s=5.0,
        heading_rad=0.0,
        roll_rad=0.0,
        rpm=[4000.0] * 4,
        force_ap_flu_N=[1.0, 2.0, 3.0],
        support_moment_pitch_Nm=-0.2,
        day="20260917",
    )
    assert row["Cy_observed"] < 0
    assert row["Cz_observed"] < 0
    assert row["Cm_observed"] > 0
    assert row["minus_acceleration_over_moment_scale"] < 0
    assert all(abs(row[name]) < 1e-12 for name in ("up", "uq", "ur"))


def test_dynamic_cm_candidates_include_sun_rate_and_pitch_input():
    row = dynamic_pitch_sample(
        angle_rad=0.1,
        rate_rad_s=0.2,
        acceleration_rad_s2=0.3,
        wind_speed_m_s=5.0,
        heading_rad=0.2,
        roll_rad=0.0,
        rpm=[4000.0, 3900.0, 4100.0, 3950.0],
        force_ap_flu_N=[1.0, 2.0, 3.0],
        support_moment_pitch_Nm=0.1,
        day="20260917",
    )
    row["nu_in"] = 0.01
    candidates, _, _ = dynamic_pitch_candidate_terms([row], "Cm")
    assert "q_bar" in candidates
    assert "uq" in candidates
    assert "mu_x*q_bar" in candidates
    assert "abs_mu_y*uq" in candidates
    assert math.isfinite(float(candidates["mu_x*q_bar"][0]))


def test_pitch_recommends_model_with_standard_control_and_damping_terms():
    rows = []
    for index in range(360):
        role = ("T", "D", "H")[index % 3]
        row = dynamic_pitch_sample(
            angle_rad=0.05,
            rate_rad_s=0.2 * np.sin(index * 0.1),
            acceleration_rad_s2=0.3 * np.cos(index * 0.1),
            wind_speed_m_s=5.0,
            heading_rad=0.0,
            roll_rad=0.0,
            rpm=[4000.0 + 50.0 * np.sin(index * 0.13), 3900.0,
                 4100.0, 3950.0],
            force_ap_flu_N=[1.0, 0.0, 3.0],
            support_moment_pitch_Nm=0.1,
            day="20260917",
        )
        row.update({"role": role, "wind_speed_m_s": 5.0})
        rows.append(row)
    result = identify_dynamic_pitch_models(rows)
    assert result["recommended_model"] == "physics_forced_stepwise"
    model = result["models"]["Cm"]["physics_forced_stepwise"]
    assert "uq" in model["feature_names"]
    assert "q_bar" in model["feature_names"]


def test_pitch_can_fix_independently_measured_inertia():
    rows = []
    for index in range(360):
        row = dynamic_pitch_sample(
            angle_rad=0.05, rate_rad_s=0.2 * np.sin(index * 0.1),
            acceleration_rad_s2=0.3 * np.cos(index * 0.1), wind_speed_m_s=5.0,
            heading_rad=0.0, roll_rad=0.0,
            rpm=[4000.0 + 50.0 * np.sin(index * 0.13), 3900.0, 4100.0, 3950.0],
            force_ap_flu_N=[1.0, 0.0, 3.0], support_moment_pitch_Nm=0.1,
            day="20260917",
        )
        row.update({"role": ("T", "D", "H")[index % 3], "wind_speed_m_s": 5.0})
        rows.append(row)
    result = identify_dynamic_pitch_models(rows, pitch_inertia_kg_m2=0.0201)
    model = result["models"]["Cm"]["physics_forced_stepwise"]
    assert result["pitch_inertia"]["mode"] == "fixed_independent_measurement"
    assert "Iyy_kg_m2" not in model["feature_names"]
