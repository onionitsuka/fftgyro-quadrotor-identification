import numpy as np

from fftgyro_quad_id.identification.sun_dynamic_yaw import (
    dynamic_yaw_candidate_terms,
    dynamic_yaw_sample,
    identify_shared_physics_piecewise_yaw_models,
    identify_sun_eq43_piecewise_yaw_models,
    sun_eq43_cn_candidate_terms,
)


def test_dynamic_yaw_sample_has_body_and_rotor_inertia_regressors():
    row = dynamic_yaw_sample(
        rate_rad_s=0.2,
        acceleration_rad_s2=0.3,
        rotor_spin_acceleration_rad_s2=-40.0,
        wind_speed_m_s=5.0,
        pitch_rad=-0.2,
        heading_rad=0.1,
        roll_rad=0.0,
        rpm_motor_order=[4000.0, 3900.0, 4100.0, 4200.0],
        support_moment_yaw_Nm=-0.1,
        day="20260917",
    )
    assert row["Cn_observed"] > 0
    assert row["minus_acceleration_over_moment_scale"] < 0
    assert row["rotor_reaction_acceleration_over_moment_scale"] < 0
    assert np.isfinite(row["r_bar"])


def test_dynamic_yaw_candidates_include_rate_and_yaw_input():
    row = dynamic_yaw_sample(
        rate_rad_s=0.2,
        acceleration_rad_s2=0.3,
        rotor_spin_acceleration_rad_s2=40.0,
        wind_speed_m_s=5.0,
        pitch_rad=-0.2,
        heading_rad=0.1,
        roll_rad=0.0,
        rpm_motor_order=[4000.0, 3900.0, 4100.0, 4200.0],
        support_moment_yaw_Nm=0.1,
        day="20260917",
    )
    candidates = dynamic_yaw_candidate_terms([row])
    assert "r_bar" in candidates
    assert "ur" in candidates
    assert "mu_x*r_bar" in candidates
    assert "mu_y*ur" in candidates


def test_sun_eq43_candidates_are_products_of_three_polynomial_sets():
    row = dynamic_yaw_sample(
        rate_rad_s=0.2, acceleration_rad_s2=0.3,
        rotor_spin_acceleration_rad_s2=40.0, wind_speed_m_s=5.0,
        pitch_rad=-0.2, heading_rad=0.1, roll_rad=0.0,
        rpm_motor_order=[4000.0, 3900.0, 4100.0, 4200.0],
        support_moment_yaw_Nm=0.1, day="20260917",
    )
    candidates = sun_eq43_cn_candidate_terms([row])
    assert "mu_x^5*r_bar^3*ur^3" in candidates
    assert "mu_x*mu_y*mu_z*r_bar^2*ur^2" in candidates
    assert "r_bar^3" in candidates
    assert "ur^3" in candidates
    assert "1" not in candidates


def test_piecewise_sun_model_protects_standard_yaw_control_and_damping_terms():
    rows = []
    roles = ("T", "D", "H")
    for index in range(360):
        ur = 0.25 * np.sin(index * 0.17)
        r_bar = 0.01 * np.cos(index * 0.11)
        role = roles[index % len(roles)]
        rows.append({
            "role": role,
            "day": "20260917",
            "mu": 0.03,
            "mu_x": 0.02,
            "mu_y": 0.0,
            "mu_z": 0.01,
            "beta_rad": 0.0,
            "r_bar": r_bar,
            "ur": ur,
            "inverse_moment_scale_per_Nm": 1.0,
            "minus_acceleration_over_moment_scale": 0.0,
            "rotor_reaction_acceleration_over_moment_scale": 0.0,
            "Cn_observed": 0.2 * ur - 0.4 * r_bar,
        })
    # Populate the three non-hover regimes so the production guard is met.
    expanded = []
    for beta_deg in (15.0, 45.0, 75.0):
        for row in rows:
            copy = dict(row)
            copy["mu"] = 0.1
            copy["beta_rad"] = np.deg2rad(beta_deg)
            expanded.append(copy)
    result = identify_sun_eq43_piecewise_yaw_models(
        [*rows, *expanded], body_inertia_kg_m2=0.0256,
        rotor_inertia_cases_kg_m2={"omitted": 0.0},
    )
    assert result["recommended_model"] == "physics_protected_stepwise"
    for regime_models in result["models"]["omitted"].values():
        protected = regime_models["physics_protected_stepwise"]
        assert protected["protected_physics_terms"] == ["ur", "r_bar"]
        assert "ur" in protected["feature_names"]
        assert "r_bar" in protected["feature_names"]


def test_shared_physics_model_uses_one_kq_and_dissipative_damping():
    rows = []
    roles = ("T", "D", "H")
    regimes = ((0.03, 0.0), (0.1, 15.0), (0.1, 45.0), (0.1, 75.0))
    for mu, beta_deg in regimes:
        for index in range(360):
            ur = 0.25 * np.sin(index * 0.17)
            r_bar = 0.01 * np.cos(index * 0.11)
            rows.append({
                "role": roles[index % len(roles)],
                "day": "20260917",
                "mu": mu,
                "mu_x": 0.02,
                "mu_y": 0.01 * np.sin(np.deg2rad(beta_deg)),
                "mu_z": 0.01,
                "beta_rad": np.deg2rad(beta_deg),
                "r_bar": r_bar,
                "ur": ur,
                "inverse_moment_scale_per_Nm": 1.0,
                "minus_acceleration_over_moment_scale": 0.0,
                "rotor_reaction_acceleration_over_moment_scale": 0.0,
                "Cn_observed": 0.2 * ur - 0.4 * r_bar,
            })
    result = identify_shared_physics_piecewise_yaw_models(
        rows, body_inertia_kg_m2=0.0256,
        rotor_inertia_cases_kg_m2={"omitted": 0.0},
    )
    physics = result["models"]["omitted"]["shared_physics"]
    np.testing.assert_allclose(physics["Cn_ur"], 0.2, rtol=1e-6)
    np.testing.assert_allclose(physics["Cn_r"], -0.4, rtol=1e-6)


def test_shared_physics_model_can_fix_independent_kq():
    rows = []
    roles = ("T", "D", "H")
    regimes = ((0.03, 0.0), (0.1, 15.0), (0.1, 45.0), (0.1, 75.0))
    for mu, beta_deg in regimes:
        for index in range(360):
            ur = 0.25 * np.sin(index * 0.17)
            r_bar = 0.01 * np.cos(index * 0.11)
            rows.append({
                "role": roles[index % 3], "day": "20260917", "mu": mu,
                "mu_x": 0.02, "mu_y": 0.01, "mu_z": 0.01,
                "beta_rad": np.deg2rad(beta_deg), "r_bar": r_bar, "ur": ur,
                "inverse_moment_scale_per_Nm": 1.0,
                "minus_acceleration_over_moment_scale": 0.0,
                "rotor_reaction_acceleration_over_moment_scale": 0.0,
                "Cn_observed": 0.2 * ur - 0.4 * r_bar,
            })
    # With the default geometry, this KQ corresponds exactly to Cn_ur=0.2.
    from fftgyro_quad_id.identification.sun_static import SunStaticConfig
    cfg = SunStaticConfig()
    scale = (cfg.reference_length_m * cfg.air_density_kg_m3
             * cfg.rotor_count * np.pi * cfg.rotor_radius_m**2
             * cfg.rotor_radius_m**2)
    result = identify_shared_physics_piecewise_yaw_models(
        rows, body_inertia_kg_m2=0.0256,
        rotor_inertia_cases_kg_m2={"measured": 0.0},
        fixed_kq_Nm_per_rad_s_squared=0.2 * scale,
    )
    physics = result["models"]["measured"]["shared_physics"]
    np.testing.assert_allclose(physics["Cn_ur"], 0.2, rtol=1e-12)
    np.testing.assert_allclose(physics["Cn_r"], -0.4, rtol=1e-6)
