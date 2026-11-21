from __future__ import annotations

import math

import numpy as np

from fftgyro_quad_id.identification.motor_single import (
    DynamicSeries,
    fit_centered_voltage_polynomial_command_rpm_map,
    fit_command_rpm_map,
    fit_dynamic_model,
    fit_wrench_coefficients,
    predict_wrench,
    simulate_first_order,
    simulate_first_order_dead_time,
    simulate_second_order,
)


def test_formal_command_map_recovers_zero_intercept_coefficients() -> None:
    command = np.tile(np.linspace(0.1, 0.8, 8), 3)
    voltage = np.repeat(np.asarray([14.7, 15.2, 15.7]), 8)
    rpm = voltage * (950.0 * command - 280.0 * np.square(command))

    fitted = fit_command_rpm_map(command, voltage, rpm)

    assert math.isclose(fitted.linear_per_volt, 950.0, rel_tol=1e-10)
    assert math.isclose(fitted.quadratic_per_volt, -280.0, rel_tol=1e-10)
    np.testing.assert_allclose(fitted.predict(command, voltage), rpm)


def test_centered_voltage_polynomial_recovers_local_candidate() -> None:
    command = np.tile(np.linspace(0.1, 0.8, 8), 4)
    voltage = np.repeat(np.asarray([14.7, 15.2, 15.7, 16.2]), 8)
    delta_voltage = voltage - 15.2
    rpm = (
        1150.0
        + 9900.0 * command
        - 850.0 * np.square(command)
        + 130.0 * delta_voltage
        - 30.0 * delta_voltage * command
        + 420.0 * delta_voltage * np.square(command)
    )

    fitted = fit_centered_voltage_polynomial_command_rpm_map(
        command, voltage, rpm
    )

    np.testing.assert_allclose(fitted.predict(command, voltage), rpm, atol=1e-8)
    assert math.isclose(fitted.intercept_rpm, 1150.0, rel_tol=1e-10)


def test_first_order_uses_exact_zero_order_hold_discretization() -> None:
    time = np.asarray([0.0, 0.1, 0.2])
    target = np.asarray([1000.0, 1000.0, 1000.0])

    predicted = simulate_first_order(time, target, 0.1, 0.0)

    assert math.isclose(predicted[1], 1000.0 * (1.0 - math.exp(-1.0)))
    assert math.isclose(predicted[2], 1000.0 * (1.0 - math.exp(-2.0)))


def test_repeated_second_order_is_monotonic_and_slower_initially() -> None:
    time = np.linspace(0.0, 1.0, 101)
    target = np.full(time.shape, 1000.0)
    first = simulate_first_order(time, target, 0.1, 0.0)
    second = simulate_second_order(time, target, 0.1, 0.1, 0.0)

    assert np.all(np.diff(second) >= 0.0)
    assert second[10] < first[10]
    assert second[-1] > 999.0


def test_rpm_fit_recovers_first_order_time_constant() -> None:
    time = np.arange(0.0, 4.0, 0.01)
    target = np.where(time < 1.0, 2500.0, 6500.0)
    measured = simulate_first_order(time, target, 0.12, 2500.0)
    series = DynamicSeries(
        run_id="synthetic",
        time_s=time,
        rpm_setpoint=target,
        rpm_measured=measured,
        evaluation_mask=np.ones(time.shape, dtype=bool),
    )

    fitted = fit_dynamic_model("first_order", [series])

    assert math.isclose(fitted.parameter_values[0], 0.12, rel_tol=1e-4)
    assert fitted.metrics.rmse < 1e-6


def test_dead_time_fit_uses_explicit_ten_millisecond_grid() -> None:
    time = np.arange(0.0, 4.0, 0.01)
    target = np.where(time < 1.0, 2500.0, 6500.0)
    measured = simulate_first_order_dead_time(
        time,
        target,
        time_constant_s=0.08,
        dead_time_s=0.04,
        initial_rpm=2500.0,
    )
    series = DynamicSeries(
        run_id="synthetic",
        time_s=time,
        rpm_setpoint=target,
        rpm_measured=measured,
        evaluation_mask=np.ones(time.shape, dtype=bool),
    )

    fitted = fit_dynamic_model("first_order_dead_time", [series])

    assert math.isclose(fitted.parameter_values[0], 0.08, rel_tol=1e-4)
    assert math.isclose(fitted.parameter_values[1], 0.04, abs_tol=1e-12)
    assert fitted.metrics.rmse < 1e-6


def test_wrench_coefficients_use_radians_per_second() -> None:
    rpm = np.asarray([2000.0, 4000.0, 6000.0, 8000.0])
    omega_squared = np.square(rpm * 2.0 * math.pi / 60.0)
    thrust = 1.2e-5 * omega_squared
    torque = 1.7e-7 * omega_squared

    fitted = fit_wrench_coefficients(rpm, thrust, torque)
    predicted_thrust, predicted_torque = predict_wrench(rpm, fitted)

    assert math.isclose(fitted.thrust_per_rad_s_squared, 1.2e-5)
    assert math.isclose(fitted.torque_per_rad_s_squared, 1.7e-7)
    np.testing.assert_allclose(predicted_thrust, thrust)
    np.testing.assert_allclose(predicted_torque, torque)
