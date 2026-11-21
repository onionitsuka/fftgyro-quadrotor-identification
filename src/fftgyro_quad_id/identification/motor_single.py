"""Model primitives for single-motor propulsion-system identification.

The Phase-1 identification path is deliberately staged:

1. fit command and measured voltage to steady RPM;
2. fit motor dynamics to measured transient RPM;
3. fit thrust and reaction torque to measured RPM;
4. validate the complete command-to-wrench chain on held-out runs.

Keeping these steps separate prevents load-cell or fixture dynamics from being
silently absorbed into the motor time constant when an independent RPM
measurement is available.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.optimize import least_squares

FloatArray = NDArray[np.float64]
DynamicModelName = Literal["first_order", "first_order_dead_time", "second_order"]


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """Common error metrics for one measured/predicted signal pair."""

    count: int
    rmse: float
    mae: float
    max_abs_error: float
    bias: float
    r_squared: float
    lag1_residual_correlation: float


@dataclass(frozen=True, slots=True)
class CommandRpmMap:
    """Zero-intercept map ``RPM = V * (a1*u + a2*u**2)``."""

    linear_per_volt: float
    quadratic_per_volt: float
    design_condition_number: float

    def predict(self, command: FloatArray, voltage_v: FloatArray) -> FloatArray:
        return voltage_v * (
            self.linear_per_volt * command
            + self.quadratic_per_volt * np.square(command)
        )

    def derivative(self, command: FloatArray, voltage_v: FloatArray) -> FloatArray:
        return voltage_v * (
            self.linear_per_volt + 2.0 * self.quadratic_per_volt * command
        )


@dataclass(frozen=True, slots=True)
class CenteredVoltagePolynomialCommandRpmMap:
    """Exploratory local polynomial with voltage-command interactions.

    With ``dV = V-reference_voltage``, the model is
    ``c0 + c1*u + c2*u² + c3*dV + c4*dV*u + c5*dV*u²``.  The finite
    intercept makes this a local model for the tested 10--100% range, not a
    physical extrapolation to a stopped motor.
    """

    reference_voltage_v: float
    intercept_rpm: float
    command_linear: float
    command_quadratic: float
    voltage_linear: float
    voltage_command_linear: float
    voltage_command_quadratic: float
    design_condition_number: float

    def predict(self, command: FloatArray, voltage_v: FloatArray) -> FloatArray:
        voltage_delta = voltage_v - self.reference_voltage_v
        return (
            self.intercept_rpm
            + self.command_linear * command
            + self.command_quadratic * np.square(command)
            + self.voltage_linear * voltage_delta
            + self.voltage_command_linear * voltage_delta * command
            + self.voltage_command_quadratic
            * voltage_delta
            * np.square(command)
        )


@dataclass(frozen=True, slots=True)
class DynamicSeries:
    """One synchronized transient run used to fit or score a dynamic model."""

    run_id: str
    time_s: FloatArray
    rpm_setpoint: FloatArray
    rpm_measured: FloatArray
    evaluation_mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class DynamicFit:
    """Optimized dynamic-model parameters and training diagnostics."""

    model: DynamicModelName
    parameter_names: tuple[str, ...]
    parameter_values: tuple[float, ...]
    parameter_standard_errors: tuple[float, ...]
    metrics: RegressionMetrics
    aicc: float
    optimizer_success: bool
    optimizer_message: str

    def parameter_dict(self) -> dict[str, float]:
        return dict(zip(self.parameter_names, self.parameter_values, strict=True))

    def standard_error_dict(self) -> dict[str, float]:
        return dict(
            zip(self.parameter_names, self.parameter_standard_errors, strict=True)
        )


@dataclass(frozen=True, slots=True)
class WrenchCoefficients:
    """Zero-intercept quadratic propulsion coefficients in SI units."""

    thrust_per_rad_s_squared: float
    torque_per_rad_s_squared: float
    torque_per_thrust_direct: float
    torque_per_thrust_ratio: float


def regression_metrics(measured: FloatArray, predicted: FloatArray) -> RegressionMetrics:
    """Calculate metrics after dropping non-finite pairs."""

    valid = np.isfinite(measured) & np.isfinite(predicted)
    actual = np.asarray(measured[valid], dtype=float)
    estimate = np.asarray(predicted[valid], dtype=float)
    if actual.size == 0:
        raise ValueError("at least one finite measured/predicted pair is required")
    residual = estimate - actual
    residual_sum = float(np.sum(np.square(residual)))
    total_sum = float(np.sum(np.square(actual - np.mean(actual))))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else math.nan
    lag1 = math.nan
    if residual.size >= 3:
        left = residual[:-1]
        right = residual[1:]
        if float(np.std(left)) > 0.0 and float(np.std(right)) > 0.0:
            lag1 = float(np.corrcoef(left, right)[0, 1])
    return RegressionMetrics(
        count=int(actual.size),
        rmse=float(np.sqrt(np.mean(np.square(residual)))),
        mae=float(np.mean(np.abs(residual))),
        max_abs_error=float(np.max(np.abs(residual))),
        bias=float(np.mean(residual)),
        r_squared=r_squared,
        lag1_residual_correlation=lag1,
    )


def fit_command_rpm_map(
    command: FloatArray,
    voltage_v: FloatArray,
    rpm: FloatArray,
) -> CommandRpmMap:
    """Fit the LaTeX-specified zero-intercept command/voltage map."""

    valid = np.isfinite(command) & np.isfinite(voltage_v) & np.isfinite(rpm)
    u = np.asarray(command[valid], dtype=float)
    voltage = np.asarray(voltage_v[valid], dtype=float)
    measured = np.asarray(rpm[valid], dtype=float)
    if measured.size < 3:
        raise ValueError("at least three steady points are required")
    design = np.column_stack((voltage * u, voltage * np.square(u)))
    coefficients, *_ = np.linalg.lstsq(design, measured, rcond=None)
    return CommandRpmMap(
        linear_per_volt=float(coefficients[0]),
        quadratic_per_volt=float(coefficients[1]),
        design_condition_number=float(np.linalg.cond(design)),
    )


def fit_centered_voltage_polynomial_command_rpm_map(
    command: FloatArray,
    voltage_v: FloatArray,
    rpm: FloatArray,
    *,
    reference_voltage_v: float = 15.2,
) -> CenteredVoltagePolynomialCommandRpmMap:
    """Fit the exploratory centered-voltage polynomial candidate."""

    valid = np.isfinite(command) & np.isfinite(voltage_v) & np.isfinite(rpm)
    u = np.asarray(command[valid], dtype=float)
    voltage = np.asarray(voltage_v[valid], dtype=float)
    measured = np.asarray(rpm[valid], dtype=float)
    if measured.size < 7:
        raise ValueError("at least seven steady points are required")
    voltage_delta = voltage - reference_voltage_v
    design = np.column_stack(
        (
            np.ones_like(u),
            u,
            np.square(u),
            voltage_delta,
            voltage_delta * u,
            voltage_delta * np.square(u),
        )
    )
    coefficients, *_ = np.linalg.lstsq(design, measured, rcond=None)
    return CenteredVoltagePolynomialCommandRpmMap(
        reference_voltage_v=reference_voltage_v,
        intercept_rpm=float(coefficients[0]),
        command_linear=float(coefficients[1]),
        command_quadratic=float(coefficients[2]),
        voltage_linear=float(coefficients[3]),
        voltage_command_linear=float(coefficients[4]),
        voltage_command_quadratic=float(coefficients[5]),
        design_condition_number=float(np.linalg.cond(design)),
    )


def rpm_to_rad_s(rpm: FloatArray) -> FloatArray:
    """Convert revolutions per minute to radians per second."""

    return np.asarray(rpm, dtype=float) * (2.0 * math.pi / 60.0)


def fit_zero_intercept(explanatory: FloatArray, response: FloatArray) -> float:
    """Fit ``response = coefficient * explanatory`` without an intercept."""

    valid = np.isfinite(explanatory) & np.isfinite(response)
    x = np.asarray(explanatory[valid], dtype=float)
    y = np.asarray(response[valid], dtype=float)
    denominator = float(np.dot(x, x))
    if denominator <= 0.0:
        raise ValueError("zero-intercept regression requires non-zero input")
    return float(np.dot(x, y) / denominator)


def fit_wrench_coefficients(
    rpm: FloatArray,
    thrust_n: FloatArray,
    reaction_torque_nm: FloatArray,
) -> WrenchCoefficients:
    """Fit ``T=Kf*w²``, ``Q=Kq*w²``, and two equivalent ``Ktau`` forms."""

    omega_squared = np.square(rpm_to_rad_s(rpm))
    k_f = fit_zero_intercept(omega_squared, thrust_n)
    k_q = fit_zero_intercept(omega_squared, reaction_torque_nm)
    k_tau_direct = fit_zero_intercept(thrust_n, reaction_torque_nm)
    return WrenchCoefficients(
        thrust_per_rad_s_squared=k_f,
        torque_per_rad_s_squared=k_q,
        torque_per_thrust_direct=k_tau_direct,
        torque_per_thrust_ratio=k_q / k_f,
    )


def predict_wrench(
    rpm: FloatArray,
    coefficients: WrenchCoefficients,
) -> tuple[FloatArray, FloatArray]:
    """Predict non-negative thrust and reaction-torque magnitudes."""

    omega_squared = np.square(rpm_to_rad_s(rpm))
    return (
        coefficients.thrust_per_rad_s_squared * omega_squared,
        coefficients.torque_per_rad_s_squared * omega_squared,
    )


def _validate_dynamic_inputs(
    time_s: FloatArray,
    rpm_setpoint: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    time = np.asarray(time_s, dtype=float)
    target = np.asarray(rpm_setpoint, dtype=float)
    if time.ndim != 1 or target.ndim != 1 or time.size != target.size:
        raise ValueError("time and RPM setpoint must be equal-length vectors")
    if time.size < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("time must contain at least two strictly increasing samples")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(target)):
        raise ValueError("dynamic inputs must be finite")
    return time, target


def _step_hold_delayed(
    time_s: FloatArray,
    values: FloatArray,
    delay_s: float,
) -> FloatArray:
    query = time_s - delay_s
    indices = np.searchsorted(time_s, query, side="right") - 1
    indices = np.maximum(indices, 0)
    return values[indices]


def simulate_first_order(
    time_s: FloatArray,
    rpm_setpoint: FloatArray,
    time_constant_s: float,
    initial_rpm: float,
) -> FloatArray:
    """Exact ZOH discretization of a continuous first-order motor model.

    The command at sample ``k-1`` is held over ``[t[k-1], t[k])``.
    """

    if time_constant_s <= 0.0:
        raise ValueError("time constant must be positive")
    time, target = _validate_dynamic_inputs(time_s, rpm_setpoint)
    predicted = np.empty_like(target)
    predicted[0] = initial_rpm
    for index, delta_t in enumerate(np.diff(time), start=1):
        alpha = math.exp(-float(delta_t) / time_constant_s)
        predicted[index] = (
            alpha * predicted[index - 1] + (1.0 - alpha) * target[index - 1]
        )
    return predicted


def simulate_first_order_dead_time(
    time_s: FloatArray,
    rpm_setpoint: FloatArray,
    time_constant_s: float,
    dead_time_s: float,
    initial_rpm: float,
) -> FloatArray:
    """First-order-plus-dead-time model with step-held delayed input."""

    if dead_time_s < 0.0:
        raise ValueError("dead time must be non-negative")
    time, target = _validate_dynamic_inputs(time_s, rpm_setpoint)
    delayed = _step_hold_delayed(time, target, dead_time_s)
    return simulate_first_order(time, delayed, time_constant_s, initial_rpm)


def simulate_second_order(
    time_s: FloatArray,
    rpm_setpoint: FloatArray,
    first_time_constant_s: float,
    second_time_constant_s: float,
    initial_rpm: float,
) -> FloatArray:
    """Exact ZOH simulation of two cascaded monotonic first-order lags."""

    if first_time_constant_s <= 0.0 or second_time_constant_s <= 0.0:
        raise ValueError("both second-order time constants must be positive")
    time, target = _validate_dynamic_inputs(time_s, rpm_setpoint)
    fast, slow = sorted((first_time_constant_s, second_time_constant_s))
    system = np.asarray(
        [
            [-1.0 / fast, 0.0],
            [1.0 / slow, -1.0 / slow],
        ],
        dtype=float,
    )
    input_vector = np.asarray([[1.0 / fast], [0.0]], dtype=float)
    state = np.asarray([initial_rpm, initial_rpm], dtype=float)
    predicted = np.empty_like(target)
    predicted[0] = initial_rpm
    discretizations: dict[float, tuple[FloatArray, FloatArray]] = {}
    for index, delta_t in enumerate(np.diff(time), start=1):
        cache_key = round(float(delta_t), 12)
        if cache_key not in discretizations:
            augmented = np.zeros((3, 3), dtype=float)
            augmented[:2, :2] = system
            augmented[:2, 2:] = input_vector
            discrete = expm(augmented * cache_key)
            discretizations[cache_key] = (
                np.asarray(discrete[:2, :2], dtype=float),
                np.asarray(discrete[:2, 2], dtype=float),
            )
        transition, control = discretizations[cache_key]
        state = transition @ state + control * target[index - 1]
        predicted[index] = state[1]
    return predicted


def simulate_dynamic_model(
    model: DynamicModelName,
    time_s: FloatArray,
    rpm_setpoint: FloatArray,
    parameters: Sequence[float],
    initial_rpm: float,
) -> FloatArray:
    """Dispatch a named dynamic model using its canonical parameter order."""

    if model == "first_order":
        return simulate_first_order(
            time_s, rpm_setpoint, float(parameters[0]), initial_rpm
        )
    if model == "first_order_dead_time":
        return simulate_first_order_dead_time(
            time_s,
            rpm_setpoint,
            float(parameters[0]),
            float(parameters[1]),
            initial_rpm,
        )
    if model == "second_order":
        return simulate_second_order(
            time_s,
            rpm_setpoint,
            float(parameters[0]),
            float(parameters[0]),
            initial_rpm,
        )
    raise ValueError(f"unsupported dynamic model: {model}")


def _candidate_definition(
    model: DynamicModelName,
) -> tuple[tuple[str, ...], FloatArray, FloatArray, FloatArray]:
    if model == "first_order":
        return (
            ("time_constant_s",),
            np.asarray([0.10]),
            np.asarray([0.005]),
            np.asarray([0.80]),
        )
    if model == "first_order_dead_time":
        return (
            ("time_constant_s", "dead_time_s"),
            np.asarray([0.08, 0.02]),
            np.asarray([0.005, 0.0]),
            np.asarray([0.80, 0.20]),
        )
    if model == "second_order":
        return (
            ("repeated_time_constant_s",),
            np.asarray([0.06]),
            np.asarray([0.005]),
            np.asarray([0.80]),
        )
    raise ValueError(f"unsupported dynamic model: {model}")


def predict_dynamic_series(
    fit: DynamicFit,
    series: DynamicSeries,
) -> FloatArray:
    """Run a fitted model over one series."""

    return simulate_dynamic_model(
        fit.model,
        series.time_s,
        series.rpm_setpoint,
        fit.parameter_values,
        float(series.rpm_measured[0]),
    )


def score_dynamic_fit(
    fit: DynamicFit,
    series: Sequence[DynamicSeries],
) -> RegressionMetrics:
    """Score a fixed dynamic fit over one or more complete runs."""

    measured: list[FloatArray] = []
    predicted: list[FloatArray] = []
    for run in series:
        estimate = predict_dynamic_series(fit, run)
        measured.append(run.rpm_measured[run.evaluation_mask])
        predicted.append(estimate[run.evaluation_mask])
    return regression_metrics(np.concatenate(measured), np.concatenate(predicted))


def fit_dynamic_model(
    model: DynamicModelName,
    series: Sequence[DynamicSeries],
) -> DynamicFit:
    """Fit one dynamic candidate by nonlinear least squares on measured RPM."""

    if not series:
        raise ValueError("at least one dynamic series is required")
    names, initial, lower, upper = _candidate_definition(model)

    def residual(parameters: FloatArray) -> FloatArray:
        blocks: list[FloatArray] = []
        for run in series:
            estimate = simulate_dynamic_model(
                model,
                run.time_s,
                run.rpm_setpoint,
                parameters,
                float(run.rpm_measured[0]),
            )
            blocks.append(
                estimate[run.evaluation_mask]
                - run.rpm_measured[run.evaluation_mask]
            )
        return np.concatenate(blocks)

    delay_grid_resolution_s: float | None = None
    covariance_indices: tuple[int, ...]
    if model == "first_order_dead_time":
        delay_grid_resolution_s = 0.01
        best_result = None
        best_delay = math.nan
        best_rss = math.inf
        for delay in np.arange(0.0, 0.2001, delay_grid_resolution_s):
            optimized_at_delay = least_squares(
                lambda time_constant, selected_delay=float(delay): residual(
                    np.asarray([time_constant[0], selected_delay], dtype=float)
                ),
                np.asarray([initial[0]], dtype=float),
                bounds=(
                    np.asarray([lower[0]], dtype=float),
                    np.asarray([upper[0]], dtype=float),
                ),
                method="trf",
                x_scale="jac",
            )
            candidate_error = residual(
                np.asarray([optimized_at_delay.x[0], delay], dtype=float)
            )
            candidate_rss = float(np.dot(candidate_error, candidate_error))
            if candidate_rss < best_rss:
                best_result = optimized_at_delay
                best_delay = float(delay)
                best_rss = candidate_rss
        if best_result is None:
            raise RuntimeError("dead-time grid search did not produce a candidate")
        optimized = best_result
        values = np.asarray([optimized.x[0], best_delay], dtype=float)
        covariance_indices = (0,)
        optimizer_message = (
            f"{optimized.message}; dead time selected on a "
            f"{delay_grid_resolution_s:g} s grid"
        )
    else:
        optimized = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
        )
        values = np.asarray(optimized.x, dtype=float)
        covariance_indices = tuple(range(len(values)))
        optimizer_message = str(optimized.message)
    errors = residual(values)
    measured = np.concatenate(
        [run.rpm_measured[run.evaluation_mask] for run in series]
    )
    metrics = regression_metrics(measured, measured + errors)
    parameter_count = len(values)
    sample_count = len(errors)
    rss = max(float(np.dot(errors, errors)), np.finfo(float).tiny)
    aic = sample_count * math.log(rss / sample_count) + 2.0 * parameter_count
    correction = (
        2.0 * parameter_count * (parameter_count + 1)
        / (sample_count - parameter_count - 1)
        if sample_count > parameter_count + 1
        else math.inf
    )
    standard_errors = np.full(parameter_count, math.nan, dtype=float)
    if optimized.jac.shape[0] > optimized.jac.shape[1]:
        degrees_of_freedom = optimized.jac.shape[0] - optimized.jac.shape[1]
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                normal_matrix = optimized.jac.T @ optimized.jac
            if np.all(np.isfinite(normal_matrix)):
                covariance = (
                    np.linalg.pinv(normal_matrix, hermitian=True)
                    * rss
                    / degrees_of_freedom
                )
                estimated = np.sqrt(np.maximum(np.diag(covariance), 0.0))
                for index, estimate in zip(
                    covariance_indices, estimated, strict=True
                ):
                    standard_errors[index] = estimate
        except np.linalg.LinAlgError:
            pass
    if delay_grid_resolution_s is not None:
        standard_errors[1] = delay_grid_resolution_s / math.sqrt(12.0)
    return DynamicFit(
        model=model,
        parameter_names=names,
        parameter_values=tuple(float(value) for value in values),
        parameter_standard_errors=tuple(
            float(value) for value in standard_errors
        ),
        metrics=metrics,
        aicc=float(aic + correction),
        optimizer_success=bool(optimized.success),
        optimizer_message=optimizer_message,
    )
