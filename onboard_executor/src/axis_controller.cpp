#include "fftgyro_executor/axis_controller.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace fftgyro
{
namespace
{
constexpr double kRadToDeg = 180.0 / std::numbers::pi;

bool fail(const char* _message, ControlOutput& _output) noexcept
{
  _output.error = _message;
  return false;
}

std::array<double, 4>
relativeRotation(const std::array<double, 4>& _zero, const std::array<double, 4>& _attitude) noexcept
{
  const auto& a = _zero;
  const auto& b = _attitude;
  // conjugate(zero) * attitude: R_body_to_world = R_zero * R_joint.
  return { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3],
           a[0] * b[1] - a[1] * b[0] - a[2] * b[3] + a[3] * b[2],
           a[0] * b[2] + a[1] * b[3] - a[2] * b[0] - a[3] * b[1],
           a[0] * b[3] - a[1] * b[2] + a[2] * b[1] - a[3] * b[0] };
}
}  // namespace

bool normalizeQuaternion(std::array<double, 4>& _quaternion) noexcept
{
  double squared_norm = 0;
  for (const double value : _quaternion) {
    if (!std::isfinite(value)) {
      return false;
    }
    squared_norm += value * value;
  }
  if (!(squared_norm > 0.81 && squared_norm < 1.21)) {
    return false;
  }
  for (double& value : _quaternion) {
    value /= std::sqrt(squared_norm);
  }
  return true;
}

AxisController::AxisController(const Config& _hardware, const PidConfig& _pid) noexcept
  : hardware_(_hardware), pid_(_pid)
{
}

void AxisController::resetBeforeStart() noexcept
{
  integral_ = continuous_ = previous_wrapped_ = saturation_s_ = 0;
  initialized_ = derivative_initialized_ = false;
  previous_feedback_boot_ms_ = 0;
  previous_rate_deg_s_ = filtered_accel_deg_s2_ = 0;
}

bool AxisController::readJoint(const Feedback& _feedback, std::int64_t _now_ns, ControlOutput& _output) noexcept
{
  _output = {};
  if (
    !_feedback.valid || _now_ns < _feedback.received_ns ||
    (_now_ns - _feedback.received_ns) * 1e-9 > pid_.feedback_timeout_s) {
    return fail("feedback missing/stale", _output);
  }
  auto zero = pid_.zero_quaternion;
  auto attitude = _feedback.quaternion;
  if (!normalizeQuaternion(zero) || !normalizeQuaternion(attitude)) {
    return fail("invalid quaternion", _output);
  }
  const auto rotation = relativeRotation(zero, attitude);
  const int axis = pid_.axis == "PITCH" ? 2 : 3;
  const double twist_norm = std::hypot(rotation[0], rotation[axis]);
  if (twist_norm <= 1e-6) {
    return fail("joint-angle singularity", _output);
  }

  // ±180°をまたいでも、指定した側へ連続して進める。
  _output.wrapped_deg = std::remainder(2 * std::atan2(rotation[axis], rotation[0]) * kRadToDeg, 360.0);
  continuous_ =
    initialized_ ? continuous_ + std::remainder(_output.wrapped_deg - previous_wrapped_, 360.0) : _output.wrapped_deg;
  initialized_ = true;
  previous_wrapped_ = _output.wrapped_deg;
  _output.angle_deg = pid_.zero_angle_deg + continuous_;
  _output.swing_deg = 2 * std::acos(std::clamp(twist_norm, 0.0, 1.0)) * kRadToDeg;

  // R_relative^T * joint_axis: body gyroを固定された関節軸へ投影する。
  const auto [w, x, y, z] = rotation;
  const std::array<double, 3> joint_axis =
    axis == 2 ? std::array<double, 3>{ 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x) } :
                std::array<double, 3>{ 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y) };
  for (int i = 0; i < 3; ++i) {
    const double rate = _feedback.body_rate_deg_s[i];
    if (!std::isfinite(rate) || std::abs(rate) > pid_.rate_limit_deg_s) {
      return fail("body rate limit", _output);
    }
    _output.rate_deg_s += joint_axis[i] * rate;
  }
  if (_output.angle_deg < pid_.angle_min_deg || _output.angle_deg > pid_.angle_max_deg) {
    return fail("joint angle limit", _output);
  }
  if (_output.swing_deg > pid_.swing_limit_deg) {
    return fail("non-target attitude limit", _output);
  }
  if (std::abs(_output.rate_deg_s) > pid_.rate_limit_deg_s) {
    return fail("joint rate limit", _output);
  }
  return true;
}

bool AxisController::update(
  const Feedback& _feedback,
  std::int64_t _now_ns,
  const Reference& _reference,
  double _collective,
  double _dt,
  const std::array<float, 4>& _previous,
  ControlOutput& _output,
  ControlPolicy _policy) noexcept
{
  _output = {};
  if (
    !std::isfinite(_dt) || _dt <= 0 || _dt > 0.02 || !std::isfinite(_collective) || _collective < 0 ||
    _collective > hardware_.motor_max || !std::isfinite(_reference.angle_deg) ||
    !std::isfinite(_reference.rate_deg_s) || !std::isfinite(_reference.external)) {
    return fail("invalid control input/timing", _output);
  }
  if (!readJoint(_feedback, _now_ns, _output)) {
    return false;
  }
  const double error = _reference.angle_deg - _output.angle_deg;
  if (
    _reference.angle_deg < pid_.angle_min_deg || _reference.angle_deg > pid_.angle_max_deg ||
    (_policy.check_tracking && std::abs(error) > pid_.tracking_limit_deg)) {
    return fail("tracking error limit", _output);
  }

  // 送信可能な4指令の範囲は常に守り、追加制限は設定で切り替える。
  const bool limited = pid_.additional_limits_enabled;
  double lower = limited ? -pid_.differential_limit : -1.0;
  double upper = limited ? pid_.differential_limit : 1.0;
  double slew_lower = -1.0, slew_upper = 1.0;
  const double floor = _collective < hardware_.motor_min ? 0.0 : hardware_.motor_min;
  const double step = pid_.motor_slew_time_s > 0 ? _dt / pid_.motor_slew_time_s : 1.0;
  for (int i = 0; i < 4; ++i) {
    if (!std::isfinite(_previous[i])) {
      return fail("nonfinite previous motor", _output);
    }
    const double a = (floor - _collective) / hardware_.axis_mix[i];
    const double b = (hardware_.motor_max - _collective) / hardware_.axis_mix[i];
    lower = std::max(lower, std::min(a, b));
    upper = std::min(upper, std::max(a, b));
    const double c = (_previous[i] - step - _collective) / hardware_.axis_mix[i];
    const double d = (_previous[i] + step - _collective) / hardware_.axis_mix[i];
    slew_lower = std::max(slew_lower, std::min(c, d));
    slew_upper = std::min(slew_upper, std::max(c, d));
  }
  const double static_lower = lower, static_upper = upper;
  lower = std::max(lower, slew_lower);
  upper = std::min(upper, slew_upper);
  if (lower > upper) {
    return fail("no feasible differential under motor/slew limits", _output);
  }

  const bool cascade = pid_.law == "ANGLE_RATE_PID";
  double actuator_gain_scale = 1.0;
  if (pid_.cascade.collective_gain_compensation) {
    const auto& rate = pid_.cascade;
    if (!cascade || pid_.axis != "PITCH" || !std::isfinite(rate.thrust_curve_linear) ||
        !std::isfinite(rate.thrust_curve_quadratic) || rate.thrust_curve_linear <= 0 ||
        rate.thrust_curve_quadratic < 0 || !std::isfinite(rate.effectiveness_reference_collective) ||
        !(rate.thrust_curve_command_min > 0 && rate.thrust_curve_command_max <= 1 &&
          rate.thrust_curve_command_min < rate.thrust_curve_command_max &&
          rate.thrust_curve_command_max >= hardware_.motor_max &&
          rate.effectiveness_reference_collective >= rate.thrust_curve_command_min &&
          rate.effectiveness_reference_collective <= hardware_.motor_max)) {
      return fail("invalid actuator gain model", _output);
    }
    const auto slope = [&](double collective) {
      return rate.thrust_curve_linear + 2 * rate.thrust_curve_quadratic * collective;
    };
    // Below the measured reference, preserve the base controller; never boost startup.
    _output.actuator_gain_floor_active = _collective < rate.effectiveness_reference_collective;
    actuator_gain_scale = slope(rate.effectiveness_reference_collective) /
                          slope(std::max(_collective, rate.effectiveness_reference_collective));
    if (!std::isfinite(slope(hardware_.motor_max)) || !std::isfinite(actuator_gain_scale) ||
        actuator_gain_scale < 0.2 || actuator_gain_scale > 1) {
      return fail("invalid actuator gain compensation", _output);
    }
  }
  _output.actuator_gain_scale = actuator_gain_scale;
  double integral_step;
  if (cascade) {
    const auto& rate = pid_.cascade;
    constexpr double rad = std::numbers::pi / 180.0;
    // One-axis quaternion P law; bounded rate setpoint, not a motor slew limiter.
    _output.rate_target_deg_s = std::clamp(rate.trajectory_rate_feedforward * _reference.rate_deg_s +
                                          2 * std::sin(error * rad / 2) * rate.attitude_kp / rad,
                                          -rate.target_rate_limit_deg_s, rate.target_rate_limit_deg_s);
    _output.rate_error_deg_s = _output.rate_target_deg_s - _output.rate_deg_s;
    if (!derivative_initialized_) {
      derivative_initialized_ = true;
      previous_feedback_boot_ms_ = _feedback.boot_ms;
      previous_rate_deg_s_ = _output.rate_deg_s;
    }
    else if (_feedback.boot_ms != previous_feedback_boot_ms_) {
      const auto elapsed_ms = static_cast<std::int32_t>(_feedback.boot_ms - previous_feedback_boot_ms_);
      if (elapsed_ms <= 0) return fail("nonmonotonic derivative feedback", _output);
      const double sample_dt = elapsed_ms * .001;
      if (sample_dt <= pid_.feedback_timeout_s) {
        const double accel = (_output.rate_deg_s - previous_rate_deg_s_) / sample_dt;
        const double alpha = 1 - std::exp(-2 * std::numbers::pi * rate.derivative_cutoff_hz * sample_dt);
        filtered_accel_deg_s2_ += alpha * (accel - filtered_accel_deg_s2_);
      }
      else {
        filtered_accel_deg_s2_ = 0;  // Do not differentiate across an unobserved gap.
      }
      previous_feedback_boot_ms_ = _feedback.boot_ms;
      previous_rate_deg_s_ = _output.rate_deg_s;
    }
    _output.angular_accel_filtered_deg_s2 = filtered_accel_deg_s2_;
    _output.p = rate.torque_to_delta * rate.kp * _output.rate_error_deg_s * rad;
    _output.d = -rate.torque_to_delta * rate.kd * filtered_accel_deg_s2_ * rad;
    const double i_factor = std::max(0.0, 1 - std::pow(_output.rate_error_deg_s / 400.0, 2));
    integral_step = rate.torque_to_delta * rate.ki * _output.rate_error_deg_s * rad * i_factor * _dt;
  }
  else {
    _output.p = pid_.kp * error;
    _output.d = -pid_.kd * _output.rate_deg_s;  // 目標変更の微分キックを避ける。
    integral_step = pid_.ki * error * _dt;
  }
  previous_integral_ = integral_;
  const double accumulated = integral_ + integral_step;
  const double candidate = limited ? std::clamp(accumulated, -pid_.integral_limit, pid_.integral_limit) : accumulated;
  const double virtual_wanted = _output.p + candidate + _output.d;
  const double wanted = actuator_gain_scale * virtual_wanted + _reference.external;
  if (!std::isfinite(wanted)) return fail("nonfinite controller demand", _output);
  const bool can_integrate = (wanted >= lower && wanted <= upper) || (wanted > upper && candidate < integral_) ||
                             (wanted < lower && candidate > integral_);
  if (_policy.enable_integral && _collective > 0 && _collective >= hardware_.motor_min && can_integrate) {
    integral_ = candidate;
  }
  _output.i = integral_;
  const double virtual_requested = _output.p + _output.i + _output.d;
  const double requested = actuator_gain_scale * virtual_requested + _reference.external;
  if (!std::isfinite(requested)) return fail("nonfinite controller demand", _output);
  _output.virtual_delta_requested = virtual_requested;
  _output.delta_requested = requested;
  _output.delta_lower = lower;
  _output.delta_upper = upper;
  _output.delta = std::clamp(requested, lower, upper);
  const double bounded = std::clamp(requested, static_lower, static_upper);
  _output.authority_limited = std::abs(requested - bounded) > 1e-10;
  _output.slew_limited = std::abs(bounded - _output.delta) > 1e-10;
  _output.actuator_model_in_range = pid_.cascade.collective_gain_compensation;
  for (int i = 0; i < 4; ++i) {
    _output.requested[i] = _collective + hardware_.axis_mix[i] * requested;
    _output.motor[i] = static_cast<float>(_collective + hardware_.axis_mix[i] * _output.delta);
    _output.actuator_model_in_range &=
      _output.motor[i] >= pid_.cascade.thrust_curve_command_min - 1e-7 &&
      _output.motor[i] <= pid_.cascade.thrust_curve_command_max + 1e-7;
    _output.saturated |= _output.requested[i] < floor - 1e-10 || _output.requested[i] > hardware_.motor_max + 1e-10;
  }
  saturation_s_ = _policy.check_saturation_timeout &&
                    (_output.saturated || (cascade && _output.authority_limited)) ? saturation_s_ + _dt : 0;
  if (_policy.check_saturation_timeout && saturation_s_ > pid_.saturation_timeout_s) {
    return fail("sustained saturation", _output);
  }
  return true;
}
}  // namespace fftgyro
