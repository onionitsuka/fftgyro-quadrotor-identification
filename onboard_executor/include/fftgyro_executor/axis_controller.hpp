#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "fftgyro_executor/executor.hpp"

namespace fftgyro
{
struct RateLoopConfig
{
  double attitude_kp{4.0};
  double kp{0.14}, ki{0.30}, kd{0.0};  // PX4-like torque units; rate errors in rad/s.
  double torque_to_delta{0.7071067811865476};
  double target_rate_limit_deg_s{30.0};
  double derivative_cutoff_hz{10.0};
  double trajectory_rate_feedforward{0.0};  // 0 preserves legacy behavior; 1 tracks the analytic trajectory rate.
  // Normalize the motor-command differential by the local slope of the
  // measured thrust curve F(u)=linear*u+quadratic*u^2.  The coefficients may
  // share a CONSTANT force scale. This does not compensate changing voltage,
  // motor dynamics, or aerodynamic/rig effects. Pitch only.
  bool collective_gain_compensation{false};
  double effectiveness_reference_collective{0.1};
  double thrust_curve_linear{1.0};
  double thrust_curve_quadratic{0.0};
  double thrust_curve_command_min{0.1}, thrust_curve_command_max{0.8};
};

// Angle: deg. Rate: deg/s. Quaternion: Hamilton wxyz.
struct PidConfig
{
  std::string axis;                        // PITCH or YAW; no other controller exists in a session.
  std::string law{"ANGLE_PID"};
  RateLoopConfig cascade;
  bool additional_limits_enabled{ true };  // 差動・積分の追加上限。
  double motor_slew_time_s{ 0.2 };         // 0→1／1→0の最短時間。0なら変化量制限なし。
  std::array<double, 4> zero_quaternion{ 1, 0, 0, 0 };
  double zero_angle_deg = {};
  double kp = {}, ki = {}, kd = {}, integral_limit = {}, differential_limit = {};
  double angle_min_deg = {}, angle_max_deg = {}, rate_limit_deg_s = {};
  // Validation of commanded reference slew only. Measured body/joint-rate stop
  // remains governed by rate_limit_deg_s in AxisController.
  double reference_rate_limit_deg_s = {};
  double swing_limit_deg = {}, tracking_limit_deg = {};
  double feedback_timeout_s{ 0.1 };
  double saturation_timeout_s{ 2.0 };
};

struct Reference
{
  double angle_deg = {}, external = {};
  double rate_deg_s = {};  // Analytic trajectory rate; external stays in physical motor-command units.
};

struct ControlPolicy
{
  bool check_tracking{ true };
  bool enable_integral{ true };
  bool check_saturation_timeout{ true };  // Startup may clip without treating unavailable low-thrust authority as a fault.
};

struct ControlOutput
{
  std::array<float, 4> motor = {};
  std::array<double, 4> requested = {};
  double angle_deg = {}, wrapped_deg = {}, rate_deg_s = {}, swing_deg = {};
  double p = {}, i = {}, d = {}, delta = {};
  double rate_target_deg_s = {}, rate_error_deg_s = {}, angular_accel_filtered_deg_s2 = {};
  double virtual_delta_requested = {}, actuator_gain_scale{1.0};
  bool actuator_gain_floor_active = {}, actuator_model_in_range = {};
  double delta_requested = {}, delta_lower = {}, delta_upper = {};
  bool authority_limited = {};
  bool saturated = {};     // モーター要求の出力上下限超過。
  bool slew_limited = {};  // 変化速度制限。飽和タイマーには含めない。
  const char* error = nullptr;
};

bool normalizeQuaternion(std::array<double, 4>& _quaternion) noexcept;

/** @brief 対象軸の角度PIDまたは角度P→角速度PID。対象外軸は停止監視だけ行う。 */
class AxisController
{
public:
  // 設定は制御器より長く保持し、運転中には変更しない。
  explicit AxisController(const Config& _hardware, const PidConfig& _pid) noexcept;
  AxisController(const AxisController&) = delete;
  AxisController& operator=(const AxisController&) = delete;
  AxisController(AxisController&&) = delete;
  AxisController& operator=(AxisController&&) = delete;
  // 1周期分を計算する。falseなら出力を送らず、errorに従って停止する。
  bool update(
    const Feedback& _feedback,
    std::int64_t _now_ns,
    const Reference& _reference,
    double _collective,
    double _dt,
    const std::array<float, 4>& _previous,
    ControlOutput& _output,
    ControlPolicy _policy = {}) noexcept;

  // 関節角・角速度の計測。積分器・飽和時間は進めない。
  bool readJoint(const Feedback& _feedback, std::int64_t _now_ns, ControlOutput& _output) noexcept;
  // Only before the first motor command, after capturing a new session reference.
  void resetBeforeStart() noexcept;
  void rejectIntegralStep() noexcept { integral_ = previous_integral_; }

private:
  const Config& hardware_;
  const PidConfig& pid_;
  double integral_ = {};
  double previous_integral_ = {};
  double continuous_ = {};
  double previous_wrapped_ = {};
  double saturation_s_ = {};
  bool initialized_ = {};
  bool derivative_initialized_ = {};
  std::uint32_t previous_feedback_boot_ms_ = {};
  double previous_rate_deg_s_ = {}, filtered_accel_deg_s2_ = {};
};
}  // namespace fftgyro
