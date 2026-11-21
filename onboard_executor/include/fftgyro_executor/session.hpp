#pragma once

#include <array>
#include <filesystem>
#include <string>
#include <vector>
#include <memory>

#include "fftgyro_executor/axis_controller.hpp"
#include "fftgyro_executor/collective_ramp.hpp"

namespace fftgyro
{

struct StartupConfig
{
  std::string ramp_shape{"LINEAR"};
  double minimum_duration_s{0};
  bool capture_reference{false};
  bool encoder_absolute_anchor{false};  // zero quaternion has the measured absolute encoder angle.
  double reference_dwell_s{0.3};
  double reference_timeout_s{5.0};
  double reference_angle_tolerance_deg{1.0};
  double reference_swing_tolerance_deg{2.0};
  double reference_rate_tolerance_deg_s{5.0};
  double integral_enable_collective{ 0.1 };
  bool control_ready_verified = {};  // 最初の基準指令で姿勢制御が可能かを実機確認。
};

struct Segment
{
  std::string condition_id, segment_id, hold_id, kind;
  double duration_s = {}, start_deg = {}, end_deg = {};
  double amplitude_deg = {}, frequency_hz = {}, phase_rad = {};
  double external_amplitude = {}, external_frequency_hz = {}, external_phase_rad = {};
  double fade_s = {};
  double collective = {};
  bool collective_step_up = false;  // Fixed HOLD transitions only; never step down.
  // [C, P, R, Y].  C is the common command; P/R/Y are signed
  // differentials. fixed_amplitude is used only by a fixed-input SINE.
  std::array<double, 4> fixed_input = {};
  std::array<double, 4> fixed_amplitude = {};
  double analysis_start_s = {}, analysis_end_s = {};
  bool require_settled = {};
  bool preparation_only = {};  // Short operational settle, never an analysis window.
};

struct SessionConfig
{
  Config hardware;
  PidConfig pid;
  StartupConfig startup;
  double shutdown_minimum_duration_s{ 2 }, shutdown_external_fade_s{ 0.5 };
  std::string session_id, parent_series_id, role;
  std::string snapshot_json;
  int repetition = {};
  double wind_m_s = {}, heading_deg = {}, fixed_pitch_deg = {};
  double maximum_duration_s = {}, transition_rate_deg_s = {};
  double settled_error_deg = {}, settled_rate_deg_s = {}, settled_dwell_s = {};
  double settled_window_s{ 0.15 };
  double maximum_hold_s{ 15 };
  bool runtime_qualified = {};
  std::string control_mode = "AXIS_PID";
  bool mechanically_locked_verified = {};
  bool motor_mapping_verified = {}, feedback_frame_verified = {};
  // Rows are motors 1..4; columns are [C, P, R, Y].
  std::array<std::array<double, 4>, 4> fixed_input_mix = {};
  std::vector<Segment> segments;
  std::shared_ptr<SessionConfig> coupled_yaw;
  std::array<double,4> rig_base_quaternion{1,0,0,0};
  bool coupled_anchor_resolved{false};
};

SessionConfig loadSession(const std::filesystem::path& _path);
void validateSession(const SessionConfig& _config);
void validatePropsOffSession(const SessionConfig& _config);
void validateFixedCommissioning(const SessionConfig& _config);
void validatePitchHoldCommissioning(const SessionConfig& _config);
void validateYawHoldCommissioning(const SessionConfig& _config);
Reference segmentReference(const Segment& _segment, double _time_s) noexcept;
std::array<double, 4> fixedInputReference(const Segment& _segment, double _time_s) noexcept;
std::array<float, 4> fixedMotorCommands(
  const std::array<std::array<double, 4>, 4>& _mix,
  const std::array<double, 4>& _input) noexcept;

}  // namespace fftgyro
