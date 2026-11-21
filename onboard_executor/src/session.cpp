#include "fftgyro_executor/session.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numbers>
#include <set>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace fftgyro
{
namespace
{
constexpr double kRadToDeg = 180.0 / std::numbers::pi;
using Json = nlohmann::json;
void check(bool _ok, const std::string& _message)
{
  if (!_ok) {
    throw std::runtime_error(_message);
  }
}
bool isFinite(double _value)
{
  return std::isfinite(_value);
}
bool isValidId(const std::string& _id)
{
  return !_id.empty() && _id.size() <= 128 &&
         std::all_of(
           _id.begin(),
           _id.end(),
           [](unsigned char c)
           {
             return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' ||
                    c == '_' || c == '.';
           });
}
}  // namespace

void updateAttitudeFeedback(
  Feedback& _feedback,
  const mavlink_message_t& _message,
  std::uint8_t _target_system,
  std::uint8_t _target_component,
  std::int64_t _received_ns) noexcept
{
  if (
    _message.sysid != _target_system || _message.compid != _target_component ||
    _message.msgid != MAVLINK_MSG_ID_ATTITUDE_QUATERNION || _feedback.fault) {
    return;
  }
  mavlink_attitude_quaternion_t attitude{};
  mavlink_msg_attitude_quaternion_decode(&_message, &attitude);
  if (_feedback.valid) {
    const auto elapsed = static_cast<std::int32_t>(attitude.time_boot_ms - _feedback.boot_ms);
    if (elapsed == 0) {
      return;
    }
    if (elapsed < 0) {
      _feedback.fault = true;
      _feedback.valid = false;
      return;
    }
  }
  _feedback.quaternion = { attitude.q1, attitude.q2, attitude.q3, attitude.q4 };
  _feedback.body_rate_deg_s = { attitude.rollspeed * kRadToDeg,
                                attitude.pitchspeed * kRadToDeg,
                                attitude.yawspeed * kRadToDeg };
  _feedback.boot_ms = attitude.time_boot_ms;
  _feedback.received_ns = _received_ns;
  _feedback.valid = true;
}

Reference segmentReference(const Segment& _segment, double _time_s) noexcept
{
  _time_s = std::clamp(_time_s, 0.0, _segment.duration_s);
  Reference result{ _segment.start_deg, 0 };
  if (_segment.kind == "MOVE") {
    result.angle_deg += (_segment.end_deg - _segment.start_deg) * quintic(_time_s / _segment.duration_s);
    result.rate_deg_s = (_segment.end_deg - _segment.start_deg) / _segment.duration_s *
                        quinticDerivative(_time_s / _segment.duration_s);
  }
  if (_segment.kind == "SCAN") {
    // Each end has a one-second half-cosine velocity ramp.
    const double velocity = (_segment.end_deg - _segment.start_deg) / (_segment.duration_s - 1.0);
    double distance;
    if (_time_s < 1.0) {
      distance = (_time_s - std::sin(std::numbers::pi * _time_s) / std::numbers::pi) / 2.0;
      result.rate_deg_s = velocity * (1 - std::cos(std::numbers::pi * _time_s)) / 2;
    }
    else if (_time_s <= _segment.duration_s - 1.0) {
      distance = _time_s - 0.5;
      result.rate_deg_s = velocity;
    }
    else {
      const double tail = _segment.duration_s - _time_s;
      distance = _segment.duration_s - 1.0 - (tail - std::sin(std::numbers::pi * tail) / std::numbers::pi) / 2.0;
      result.rate_deg_s = velocity * (1 - std::cos(std::numbers::pi * tail)) / 2;
    }
    result.angle_deg += velocity * distance;
  }
  if (_segment.kind == "SINE") {
    const double envelope =
      _segment.fade_s == 0 ? 1.0 : quintic(std::min(_time_s, _segment.duration_s - _time_s) / _segment.fade_s);
    result.angle_deg += envelope * _segment.amplitude_deg *
                        std::sin(2.0 * std::numbers::pi * _segment.frequency_hz * _time_s + _segment.phase_rad);
    const double envelope_rate = _segment.fade_s == 0 ? 0.0 :
      (_time_s < _segment.duration_s / 2 ? 1.0 : -1.0) / _segment.fade_s *
      quinticDerivative(std::min(_time_s, _segment.duration_s - _time_s) / _segment.fade_s);
    const double omega = 2 * std::numbers::pi * _segment.frequency_hz;
    const double phase = omega * _time_s + _segment.phase_rad;
    result.rate_deg_s = _segment.amplitude_deg *
                       (envelope_rate * std::sin(phase) + envelope * omega * std::cos(phase));
    result.external =
      envelope * _segment.external_amplitude *
      std::sin(2.0 * std::numbers::pi * _segment.external_frequency_hz * _time_s + _segment.external_phase_rad);
  }
  return result;
}

std::array<double, 4> fixedInputReference(const Segment& _segment, double _time_s) noexcept
{
  _time_s = std::clamp(_time_s, 0.0, _segment.duration_s);
  auto result = _segment.fixed_input;
  if (_segment.kind != "SINE") return result;
  const double edge = std::min(_time_s, _segment.duration_s - _time_s);
  const double envelope = _segment.fade_s == 0 ? 1.0 : quintic(std::clamp(edge / _segment.fade_s, 0.0, 1.0));
  const double wave = envelope * std::sin(
    2.0 * std::numbers::pi * _segment.frequency_hz * _time_s + _segment.phase_rad);
  for (std::size_t i = 0; i < result.size(); ++i) result[i] += _segment.fixed_amplitude[i] * wave;
  return result;
}

std::array<float, 4> fixedMotorCommands(
  const std::array<std::array<double, 4>, 4>& _mix,
  const std::array<double, 4>& _input) noexcept
{
  std::array<float, 4> result{};
  for (std::size_t motor = 0; motor < result.size(); ++motor) {
    double value = 0;
    for (std::size_t input = 0; input < _input.size(); ++input) value += _mix[motor][input] * _input[input];
    result[motor] = static_cast<float>(value);
  }
  return result;
}

static SessionConfig parseSession(const Json& json)
{
  check(json.at("schema").get<std::string>() == "fftgyro-axis-session-v1", "wrong session schema");
  SessionConfig config;
  config.snapshot_json = json.dump(2);
  config.session_id = json.at("session_id");
  config.parent_series_id = json.at("parent_series_id");
  config.role = json.at("role");
  config.repetition = json.at("repetition");
  config.wind_m_s = json.at("wind_m_s");
  config.heading_deg = json.at("heading_deg");
  config.fixed_pitch_deg = json.at("fixed_pitch_deg");
  config.runtime_qualified = json.value("runtime_qualified", false);
  config.control_mode = json.value("control_mode", std::string("AXIS_PID"));
  config.maximum_duration_s = json.at("maximum_duration_s");
  config.transition_rate_deg_s = json.at("transition_rate_deg_s");
  const auto& hardware = json.at("hardware");
  config.mechanically_locked_verified = hardware.value("mechanically_locked_verified", false);
  config.motor_mapping_verified = hardware.value("motor_mapping_verified", false);
  config.feedback_frame_verified = hardware.value("feedback_frame_verified", false);
  config.hardware.run_name = config.session_id;
  config.hardware.uart_device = hardware.at("uart_device");
  config.hardware.target_system = hardware.at("target_system").get<std::uint8_t>();
  config.hardware.target_component = hardware.at("target_component").get<std::uint8_t>();
  config.hardware.axis_mix = hardware.at("axis_mix").get<std::array<double, 4>>();
  if (config.control_mode == "FIXED_INPUT") {
    config.fixed_input_mix = hardware.at("fixed_input_mix").get<std::array<std::array<double, 4>, 4>>();
  }
  config.hardware.motor_min = hardware.at("motor_min");
  config.hardware.motor_max = hardware.at("motor_max");
  config.hardware.startup_ramp_rate = hardware.at("startup_ramp_rate");
  config.hardware.normal_ramp_rate = hardware.at("normal_ramp_rate");
  config.hardware.abort_ramp_rate = hardware.at("abort_ramp_rate");
  config.hardware.log_directory = hardware.at("log_directory").get<std::string>();
  check(hardware.at("non_target_pid_enabled").get<bool>() == false, "non-target PID must be disabled");
  if (config.runtime_qualified || json.contains("coupled_yaw")) {
    for (const auto* flag : { "motor_mapping_verified",
                              "feedback_frame_verified",
                              "stop_verified",
                              "rpm_logging_verified",
                              "load_logging_verified",
                              "sync_verified" }) {
      check(hardware.at(flag).get<bool>(), std::string("missing qualification: ") + flag);
    }
    if (config.control_mode == "FIXED_INPUT") {
      check(hardware.at("fixed_input_mapping_verified").get<bool>(),
            "missing qualification: fixed_input_mapping_verified");
    }
  }
  const auto& pid = json.at("pid");
  config.pid.axis = pid.at("axis");
  config.pid.law = pid.value("law", std::string("ANGLE_PID"));
  const auto cascade = pid.value("cascade", Json::object());
  config.pid.cascade.attitude_kp = cascade.value("attitude_kp", 4.0);
  config.pid.cascade.kp = cascade.value("kp", 0.14);
  config.pid.cascade.ki = cascade.value("ki", 0.30);
  config.pid.cascade.kd = cascade.value("kd", 0.0);
  config.pid.cascade.torque_to_delta = cascade.value("torque_to_delta", 0.7071067811865476);
  config.pid.cascade.target_rate_limit_deg_s = cascade.value("target_rate_limit_deg_s", 30.0);
  config.pid.cascade.derivative_cutoff_hz = cascade.value("derivative_cutoff_hz", 10.0);
  config.pid.cascade.trajectory_rate_feedforward = cascade.value("trajectory_rate_feedforward", 0.0);
  const auto effectiveness = cascade.value("collective_gain_compensation", Json::object());
  config.pid.cascade.collective_gain_compensation = effectiveness.value("enabled", false);
  config.pid.cascade.effectiveness_reference_collective =
    effectiveness.value("reference_collective", 0.1);
  config.pid.cascade.thrust_curve_linear = effectiveness.value("thrust_curve_linear", 1.0);
  config.pid.cascade.thrust_curve_quadratic = effectiveness.value("thrust_curve_quadratic", 0.0);
  if (config.pid.cascade.collective_gain_compensation) {
    // Enabled models must explicitly declare their measured command domain.
    config.pid.cascade.effectiveness_reference_collective = effectiveness.at("reference_collective");
    config.pid.cascade.thrust_curve_linear = effectiveness.at("thrust_curve_linear");
    config.pid.cascade.thrust_curve_quadratic = effectiveness.at("thrust_curve_quadratic");
    config.pid.cascade.thrust_curve_command_min = effectiveness.at("command_min");
    config.pid.cascade.thrust_curve_command_max = effectiveness.at("command_max");
  }
  config.pid.additional_limits_enabled = pid.value("additional_limits_enabled", true);
  config.pid.motor_slew_time_s = pid.value("motor_slew_time_s", 0.2);
  config.pid.zero_quaternion = pid.at("zero_quaternion_wxyz").get<std::array<double, 4>>();
  config.pid.zero_angle_deg = pid.at("zero_angle_deg");
  config.pid.kp = pid.at("kp");
  config.pid.ki = pid.at("ki");
  config.pid.kd = pid.at("kd");
  config.pid.integral_limit = pid.at("integral_limit");
  config.pid.differential_limit = pid.at("differential_limit");
  config.pid.angle_min_deg = pid.at("angle_min_deg");
  config.pid.angle_max_deg = pid.at("angle_max_deg");
  config.pid.rate_limit_deg_s = pid.at("rate_limit_deg_s");
  config.pid.reference_rate_limit_deg_s = pid.value("reference_rate_limit_deg_s", config.pid.rate_limit_deg_s);
  config.pid.swing_limit_deg = pid.at("swing_limit_deg");
  config.pid.tracking_limit_deg = pid.at("tracking_limit_deg");
  config.pid.feedback_timeout_s = pid.at("feedback_timeout_s");
  config.pid.saturation_timeout_s = pid.at("saturation_timeout_s");
  const auto& settling = json.at("settling");
  config.settled_error_deg = settling.at("error_deg");
  config.settled_rate_deg_s = settling.at("rate_deg_s");
  config.settled_dwell_s = settling.at("dwell_s");
  config.settled_window_s = settling.value("window_s", 0.15);
  config.maximum_hold_s = settling.at("maximum_hold_s");
  const auto startup = json.value("startup", Json::object());
  config.startup.control_ready_verified = startup.value("control_ready_verified", false);
  config.startup.ramp_shape = startup.value("ramp_shape", std::string("LINEAR"));
  config.startup.minimum_duration_s = startup.value("minimum_duration_s", 0.0);
  config.startup.capture_reference = startup.value("capture_reference", false);
  config.startup.encoder_absolute_anchor =
    startup.value("encoder_absolute_anchor", startup.value("encoder_pitch_anchor", false));
  config.startup.reference_dwell_s = startup.value("reference_dwell_s", 0.3);
  config.startup.reference_timeout_s = startup.value("reference_timeout_s", 5.0);
  config.startup.reference_angle_tolerance_deg = startup.value("reference_angle_tolerance_deg", 1.0);
  config.startup.reference_swing_tolerance_deg = startup.value("reference_swing_tolerance_deg", 2.0);
  config.startup.reference_rate_tolerance_deg_s = startup.value("reference_rate_tolerance_deg_s", 5.0);
  config.startup.integral_enable_collective = startup.value("integral_enable_collective", 0.1);
  const auto shutdown = json.value("shutdown", Json::object());
  config.shutdown_minimum_duration_s = shutdown.value("minimum_duration_s", 2.0);
  config.shutdown_external_fade_s = shutdown.value("external_fade_s", 0.5);
  for (const auto& entry : json.at("segments")) {
    Segment segment;
    segment.condition_id = entry.at("condition_id");
    segment.segment_id = entry.at("segment_id");
    segment.hold_id = entry.value("hold_id", std::string{});
    segment.kind = entry.at("kind");
    segment.duration_s = entry.at("duration_s");
    segment.start_deg = entry.at("start_deg");
    segment.end_deg = entry.value("end_deg", segment.start_deg);
    segment.collective = entry.at("collective");
    segment.collective_step_up = entry.value("collective_step_up", false);
    segment.fixed_input = entry.value(
      "fixed_input", std::array<double, 4>{ segment.collective, 0, 0, 0 });
    segment.fixed_amplitude = entry.value(
      "fixed_amplitude", std::array<double, 4>{ 0, 0, 0, 0 });
    segment.analysis_start_s = entry.value("analysis_start_s", 0.0);
    segment.analysis_end_s = entry.value("analysis_end_s", 0.0);
    segment.amplitude_deg = entry.value("amplitude_deg", 0.0);
    segment.frequency_hz = entry.value("frequency_hz", 0.0);
    segment.phase_rad = entry.value("phase_rad", 0.0);
    segment.fade_s = entry.value("fade_s", 0.0);
    segment.external_amplitude = entry.value("external_amplitude", 0.0);
    segment.external_frequency_hz = entry.value("external_frequency_hz", 0.0);
    segment.external_phase_rad = entry.value("external_phase_rad", 0.0);
    segment.require_settled = entry.value("require_settled", false);
    segment.preparation_only = entry.value("preparation_only", false);
    config.segments.push_back(segment);
  }
  if (config.role != "PILOT") {
    std::set<std::string> conditions;
    for (const auto& row : json.at("source_conditions")) {
      const std::string id = row.at("id"), axis = row.at("axis");
      check(
        conditions.insert(id).second && row.at("role").get<std::string>() == config.role &&
          row.at("wind_mps").get<double>() == config.wind_m_s &&
          row.at("heading_deg").get<double>() == config.heading_deg &&
          axis == (json.contains("coupled_yaw") ? "pitch_yaw" : config.control_mode == "FIXED_INPUT" ? "fixed" :
                    (config.pid.axis == "PITCH" ? "pitch" : "yaw")) &&
          ((axis != "yaw" && axis != "fixed") ||
           row.at("pitch_deg").get<double>() == config.fixed_pitch_deg) &&
          config.repetition <= row.at("repeats").get<int>(),
        "source conditions cross session boundaries");
    }
    for (const auto& segment : config.segments) {
      check(conditions.count(segment.condition_id) == 1, "segment not in source conditions");
    }
  }
  if (json.contains("coupled_yaw")) {
    check(!json.at("coupled_yaw").contains("coupled_yaw"), "nested coupled session");
    config.coupled_yaw = std::make_shared<SessionConfig>(parseSession(json.at("coupled_yaw")));
    config.rig_base_quaternion = json.value("rig_base_quaternion_wxyz", std::array<double,4>{1,0,0,0});
    config.coupled_anchor_resolved = json.value("coupled_anchor_resolved", false);
  }
  validateSession(config);
  return config;
}

SessionConfig loadSession(const std::filesystem::path& path)
{
  std::ifstream input(path);
  check(bool(input), "cannot read session config");
  Json json; input >> json;
  return parseSession(json);
}

void validateSession(const SessionConfig& _config)
{
  if (_config.coupled_yaw) {
    const auto& y=*_config.coupled_yaw;
    validateSession(y);
    check(_config.control_mode=="AXIS_PID" && y.control_mode=="AXIS_PID" &&
          _config.pid.axis=="PITCH" && y.pid.axis=="YAW", "coupled requires Pitch and Yaw PID");
    check(!_config.startup.capture_reference && !y.startup.capture_reference &&
          _config.pid.zero_angle_deg==0 && y.pid.zero_angle_deg==0,
          "coupled uses shared rig anchor without independent recapture");
    check(_config.pid.angle_min_deg > -75 && _config.pid.angle_max_deg < 75,
          "coupled Pitch must avoid joint singularity");
    check(_config.pid.motor_slew_time_s==0 && y.pid.motor_slew_time_s==0,
          "coupled differential slew unsupported");
    check(_config.hardware.motor_min==y.hardware.motor_min && _config.hardware.motor_max==y.hardware.motor_max &&
          _config.hardware.uart_device==y.hardware.uart_device && _config.maximum_duration_s==y.maximum_duration_s &&
          _config.segments.size()==y.segments.size(), "coupled hardware/timeline mismatch");
    double dot=0;
    for(int i=0;i<4;++i) dot+=_config.hardware.axis_mix[i]*y.hardware.axis_mix[i];
    check(std::abs(dot)<1e-9, "coupled motor mixes must be independent and orthogonal");
    auto base=_config.rig_base_quaternion;
    check(normalizeQuaternion(base), "invalid coupled rig anchor");
    for(std::size_t i=0;i<_config.segments.size();++i) {
      const auto& p=_config.segments[i]; const auto& s=y.segments[i];
      check(p.segment_id==s.segment_id && p.duration_s==s.duration_s && p.collective==s.collective &&
            p.require_settled==s.require_settled && p.preparation_only==s.preparation_only &&
            p.analysis_start_s==s.analysis_start_s && p.analysis_end_s==s.analysis_end_s,
            "coupled segment timelines differ");
    }
  }
  const auto& hardware = _config.hardware;
  const auto& pid = _config.pid;
  const auto& startup = _config.startup;
  const bool fixed_input = _config.control_mode == "FIXED_INPUT";
  check(_config.control_mode == "AXIS_PID" || _config.control_mode == "FIXED_HOLD" || fixed_input,
        "unknown control mode");
  if (_config.control_mode == "FIXED_HOLD") {
    check(pid.kp == 0 && pid.ki == 0 && pid.kd == 0, "fixed hold requires zero PID gains");
    check(_config.role == "PILOT", "fixed hold entry is PILOT only, not a full fixed T/D/H executor");
    check(!_config.runtime_qualified || _config.mechanically_locked_verified, "fixed hold requires mechanical lock");
    for (const auto& segment : _config.segments) {
      check(segment.kind == "HOLD" && segment.start_deg == pid.zero_angle_deg &&
              segment.end_deg == pid.zero_angle_deg && segment.external_amplitude == 0,
            "fixed hold rejects angle motion and external excitation");
    }
  }
  check(isValidId(_config.session_id) && isValidId(_config.parent_series_id), "invalid session identity");
  const double maximum_swing_limit_deg = pid.axis == "YAW" ? 180.0 : 90.0;
  check(
    (_config.role == "T" || _config.role == "D" || _config.role == "H" || _config.role == "PILOT") &&
      _config.repetition >= 1 && _config.repetition <= (_config.role == "H" ? 3 : 2),
    "invalid role/repetition");
  check(
    isFinite(_config.wind_m_s) && _config.wind_m_s >= 0 && _config.wind_m_s <= 10 && isFinite(_config.heading_deg) &&
      std::abs(_config.heading_deg) <= 30 && isFinite(_config.fixed_pitch_deg),
    "invalid environment");
  check((_config.wind_m_s != 3.75 && _config.wind_m_s != 8.75) || _config.role == "H", "H-only wind");
  check(pid.axis == "PITCH" || pid.axis == "YAW", "target axis must be PITCH or YAW");
  const bool cascaded = pid.law == "ANGLE_RATE_PID";
  check(cascaded || pid.law == "ANGLE_PID", "unknown PID law");
  check(!cascaded || _config.control_mode == "AXIS_PID", "cascade requires AXIS_PID mode");
  auto quaternion = pid.zero_quaternion;
  check(normalizeQuaternion(quaternion), "invalid quaternion norm");
  for (double v : { pid.zero_angle_deg,
                    pid.motor_slew_time_s,
                    pid.kp,
                    pid.ki,
                    pid.kd,
                    pid.integral_limit,
                    pid.differential_limit,
                    pid.angle_min_deg,
                    pid.angle_max_deg,
                    pid.rate_limit_deg_s,
                    pid.swing_limit_deg,
                    pid.tracking_limit_deg,
                    pid.feedback_timeout_s,
                    pid.saturation_timeout_s,
                    hardware.motor_min,
                    hardware.motor_max,
                    hardware.startup_ramp_rate,
                    hardware.normal_ramp_rate,
                    hardware.abort_ramp_rate,
                    _config.maximum_duration_s,
                    _config.transition_rate_deg_s,
                    _config.settled_error_deg,
                    _config.settled_rate_deg_s,
                    _config.settled_dwell_s,
                    _config.settled_window_s,
                    _config.maximum_hold_s,
                     startup.integral_enable_collective,
                     startup.minimum_duration_s,
                     startup.reference_dwell_s,
                     startup.reference_timeout_s,
                     startup.reference_angle_tolerance_deg,
                     startup.reference_swing_tolerance_deg,
                     startup.reference_rate_tolerance_deg_s,
                     _config.shutdown_minimum_duration_s,
                    _config.shutdown_external_fade_s }) {
    check(isFinite(v), "nonfinite setting");
  }
  const auto& rate = pid.cascade;
  for (double value : {rate.attitude_kp, rate.kp, rate.ki, rate.kd, rate.torque_to_delta,
                       rate.target_rate_limit_deg_s, rate.derivative_cutoff_hz,
                       rate.effectiveness_reference_collective, rate.thrust_curve_linear,
                       rate.thrust_curve_quadratic, rate.thrust_curve_command_min,
                       rate.thrust_curve_command_max, rate.trajectory_rate_feedforward}) {
    check(isFinite(value), "nonfinite cascade setting");
  }
  check(!rate.collective_gain_compensation || (cascaded && pid.axis == "PITCH"),
        "thrust-curve compensation requires PITCH ANGLE_RATE_PID (not yaw drag torque)");
  check(rate.trajectory_rate_feedforward >= 0 && rate.trajectory_rate_feedforward <= 1 &&
          (cascaded || rate.trajectory_rate_feedforward == 0), "invalid trajectory rate feedforward");
  if (cascaded) {
    check(pid.kp == 0 && pid.ki == 0 && pid.kd == 0, "cascade must not also enable legacy PID gains");
    check(pid.additional_limits_enabled && rate.attitude_kp > 0 && rate.kp > 0 && rate.ki >= 0 && rate.kd >= 0 &&
            rate.torque_to_delta > 0 && rate.torque_to_delta <= 1 && rate.target_rate_limit_deg_s > 0 &&
            rate.target_rate_limit_deg_s < pid.rate_limit_deg_s && rate.derivative_cutoff_hz > 0 &&
            rate.derivative_cutoff_hz <= 40,
          "invalid cascade gains/rate/filter/authority settings");
    if (rate.collective_gain_compensation) {
      const auto slope = [&](double collective) {
        return rate.thrust_curve_linear + 2 * rate.thrust_curve_quadratic * collective;
      };
      check(rate.thrust_curve_command_min > 0 &&
              rate.thrust_curve_command_min < rate.thrust_curve_command_max &&
              rate.thrust_curve_command_max <= 1 && hardware.motor_max <= rate.thrust_curve_command_max &&
              rate.effectiveness_reference_collective >= rate.thrust_curve_command_min &&
              rate.effectiveness_reference_collective <= hardware.motor_max &&
              rate.thrust_curve_linear > 0 && rate.thrust_curve_quadratic >= 0 &&
              isFinite(slope(rate.effectiveness_reference_collective)) &&
              isFinite(slope(hardware.motor_max)) && slope(hardware.motor_max) > 0,
            "invalid collective gain compensation model");
      const double reference_slope = slope(rate.effectiveness_reference_collective);
      const double scale_at_max = reference_slope / slope(hardware.motor_max);
      check(scale_at_max >= 0.2 && scale_at_max <= 1,
            "collective gain compensation scale outside validated range");
    }
  }
  if (fixed_input) {
    check(pid.kp == 0 && pid.ki == 0 && pid.kd == 0, "fixed input requires zero PID gains");
    check(_config.role != "PILOT", "formal fixed input must reference a T/D/H condition");
    check(_config.mechanically_locked_verified, "fixed input requires full mechanical lock");
    const std::array<std::array<double, 4>, 4> expected{{
      {{1, 1, 1, 1}}, {{1, -1, -1, 1}}, {{1, 1, -1, -1}}, {{1, -1, 1, -1}}
    }};
    check(_config.fixed_input_mix == expected,
          "fixed_input_mix must be PX4 Quad X columns C/P/R/Y");
  }
  check((startup.ramp_shape == "LINEAR" || startup.ramp_shape == "QUINTIC") &&
          startup.minimum_duration_s >= 0 && startup.minimum_duration_s <= 120 &&
          (startup.ramp_shape == "QUINTIC" || startup.minimum_duration_s == 0), "invalid startup ramp");
  check(startup.reference_dwell_s >= .3 && startup.reference_dwell_s <= 2 &&
          startup.reference_timeout_s >= startup.reference_dwell_s && startup.reference_timeout_s <= 10 &&
          startup.reference_angle_tolerance_deg >= .5 && startup.reference_angle_tolerance_deg <= 3 &&
          startup.reference_swing_tolerance_deg >= 1 && startup.reference_swing_tolerance_deg <= 5 &&
          startup.reference_rate_tolerance_deg_s >= 2 && startup.reference_rate_tolerance_deg_s <= 10 &&
          (!startup.capture_reference ||
           ((_config.control_mode == "AXIS_PID" || fixed_input) &&
            (pid.zero_angle_deg == 0 || startup.encoder_absolute_anchor))),
        "invalid session reference capture");
  check(!startup.encoder_absolute_anchor ||
          (startup.capture_reference && _config.control_mode == "AXIS_PID" &&
           (pid.axis == "PITCH" || pid.axis == "YAW")),
        "encoder absolute anchor requires captured Pitch/Yaw AXIS_PID");
  check(
    ((_config.control_mode == "FIXED_HOLD" || fixed_input || cascaded) ? pid.kp == 0 : pid.kp > 0) &&
      pid.ki >= 0 && pid.kd >= 0 && pid.integral_limit >= 0 && pid.differential_limit > 0 &&
      pid.integral_limit <= pid.differential_limit,
    "invalid PID gains/limits");
  check(
    pid.angle_min_deg < pid.angle_max_deg && pid.zero_angle_deg >= pid.angle_min_deg &&
      pid.zero_angle_deg <= pid.angle_max_deg && pid.rate_limit_deg_s > 0 &&
      isFinite(pid.reference_rate_limit_deg_s) && pid.reference_rate_limit_deg_s > 0 &&
      pid.reference_rate_limit_deg_s <= 100 && pid.swing_limit_deg > 0 &&
      pid.swing_limit_deg <= maximum_swing_limit_deg && pid.tracking_limit_deg > 0 && pid.feedback_timeout_s > 0 &&
      pid.feedback_timeout_s <= 0.15,
    "invalid feedback limits");
  check(pid.saturation_timeout_s > 0 && pid.saturation_timeout_s <= 5, "invalid saturation timeout");
  check(pid.motor_slew_time_s >= 0 && pid.motor_slew_time_s <= 10, "invalid motor slew time");
  check(
    hardware.motor_min >= 0 && hardware.motor_max <= 1 && hardware.motor_min < hardware.motor_max,
    "motor operating bounds must lie within 0..1");
  check(
    hardware.startup_ramp_rate > 0 && hardware.startup_ramp_rate <= 0.2 && hardware.normal_ramp_rate > 0 &&
      hardware.normal_ramp_rate <= (fixed_input ? 0.3 : 0.1) &&
      hardware.abort_ramp_rate > 0 && hardware.abort_ramp_rate <= 0.5,
    "invalid ramp rate");
  // 一方のモーター群が0/1に達しても、基準指令のランプを実現できる余裕を残す。
  check(
    2 * pid.motor_slew_time_s *
        std::max({ hardware.startup_ramp_rate, hardware.normal_ramp_rate, hardware.abort_ramp_rate }) <
      1,
    "motor slew time is too long for configured collective ramps");
  check(
    !hardware.uart_device.empty() && hardware.target_system != 0 && hardware.target_component != 0 &&
      !hardware.log_directory.empty(),
    "invalid hardware configuration");
  // Exactly one verified balanced differential mode; no hidden collective/other-axis PID.
  double sum = 0;
  for (double v : hardware.axis_mix) {
    check(v == 1 || v == -1, "axis_mix must contain +/-1");
    sum += v;
  }
  check(sum == 0, "axis_mix must be balanced");
  check(
    _config.maximum_duration_s > 0 && _config.maximum_duration_s <= 3600 && _config.transition_rate_deg_s > 0 &&
      _config.transition_rate_deg_s <= pid.rate_limit_deg_s,
    "invalid session/transition limit");
  check(
    // Weak-authority axes can need a longer bounded recovery after the
    // collective ramp.  The overall session-duration budget below still
    // accounts for every maximum hold, so this does not create an unbounded
    // motor run.
    _config.maximum_hold_s >= 5 && _config.maximum_hold_s <= 30 && _config.settled_error_deg > 0 &&
      _config.settled_rate_deg_s > 0 && _config.settled_dwell_s > 0 && _config.settled_dwell_s <= 5 &&
      _config.settled_window_s >= 1.0 / kRateHz && _config.settled_window_s <= 0.5 &&
      _config.settled_error_deg < pid.tracking_limit_deg && _config.settled_rate_deg_s < pid.rate_limit_deg_s &&
      _config.settled_window_s + _config.settled_dwell_s <= _config.maximum_hold_s,
    "invalid settling settings");
  check(!_config.segments.empty() && _config.segments.size() <= 10000, "empty or excessive session");
  check(_config.segments.front().collective > 0, "startup collective must be positive");
  check(
    startup.integral_enable_collective > 0 && startup.integral_enable_collective >= hardware.motor_min &&
      startup.integral_enable_collective <= _config.segments.front().collective,
    "integral-enable collective must lie within startup operating range");
  check(
    !_config.runtime_qualified || _config.control_mode == "FIXED_HOLD" || fixed_input || startup.control_ready_verified,
    "startup collective control authority not verified");
  check(
    _config.shutdown_minimum_duration_s > 0 && _config.shutdown_external_fade_s > 0 &&
      _config.shutdown_external_fade_s <= _config.shutdown_minimum_duration_s,
    "invalid smooth shutdown timing");
  check(_config.segments.front().kind == "HOLD" && _config.segments.back().kind == "HOLD" &&
          (fixed_input ? (!_config.segments.front().require_settled && !_config.segments.back().require_settled &&
             std::all_of(_config.segments.front().fixed_input.begin() + 1,
                         _config.segments.front().fixed_input.end(), [](double v) { return v == 0; }) &&
             std::all_of(_config.segments.back().fixed_input.begin() + 1,
                         _config.segments.back().fixed_input.end(), [](double v) { return v == 0; })) :
           (_config.segments.front().require_settled && _config.segments.back().require_settled)),
        "session must start/end with valid baseline holds");
  std::set<std::string> ids;
  double total = 0;
  double budget = (startup.ramp_shape == "QUINTIC" ?
    smoothStopDuration(_config.segments.front().collective, hardware.startup_ramp_rate, startup.minimum_duration_s) :
    _config.segments.front().collective / hardware.startup_ramp_rate) + 0.1;
  if (startup.capture_reference) {
    budget += 1.875 * std::abs(_config.segments.front().start_deg - pid.zero_angle_deg) / _config.transition_rate_deg_s;
  }
  double peak_collective = _config.segments.front().collective;
  double previous_angle = _config.segments.front().start_deg;
  double previous_base = _config.segments.front().collective;
  for (const auto& segment : _config.segments) {
    peak_collective = std::max(peak_collective, segment.collective);
    check(
      isValidId(segment.condition_id) && isValidId(segment.segment_id) && ids.insert(segment.segment_id).second &&
        (segment.hold_id.empty() || isValidId(segment.hold_id)),
      "invalid/duplicate segment identity");
    check(
      segment.kind == "HOLD" || segment.kind == "SINE" || segment.kind == "MOVE" || segment.kind == "SCAN",
      "unknown segment kind");
    for (double v : { segment.duration_s,
                      segment.start_deg,
                      segment.end_deg,
                      segment.amplitude_deg,
                      segment.frequency_hz,
                      segment.phase_rad,
                      segment.external_amplitude,
                      segment.external_frequency_hz,
                      segment.external_phase_rad,
                      segment.fade_s,
                      segment.collective,
                      segment.analysis_start_s,
                      segment.analysis_end_s }) {
      check(isFinite(v), "nonfinite segment setting");
    }
    for (double v : segment.fixed_input) check(isFinite(v), "nonfinite fixed input");
    for (double v : segment.fixed_amplitude) check(isFinite(v), "nonfinite fixed amplitude");
    check(
      segment.analysis_start_s >= 0 && segment.analysis_start_s <= segment.analysis_end_s &&
        segment.analysis_end_s <= segment.duration_s,
      "invalid analysis window");
    check(
      segment.duration_s >= 0.01 && segment.duration_s <= _config.maximum_duration_s &&
        segment.collective >= hardware.motor_min && segment.collective <= hardware.motor_max,
      "invalid segment duration/collective");
    check(!segment.collective_step_up ||
            (_config.control_mode == "FIXED_HOLD" && segment.kind == "HOLD" &&
             pid.motor_slew_time_s == 0 && segment.external_amplitude == 0),
          "step-up requires fixed HOLD without output slew or external input");
    check(
      segment.start_deg >= pid.angle_min_deg && segment.start_deg <= pid.angle_max_deg &&
        segment.end_deg >= pid.angle_min_deg && segment.end_deg <= pid.angle_max_deg,
      "target outside angle bounds");
    check(
      !segment.require_settled || (segment.kind == "HOLD" && segment.duration_s >= (segment.preparation_only ? 1 : 5) &&
                                   segment.duration_s <= _config.maximum_hold_s && isValidId(segment.hold_id)),
      "invalid settling hold");
    check(!segment.preparation_only || (segment.kind == "HOLD" && segment.require_settled &&
            segment.analysis_start_s == 0 && segment.analysis_end_s == 0),
          "short preparation hold cannot contain an analysis window or excitation");
    if (fixed_input) {
      check(segment.kind == "HOLD" || segment.kind == "SINE", "fixed input only supports HOLD/SINE");
      check(segment.collective == segment.fixed_input[0], "collective must equal fixed C input");
      check(segment.start_deg == pid.zero_angle_deg && segment.end_deg == pid.zero_angle_deg &&
              segment.amplitude_deg == 0 && segment.external_amplitude == 0 && !segment.require_settled,
            "fixed input cannot contain angle PID/external excitation/settling");
      const bool has_amplitude = std::any_of(segment.fixed_amplitude.begin(), segment.fixed_amplitude.end(),
                                             [](double v) { return v != 0; });
      check((segment.kind == "SINE") == has_amplitude, "fixed sine amplitude mismatch");
      if (segment.kind == "SINE") {
        check(segment.frequency_hz > 0 && segment.frequency_hz <= 2 && segment.fade_s > 0 &&
                2 * segment.fade_s <= segment.duration_s,
              "invalid fixed sine/fade");
      }
      auto previous_motor = fixedMotorCommands(_config.fixed_input_mix, fixedInputReference(segment, 0));
      for (double t = 0.01; t < segment.duration_s + 0.01; t += 0.01) {
        const auto input = fixedInputReference(segment, std::min(t, segment.duration_s));
        const auto next_motor = fixedMotorCommands(_config.fixed_input_mix, input);
        for (std::size_t motor = 0; motor < next_motor.size(); ++motor) {
          check(next_motor[motor] >= hardware.motor_min - 1e-6 && next_motor[motor] <= hardware.motor_max + 1e-6,
                "fixed input exceeds motor bounds");
          check(std::abs(next_motor[motor] - previous_motor[motor]) <= hardware.normal_ramp_rate * .01 + 1e-6,
                "fixed waveform exceeds motor command rate");
        }
        previous_motor = next_motor;
      }
    }
    else if (segment.kind == "SINE") {
      check(
        segment.amplitude_deg > 0 && segment.frequency_hz > 0 && segment.frequency_hz <= 1 && segment.fade_s > 0 &&
          2 * segment.fade_s <= segment.duration_s,
        "invalid dynamic sine/fade");
      check(
        segment.start_deg - segment.amplitude_deg >= pid.angle_min_deg &&
          segment.start_deg + segment.amplitude_deg <= pid.angle_max_deg,
        "sine exceeds angle envelope");
      check(
        segment.external_amplitude >= 0 &&
          (!pid.additional_limits_enabled || segment.external_amplitude <= pid.differential_limit) &&
          (segment.external_amplitude == 0 || (segment.external_frequency_hz > 0 && segment.external_frequency_hz <= 1)),
        "invalid external excitation");
    }
    else {
      check(segment.amplitude_deg == 0 && segment.external_amplitude == 0, "excitation only belongs to SINE");
    }
    if (segment.kind == "SCAN") {
      check(
        segment.duration_s >= 2 && std::abs(segment.end_deg - segment.start_deg) / (segment.duration_s - 1) <= 8.000001,
        "scan speed/ramp exceeds plan");
    }
    // Sampled trajectory, including fade derivatives, must satisfy configured rate limit.
    auto previous = segmentReference(segment, 0);
    for (double t = 0.01; t < segment.duration_s + 0.01; t += 0.01) {
      const auto next = segmentReference(segment, std::min(t, segment.duration_s));
      check(isFinite(next.rate_deg_s) && (!cascaded ||
              std::abs(rate.trajectory_rate_feedforward * next.rate_deg_s) <= rate.target_rate_limit_deg_s),
            "trajectory feedforward exceeds rate target limit");
      check(
        std::abs(next.angle_deg - previous.angle_deg) <= pid.reference_rate_limit_deg_s * 0.01 + 1e-7,
        "reference exceeds rate limit");
      previous = next;
    }
    total += segment.duration_s;
    const auto start = segmentReference(segment, 0), end = segmentReference(segment, segment.duration_s);
    budget += segment.duration_s + 0.03 +
              1.875 * std::abs(start.angle_deg - previous_angle) / _config.transition_rate_deg_s +
              std::abs(segment.collective - previous_base) / hardware.normal_ramp_rate;
    if (segment.require_settled) {
      budget += _config.maximum_hold_s - segment.duration_s;
    }
    previous_angle = end.angle_deg;
    previous_base = segment.collective;
  }
  check(total < _config.maximum_duration_s, "nominal series leaves no startup/shutdown time");
  // 途中で操作停止した場合も、最大基準指令から優しく減速する時間を確保する。
  budget += smoothStopDuration(peak_collective, hardware.normal_ramp_rate, _config.shutdown_minimum_duration_s);
  check(
    budget < _config.maximum_duration_s,
    "session including ramps/transitions/maximum holds exceeds duration limit; split first");
}

struct CommissioningOutputLimits
{
  double motor_max = 0.12;
  double kp = 0.002;
  double ki = 0.0005;
  double kd = 0.001;
  double integral = 0.002;
  double differential = 0.01;
  bool allow_unlimited_slew = false;
  bool allow_cascade = false;
};

static void validateCommissioningEnvelope(const SessionConfig& _config, double _duration_limit, double _rate_limit,
                                          double _angle_limit = 5, double _tracking_limit = 3,
                                          double _swing_limit = 3, double _settled_error_limit = 1,
                                          double _settled_rate_limit = 2,
                                          CommissioningOutputLimits _limits = {},
                                          const std::string& _axis = "PITCH",
                                          const std::array<double, 4>& _axis_mix = {1, -1, 1, -1})
{
  validateSession(_config);
  check(_config.control_mode == "AXIS_PID", "props-off entry requires AXIS_PID");
  const auto& h = _config.hardware;
  const auto& p = _config.pid;
  const bool cascade = p.law == "ANGLE_RATE_PID";
  check(!cascade || _limits.allow_cascade, "cascade not allowed in this commissioning entry");
  check(!_config.startup.capture_reference || cascade, "automatic reference requires cascade commissioning");
  check(!_config.runtime_qualified && _config.role == "PILOT" && _config.wind_m_s == 0,
        "props-off must remain unqualified, PILOT, no wind");
  check(_config.maximum_duration_s <= _duration_limit && _config.maximum_hold_s == 5 && _config.segments.size() == 1,
        "commissioning requires one bounded hold within its total duration limit");
  const auto& s = _config.segments.front();
  check(s.kind == "HOLD" && s.require_settled && s.duration_s == (cascade && s.preparation_only ? 1 : 5) && s.start_deg == 0 &&
          s.end_deg == 0 && s.collective <= 0.1 && s.external_amplitude == 0,
        "props-off only permits a zero-target hold, collective <=0.10");
  check(h.motor_min == 0 && h.motor_max <= _limits.motor_max && h.startup_ramp_rate <= 0.1 &&
          h.normal_ramp_rate <= 0.1 && h.abort_ramp_rate <= 0.1,
        "props-off motor/ramp limits exceeded");
  // PX4 Quad X: 1 front-right, 2 rear-left, 3 front-left, 4 rear-right.
  // Positive Pitch torque increases front thrust and decreases rear thrust.
  check(p.axis == _axis && h.axis_mix == _axis_mix,
        "commissioning target axis/mapping mismatch");
  check(p.additional_limits_enabled && p.differential_limit <= _limits.differential && p.integral_limit <= _limits.integral &&
          p.kp <= _limits.kp && p.ki <= _limits.ki && p.kd <= _limits.kd &&
          (p.motor_slew_time_s >= 0.2 || (_limits.allow_unlimited_slew && p.motor_slew_time_s == 0)),
        "props-off PID limits exceeded");
  check(p.zero_angle_deg == 0 && p.angle_min_deg >= -_angle_limit && p.angle_max_deg <= _angle_limit &&
          p.rate_limit_deg_s <= _rate_limit && p.swing_limit_deg <= _swing_limit && p.tracking_limit_deg <= _tracking_limit &&
          p.feedback_timeout_s <= 0.1 && p.saturation_timeout_s <= 2,
        "props-off feedback limits exceeded");
  check(_config.shutdown_minimum_duration_s >= 2 && _config.settled_error_deg <= _settled_error_limit &&
          _config.settled_rate_deg_s <= _settled_rate_limit && _config.settled_window_s >= 0.15 &&
          _config.settled_dwell_s >= 0.5,
        "props-off settling/shutdown limits exceeded");
  check(h.uart_device == "/dev/serial0" || h.uart_device.rfind("/dev/pts/", 0) == 0,
        "props-off requires aircraft UART or synthetic PTY");
}

void validatePropsOffSession(const SessionConfig& _config)
{
  validateCommissioningEnvelope(_config, 12, 20);
}

void validateFixedCommissioning(const SessionConfig& _config)
{
  validateSession(_config);  // Enforces FIXED_HOLD zero gains and zero excitation.
  check(_config.control_mode == "FIXED_HOLD", "fixed commissioning requires FIXED_HOLD");
  // Reuse the same strict time/output/feedback envelope, not its execution permission.
  // This validation-only copy is never sent to a controller or saved as a qualification.
  auto envelope = _config;
  envelope.control_mode = "AXIS_PID";
  envelope.pid.kp = 0.002;
  // Ten-second startup + five-second hold + two-second stop needs timing margin.
  // Longer limits are exclusive to bounded slow FIXED_HOLD profiles.
  const double duration_limit = _config.hardware.startup_ramp_rate <= 0.01 ? 18 :
                                _config.hardware.startup_ramp_rate <= 0.02 ? 13 : 12;
  validateCommissioningEnvelope(envelope, duration_limit, 60, 20, 20, 20);
}

void validatePitchHoldCommissioning(const SessionConfig& _config)
{
  // Shared bounded hold envelope only; execution does NOT use props-off auto-zero.
  const bool cascade = _config.pid.law == "ANGLE_RATE_PID";
  // A 10 s 0->0.10 quintic ramp has a 0.01875/s peak slope.  Preserve room
  // for the bounded 5 s settling extension and 2 s controlled shutdown.
  validateCommissioningEnvelope(_config, cascade ? 18 : 12, 60, 90, 90, 90, 5, 5,
    {.motor_max = .8, .kp = .007, .ki = .001, .kd = .0017,
     .integral = .1, .differential = .3, .allow_unlimited_slew = true, .allow_cascade = true});
  if (cascade) {
    const auto& rate = _config.pid.cascade;
    check(rate.attitude_kp <= 4 && rate.kp <= .14 && rate.ki <= .30 && rate.kd <= .004 &&
            rate.target_rate_limit_deg_s <= 30 && std::abs(rate.torque_to_delta - 1 / std::sqrt(2.0)) < 1e-8,
          "Pitch cascade reference gains/mapping exceeded");
  }
  check(_config.motor_mapping_verified && _config.feedback_frame_verified,
        "Pitch hold requires verified motor mapping and feedback frame");
  check(!_config.mechanically_locked_verified && !_config.startup.control_ready_verified,
        "Pitch commissioning must remain initial, not fixed or control-qualified");
  check(_config.hardware.target_system == 1 && _config.hardware.target_component == 1 &&
          _config.segments.front().collective > 0,
        "Pitch commissioning requires aircraft 1/1 and positive collective");
  const auto& p = _config.pid;
  check(p.ki <= 0.001 && (cascade || p.kd > 0) && p.differential_limit <= 0.3 &&
          p.angle_min_deg >= -90 && p.angle_max_deg <= 90 && p.rate_limit_deg_s <= 60,
        "initial Pitch hold requires Ki <=0.001, differential <=0.3, angle +/-90, rate <=60");
}

void validateYawHoldCommissioning(const SessionConfig& _config)
{
  // First powered Yaw hold: current pose only, low collective, one short hold.
  // The measured fixed-input Yaw effectiveness is encoded by torque_to_delta;
  // this entry does not promote the controller to runtime-qualified status.
  const bool cascade = _config.pid.law == "ANGLE_RATE_PID";
  validateCommissioningEnvelope(_config, cascade ? 18 : 12, 60, 90, 90, 90, 5, 5,
    {.motor_max = .8, .kp = .007, .ki = .001, .kd = .0017,
     .integral = .08, .differential = .15, .allow_unlimited_slew = true, .allow_cascade = true},
    "YAW", {1, 1, -1, -1});
  check(cascade, "initial Yaw hold requires angle-rate cascade");
  const auto& rate = _config.pid.cascade;
  check(rate.attitude_kp <= 2 && rate.kp <= .05 && rate.ki <= .08 && rate.kd <= .004 &&
          rate.target_rate_limit_deg_s <= 15 && std::abs(rate.torque_to_delta - 1.0) < 1e-8,
        "Yaw cascade entry gains/effectiveness exceeded");
  check(_config.motor_mapping_verified && _config.feedback_frame_verified,
        "Yaw hold requires verified motor mapping and feedback frame");
  check(!_config.mechanically_locked_verified && !_config.startup.control_ready_verified,
        "Yaw commissioning must remain initial, not fixed or control-qualified");
  check(_config.hardware.target_system == 1 && _config.hardware.target_component == 1 &&
          _config.segments.front().collective > 0,
        "Yaw commissioning requires aircraft 1/1 and positive collective");
}

}  // namespace fftgyro
