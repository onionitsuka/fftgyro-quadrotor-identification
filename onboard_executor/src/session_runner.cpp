#include "fftgyro_executor/session_runner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string_view>
#include <thread>

#include "fftgyro_executor/session.hpp"
#include "fftgyro_executor/settling_monitor.hpp"
#include "fftgyro_executor/coupled_controller.hpp"

namespace fftgyro
{
namespace
{
using Clock = std::chrono::steady_clock;
constexpr std::int64_t kPeriodNs = 10'000'000;
// The executor runs on non-real-time Linux.  A single missed 10 ms deadline is
// not a control failure when feedback is still fresh; abort before the separate
// 100 ms feedback watchdog, and resynchronise instead of sending catch-up bursts.
constexpr std::int64_t kSchedulerAbortNs = 50'000'000;
constexpr double kStepSeconds = 1.0 / kRateHz;
// POSIX signal handlerから書くため、通常のメンバ状態と分ける。
volatile std::sig_atomic_t g_interrupted = 0;

void handleSignal(int)
{
  g_interrupted = 1;
}
std::int64_t nowNs()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
void previewSession(const SessionConfig& _config, const std::filesystem::path& _path)
{
  std::ofstream output(_path);
  output.exceptions(std::ios::badbit | std::ios::failbit);
  output << "# reference preview only; excludes runtime startup, transitions, settling extensions and shutdown\n"
         << "condition_id,segment_id,hold_id,time_s,target_deg,external_delta,collective,trajectory_rate_deg_s,"
            "fixed_c,fixed_p,fixed_r,fixed_y,motor1,motor2,motor3,motor4,yaw_target_deg,yaw_trajectory_rate_deg_s\n"
         << std::setprecision(17);
  double elapsed = 0;
  for (const auto& segment : _config.segments) {
    const auto ticks = static_cast<std::size_t>(std::ceil(segment.duration_s * kRateHz));
    for (std::size_t i = 0; i < ticks; ++i) {
      const double time = static_cast<double>(i) / kRateHz;
      const auto reference = segmentReference(segment, time);
      const auto fixed = fixedInputReference(segment, time);
      const auto motor = fixedMotorCommands(_config.fixed_input_mix, fixed);
      output << segment.condition_id << ',' << segment.segment_id << ',' << segment.hold_id << ',' << elapsed + time
             << ',' << reference.angle_deg << ',' << reference.external << ',' << segment.collective
             << ',' << reference.rate_deg_s;
      for (double value : fixed) output << ',' << value;
      for (float value : motor) output << ',' << value;
      const auto yaw=_config.coupled_yaw ? segmentReference(
        _config.coupled_yaw->segments[&segment-_config.segments.data()],time):Reference{};
      output << ',' << yaw.angle_deg << ',' << yaw.rate_deg_s;
      output << '\n';
    }
    elapsed += static_cast<double>(ticks) / kRateHz;
  }
  output.flush();
}

/** @brief 計測確認→立上げ→区間列→減速→release。状態はこの一か所に置く。 */
class SessionRunner
{
public:
  explicit SessionRunner(const SessionConfig& _config, bool _props_off = false,
                         bool _fixed_commissioning = false, bool _pitch_commissioning = false);
  int run() noexcept;

private:
  bool initialize() noexcept;
  bool poll() noexcept;
  bool preflight() noexcept;
  bool start() noexcept;
  bool tick(
    const Reference& _reference,
    const Segment& _segment,
    const char* _stage,
    double _collective,
    double _ramp_rate,
    const std::array<double, 4>& _fixed_input = {}) noexcept;
  bool playSegment(const Segment& _segment) noexcept;
  bool stop(double _rate) noexcept;
  bool release() noexcept;
  bool finish() noexcept;
  bool fail(const char* _message, bool _controlled_stop = false) noexcept;
  void writeRecord(const Record& _record, const Segment& _segment, const char* _stage);

  SessionConfig config_;
  const bool props_off_;
  const bool fixed_commissioning_;
  const bool pitch_commissioning_;
  bool commissioning() const noexcept { return props_off_ || fixed_commissioning_; }
  AxisController controller_;
  std::unique_ptr<CoupledController> coupled_;
  CoupledOutput coupled_output_;
  Reference yaw_reference_;
  Reference yaw_command_;
  std::unique_ptr<SettlingMonitor> yaw_settling_;
  SettlingMonitor settling_;
  std::optional<MavlinkUart> link_;
  std::filesystem::path directory_;
  std::ofstream log_, result_file_;
  std::vector<Record> records_;
  std::uint32_t run_id_ = {};
  std::int64_t scheduled_ = {}, first_send_ = {}, last_ack_ = {};
  double collective_ = {}, segment_time_ = -1;
  std::array<double, 4> fixed_input_ = {};
  std::array<float, 4> motor_ = {};
  Reference reference_;
  ControlOutput output_;
  std::array<char, 256> error_ = {};
  std::optional<Feedback> first_failure_feedback_;
  std::int64_t first_failure_ns_ = {};
  bool controlled_stop_ = false, shutdown_ = false, aborted_ = false;
  bool acquiring_target_ = false;
};

SessionRunner::SessionRunner(const SessionConfig& _config, bool _props_off,
                             bool _fixed_commissioning, bool _pitch_commissioning)
  : config_(_config)
  , props_off_(_props_off)
  , fixed_commissioning_(_fixed_commissioning)
  , pitch_commissioning_(_pitch_commissioning)
  , controller_(config_.hardware, config_.pid)
  , settling_(config_)
  , reference_{ _config.pid.zero_angle_deg, 0 }
{
  if(config_.coupled_yaw) {
    coupled_=std::make_unique<CoupledController>(config_.hardware,config_.pid,
      config_.coupled_yaw->hardware,config_.coupled_yaw->pid,config_.rig_base_quaternion);
    yaw_settling_=std::make_unique<SettlingMonitor>(*config_.coupled_yaw);
  }
}

bool SessionRunner::fail(const char* _message, bool _controlled_stop) noexcept
{
  // Snapshot before release/poll can replace the triggering feedback. No disk I/O here.
  if (!aborted_) {
    first_failure_ns_ = nowNs();
    if (link_) {
      first_failure_feedback_ = link_->feedback();
    }
  }
  aborted_ = true;
  controlled_stop_ = _controlled_stop;
  const auto used = std::char_traits<char>::length(error_.data());
  if (used < error_.size() - 1) {
    std::snprintf(error_.data() + used, error_.size() - used, "%s%s", used ? "; " : "", _message);
  }
  return false;
}

int SessionRunner::run() noexcept
{
  if (coupled_ && !config_.coupled_anchor_resolved) {
    std::cerr << "coupled encoder/PX4 anchor unresolved\n"; return 1;
  }
  if (!initialize()) {
    std::cerr << error_.data() << '\n';
    return 1;
  }
  bool complete = preflight() && start();
  for (const auto& segment : config_.segments) {
    if (!complete) {
      break;
    }
    complete = playSegment(segment);
  }
  if (complete) {
    stop(config_.hardware.normal_ramp_rate);
  }
  else if (controlled_stop_) {
    stop(config_.hardware.abort_ramp_rate);
  }
  release();  // 異常時も一度releaseを試みる。送信不能ならPX4 timeoutへ任せる。
  finish();
  return aborted_ ? 1 : 0;
}

bool SessionRunner::initialize() noexcept
{
  try {
    if (pitch_commissioning_) {
      if (config_.pid.axis == "YAW") validateYawHoldCommissioning(config_);
      else validatePitchHoldCommissioning(config_);
    }
    else if (fixed_commissioning_) {
      validateFixedCommissioning(config_);
    }
    else if (props_off_) {
      validatePropsOffSession(config_);
    }
    else if (!config_.runtime_qualified && !coupled_) {
      return fail("runtime_qualified=false: preview/test only");
    }
    directory_ = config_.hardware.log_directory / config_.session_id;
    std::filesystem::create_directories(config_.hardware.log_directory);
    if (!std::filesystem::create_directory(directory_)) {
      return fail("session directory already exists");
    }
    std::ofstream snapshot(directory_ / "session.config.json");
    snapshot.exceptions(std::ios::badbit | std::ios::failbit);
    snapshot << config_.snapshot_json << '\n';
    snapshot.flush();
    records_.reserve(static_cast<std::size_t>(config_.maximum_duration_s * kRateHz) + 10);
    std::random_device random;
    run_id_ = static_cast<std::uint32_t>(random()) | 1U;
    result_file_.open(directory_ / "result.txt");
    result_file_.exceptions(std::ios::badbit | std::ios::failbit);
    result_file_ << "run_id=" << run_id_ << "\nstatus=STARTING\n";
    result_file_ << "props_off_commissioning=" << props_off_
                 << "\nruntime_qualified=" << config_.runtime_qualified << '\n';
    result_file_ << "control_mode=" << config_.control_mode << '\n';
    result_file_ << "fixed_commissioning=" << fixed_commissioning_ << '\n';
    if (fixed_commissioning_) {
      result_file_ << "fixed_completion_requires_settling=0\n";
    }
    result_file_ << "axis_hold_commissioning=" << pitch_commissioning_ << '\n'
                 << "pitch_hold_commissioning=" << (pitch_commissioning_ && config_.pid.axis == "PITCH") << '\n'
                 << "yaw_hold_commissioning=" << (pitch_commissioning_ && config_.pid.axis == "YAW") << '\n';
    result_file_.flush();
    log_.open(directory_ / "control.csv");
    log_.exceptions(std::ios::badbit | std::ios::failbit);
    log_ << "sequence,scheduled_ns,sent_ns,feedback_received_ns,px4_boot_ms,stage,condition_id,segment_id,hold_id,"
            "segment_time_s,analysis_window,target_deg,external_delta,collective,fixed_c,fixed_p,fixed_r,fixed_y,angle_deg,wrapped_deg,joint_rate_deg_"
            "s,swing_deg,"
            "qw,qx,qy,qz,p_deg_s,q_deg_s,r_deg_s,p_term,i_term,d_term,delta,saturated,"
            "requested1,requested2,requested3,requested4,sent1,sent2,sent3,sent4,slew_limited,"
            "settling_window_ready,settling_error_rms_deg,settling_rate_rms_deg_s,settling_dwell_s,settled,"
            "tracking_guard_enabled,integral_enabled,control_mode,pid_law,rate_target_deg_s,rate_error_deg_s,"
            "angular_accel_filtered_deg_s2,virtual_delta_requested,actuator_gain_scale,delta_requested,delta_lower,"
            "delta_upper,authority_limited,preparation_only,trajectory_rate_deg_s,trajectory_rate_feedforward,"
            "actuator_gain_floor_active,actuator_model_in_range,yaw_target_deg,yaw_angle_deg,yaw_rate_deg_s,"
            "yaw_p_term,yaw_i_term,yaw_d_term,yaw_delta_requested,yaw_delta,allocation_scale,rig_roll_deg\n"
         << std::setprecision(17);
    log_.flush();
    link_.emplace(config_.hardware);
    g_interrupted = 0;
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);
    return true;
  }
  catch (const std::exception& error) {
    return fail(error.what());
  }
}

bool SessionRunner::poll() noexcept
{
  try {
    const auto result = link_->poll(run_id_, records_);
    if (result.received) {
      last_ack_ = nowNs();
    }
    return result.rejected ? fail("PX4 rejected motor request") : true;
  }
  catch (const std::exception& error) {
    return fail(error.what());
  }
}

bool SessionRunner::preflight() noexcept
{
  const PidConfig absolute_anchor = config_.pid;
  try {
    link_->requestAttitude();
  }
  catch (const std::exception& error) {
    return fail(error.what());
  }
  const auto deadline = nowNs() + static_cast<std::int64_t>(config_.startup.reference_timeout_s * 1e9);
  int fresh = 0;
  std::int64_t last_rx = 0;
  std::int64_t stationary_since = 0;
  bool stationary = false;
  const bool capture = config_.startup.capture_reference;
  // A mechanically locked fixed-input run only needs a local attitude origin
  // for observation. Wind and rig vibration may legitimately exceed the
  // commissioning stillness window before motor startup, so average the first
  // fresh samples without interpreting that motion as an unsafe axis release.
  const bool fixed_reference_average = capture && config_.control_mode == "FIXED_INPUT";
  const bool require_stationary = pitch_commissioning_ || (capture && !fixed_reference_average);
  std::array<double, 4> quaternion_sum{};
  while ((fresh < 10 || (require_stationary && !stationary)) && nowNs() < deadline && !g_interrupted) {
    if (!poll()) {
      return false;
    }
    const auto& feedback = link_->feedback();
    if (feedback.valid && feedback.received_ns != last_rx) {
      if (require_stationary && last_rx != 0 && feedback.received_ns - last_rx > 100'000'000LL) {
        stationary_since = 0;
        quaternion_sum = {};
      }
      last_rx = feedback.received_ns;
      if (capture && fresh == 0) {
        config_.pid.zero_quaternion = feedback.quaternion;
        if (!normalizeQuaternion(config_.pid.zero_quaternion)) return fail("invalid reference quaternion");
        controller_.resetBeforeStart();
      }
      if (commissioning() && feedback.fault) {
        return fail("feedback fault during commissioning preflight");
      }
      if (!commissioning()) {
        // Observation only before initial Pitch hold: don't accumulate output saturation
        // while collective is zero and no motor command has been sent.
        const bool valid = coupled_ ? coupled_->read(feedback, nowNs(), coupled_output_) :
          (require_stationary || config_.control_mode == "FIXED_INPUT") ?
          controller_.readJoint(feedback, nowNs(), output_) :
          controller_.update(feedback, nowNs(), reference_, 0, kStepSeconds, motor_, output_);
        if(coupled_) output_=coupled_output_.pitch;
        if (!valid) return fail(output_.error);
      }
      if (require_stationary) {
        if (feedback.fault) return fail("feedback fault during Pitch preflight");
        // Wind-tunnel vibration can excite mechanically fixed, non-target axes.
        // Reference capture only needs the controlled joint to be locally quiet;
        // retain the swing guard so a genuine non-target attitude excursion still fails.
        const bool still =
          std::abs(output_.angle_deg - reference_.angle_deg) <= config_.startup.reference_angle_tolerance_deg &&
          output_.swing_deg <= config_.startup.reference_swing_tolerance_deg &&
          std::isfinite(output_.rate_deg_s) &&
          std::abs(output_.rate_deg_s) <= config_.startup.reference_rate_tolerance_deg_s;
        if (!still) {
          stationary_since = 0;
          quaternion_sum = {};
          if (capture) {
            config_.pid.zero_quaternion = feedback.quaternion;
            if (!normalizeQuaternion(config_.pid.zero_quaternion)) return fail("invalid reference quaternion");
            controller_.resetBeforeStart();
          }
        }
        else {
          if (stationary_since == 0) stationary_since = feedback.received_ns;
          double dot = 0;
          for (int i = 0; i < 4; ++i) dot += feedback.quaternion[i] * config_.pid.zero_quaternion[i];
          for (int i = 0; i < 4; ++i) quaternion_sum[i] += (dot < 0 ? -1 : 1) * feedback.quaternion[i];
        }
        const double dwell = capture ? config_.startup.reference_dwell_s : 1.0;
        stationary = still && (feedback.received_ns - stationary_since) * 1e-9 >= dwell;
      }
      else if (fixed_reference_average) {
        double dot = 0;
        for (int i = 0; i < 4; ++i) dot += feedback.quaternion[i] * config_.pid.zero_quaternion[i];
        for (int i = 0; i < 4; ++i) quaternion_sum[i] += (dot < 0 ? -1 : 1) * feedback.quaternion[i];
      }
      ++fresh;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  if (fresh < 10 || g_interrupted) {
    return fail("attitude preflight incomplete");
  }
  if (require_stationary && !stationary) {
    const std::string message = config_.pid.axis + " preflight not stationary near captured reference";
    return fail(message.c_str());
  }
  if (capture) {
    double norm = 0;
    for (double value : quaternion_sum) norm += value * value;
    norm = std::sqrt(norm);
    if (!std::isfinite(norm) || norm < 1e-6) return fail("empty session reference capture");
    for (int i = 0; i < 4; ++i) config_.pid.zero_quaternion[i] = quaternion_sum[i] / norm;
    if (config_.startup.encoder_absolute_anchor) {
      // Native re-averaging must not erase the encoder-to-PX4 absolute anchor.
      AxisController anchor_controller(config_.hardware, absolute_anchor);
      auto anchored_feedback = link_->feedback();
      anchored_feedback.quaternion = config_.pid.zero_quaternion;
      ControlOutput anchored_output;
      if (!anchor_controller.readJoint(anchored_feedback, nowNs(), anchored_output))
        return fail("invalid absolute Pitch reference after capture");
      config_.pid.zero_angle_deg = anchored_output.angle_deg;
      reference_.angle_deg = config_.pid.zero_angle_deg;
    }
    controller_.resetBeforeStart();
    try {
      result_file_ << "session_reference_quaternion=" << std::setprecision(17);
      for (double value : config_.pid.zero_quaternion) result_file_ << value << ',';
      result_file_ << "\nreference_zero_angle_deg=" << config_.pid.zero_angle_deg
                   << "\nencoder_absolute_anchor=" << config_.startup.encoder_absolute_anchor
                   << "\nreference_capture_start_ns=" << stationary_since
                   << "\nreference_capture_end_ns=" << last_rx
                   << "\nreference_capture_timeout_s=" << config_.startup.reference_timeout_s
                   << "\nreference_capture_angle_tolerance_deg=" << config_.startup.reference_angle_tolerance_deg
                   << "\nreference_capture_swing_tolerance_deg=" << config_.startup.reference_swing_tolerance_deg
                   << "\nreference_capture_rate_tolerance_deg_s=" << config_.startup.reference_rate_tolerance_deg_s
                   << '\n';
      result_file_.flush();
    }
    catch (const std::exception& error) { return fail(error.what()); }
  }
  if (commissioning()) {
    // Current fixed attitude is a commissioning zero only, never a rig calibration.
    config_.pid.zero_quaternion = link_->feedback().quaternion;
    if (!normalizeQuaternion(config_.pid.zero_quaternion)) {
      return fail("invalid commissioning quaternion");
    }
    try {
      result_file_ << "commissioning_zero_quaternion=" << std::setprecision(17);
      for (double value : config_.pid.zero_quaternion) result_file_ << value << ',';
      result_file_ << '\n';
      result_file_.flush();
    }
    catch (const std::exception& error) { return fail(error.what()); }
    if (!controller_.update(link_->feedback(), nowNs(), reference_, 0, kStepSeconds, motor_, output_)) {
      return fail(output_.error);
    }
  }
  if(coupled_) {
    reference_={coupled_output_.pitch.angle_deg,0,0};
    yaw_reference_=yaw_command_={coupled_output_.yaw.angle_deg,0,0};
  }
  scheduled_ = first_send_ = nowNs() + kPeriodNs;
  return true;
}

bool SessionRunner::start() noexcept
{
  const auto& first = config_.segments.front();
  acquiring_target_ = true;
  const auto captured=reference_;
  reference_ = segmentReference(first, 0);
  if(coupled_) reference_=captured;
  reference_.rate_deg_s = 0;  // The trajectory clock has not started during collective ramp-up.
  if (config_.startup.capture_reference) reference_ = {config_.pid.zero_angle_deg, 0};
  if (config_.control_mode == "FIXED_INPUT") {
    const auto target = fixedInputReference(first, 0);
    const double duration = smoothStopDuration(target[0], config_.hardware.startup_ramp_rate,
                                               config_.startup.minimum_duration_s);
    for (double time = kStepSeconds; time < duration + kStepSeconds; time += kStepSeconds) {
      std::array<double, 4> demand{};
      const double scale = quintic(std::min(time / duration, 1.0));
      for (std::size_t i = 0; i < demand.size(); ++i) demand[i] = target[i] * scale;
      if (!tick(reference_, first, "STARTUP_RAMP", demand[0], config_.hardware.startup_ramp_rate, demand)) return false;
    }
    return true;
  }
  if (config_.startup.ramp_shape == "QUINTIC") {
    const double duration = smoothStopDuration(first.collective, config_.hardware.startup_ramp_rate,
                                               config_.startup.minimum_duration_s);
    for (double time = kStepSeconds; time < duration + kStepSeconds; time += kStepSeconds) {
      const double demand = first.collective * quintic(std::min(time / duration, 1.0));
      if (!tick(reference_, first, "STARTUP_RAMP", demand, config_.hardware.startup_ramp_rate)) return false;
    }
    return true;
  }
  while (collective_ < first.collective) {
    if (!tick(reference_, first, "STARTUP_RAMP", first.collective, config_.hardware.startup_ramp_rate)) {
      return false;
    }
  }
  return true;
}

bool SessionRunner::tick(
  const Reference& _reference,
  const Segment& _segment,
  const char* _stage,
  double _collective,
  double _ramp_rate,
  const std::array<double, 4>& _fixed_input) noexcept
{
  try {
    std::this_thread::sleep_until(Clock::time_point(std::chrono::nanoseconds(scheduled_)));
    if (!poll()) {
      return false;
    }
    const auto now = nowNs();
    const auto deadline = scheduled_;
    const auto lateness = now - deadline;
    if (pitch_commissioning_ && link_->feedback().fault) {
      return fail("feedback fault during Pitch hold");
    }
    if (g_interrupted && !shutdown_) {
      return fail("operator stop", true);
    }
    // During an active experiment, a missed real-time deadline invalidates the
    // input waveform and must stop the run.  During RAMP_DOWN, however, aborting
    // the shutdown would replace a bounded smooth decrease with an abrupt
    // release.  Keep stepping toward zero; the scheduler is re-anchored below
    // whenever lateness exceeds one period.
    if (lateness > kSchedulerAbortNs && !shutdown_) {
      return fail("scheduler late");
    }
    if ((now - first_send_) * 1e-9 >= config_.maximum_duration_s) {
      return fail("session duration limit");
    }
    if (!records_.empty() && now - (last_ack_ == 0 ? first_send_ : last_ack_) > 150'000'000) {
      return fail("ACK timeout");
    }
    if (config_.control_mode == "FIXED_INPUT") {
      fixed_input_ = _fixed_input;
      collective_ = fixed_input_[0];
      ControlOutput measured{};
      if (!controller_.readJoint(link_->feedback(), now, measured)) return fail(measured.error);
      output_ = measured;
      output_.motor = fixedMotorCommands(config_.fixed_input_mix, fixed_input_);
      for (std::size_t i = 0; i < output_.motor.size(); ++i) {
        output_.requested[i] = output_.motor[i];
        if (!std::isfinite(output_.motor[i]) || output_.motor[i] < config_.hardware.motor_min - 1e-6 ||
            output_.motor[i] > config_.hardware.motor_max + 1e-6) return fail("fixed input motor bound");
      }
    }
    else if (shutdown_ && config_.control_mode == "AXIS_PID" && !pitch_commissioning_) {
      // Once shutdown begins, target-angle holding cannot remain feasible all
      // the way to zero thrust. Replay the current four outputs through a
      // monotonic, rate-bounded fade instead of driving PID into saturation.
      // Joint angle/rate/swing and feedback watchdogs remain active.
      const bool step_up = !shutdown_ && config_.control_mode == "FIXED_HOLD" &&
        _segment.collective_step_up && std::string_view(_stage) == "COLLECTIVE_STEP_UP";
      collective_ = transitionCollective(collective_, _collective, _ramp_rate / kRateHz, step_up);
      ControlOutput measured{};
      if(coupled_) {
        if(!coupled_->read(link_->feedback(),now,coupled_output_)) return fail(coupled_output_.pitch.error);
        measured=coupled_output_.pitch;
      } else if (!controller_.readJoint(link_->feedback(), now, measured)) return fail(measured.error);
      output_ = measured;
      double mix_norm = 0, projected_delta = 0;
      for (std::size_t i = 0; i < output_.motor.size(); ++i) {
        const double demand = _fixed_input[i];
        if (!std::isfinite(demand) || demand < -1e-9 ||
            demand > config_.hardware.motor_max + 1e-6 ||
            demand > static_cast<double>(motor_[i]) + 1e-6 ||
            static_cast<double>(motor_[i]) - demand > _ramp_rate / kRateHz + 1e-5) {
          return fail("shutdown motor fade not monotonic/rate bounded");
        }
        output_.motor[i] = static_cast<float>(std::max(0.0, demand));
        output_.requested[i] = demand;
        const double mix = config_.hardware.axis_mix[i];
        mix_norm += mix * mix;
        projected_delta += mix * (demand - collective_);
      }
      output_.delta = mix_norm > 0 ? projected_delta / mix_norm : 0;
    }
    else {
      const bool step_up = !shutdown_ && config_.control_mode == "FIXED_HOLD" &&
        _segment.collective_step_up && std::string_view(_stage) == "COLLECTIVE_STEP_UP";
      collective_ = transitionCollective(collective_, _collective, _ramp_rate / kRateHz, step_up);
      const bool yaw_startup = config_.control_mode == "AXIS_PID" && (config_.pid.axis == "YAW" || coupled_) &&
                               std::string_view(_stage) == "STARTUP_RAMP";
      const ControlPolicy policy{ .check_tracking = !acquiring_target_,
                                  .enable_integral = config_.control_mode == "AXIS_PID" &&
                                    collective_ >= config_.startup.integral_enable_collective,
                                  .check_saturation_timeout = !yaw_startup };
      const bool valid=coupled_ ? coupled_->update(link_->feedback(),now,_reference,yaw_command_,
        collective_,kStepSeconds,coupled_output_,policy) :
        controller_.update(link_->feedback(), now, _reference, collective_, kStepSeconds, motor_, output_, policy);
      if(coupled_) output_=coupled_output_.pitch;
      if (!valid) {
        return fail(output_.error);
      }
    }
    Record record;
    record.sequence = static_cast<std::uint32_t>(records_.size());
    record.scheduled_ns = deadline;
    record.sent_ns = nowNs();
    record.motor = output_.motor;
    if (config_.control_mode == "FIXED_HOLD") {
      if (output_.p != 0 || output_.i != 0 || output_.d != 0 || output_.delta != 0 ||
          !std::all_of(record.motor.begin(), record.motor.end(),
                       [&](float value) { return std::isfinite(value) && value == record.motor.front(); })) {
        return fail("fixed hold output must be equal, without PID or differential");
      }
    }
    if (commissioning() || pitch_commissioning_) {
      const double maximum = pitch_commissioning_ ? std::min(config_.hardware.motor_max, 0.8) : 0.12;
      for (float value : record.motor) {
        if (!std::isfinite(value) || value < 0 || value > maximum + 1e-6) {
          return fail("commissioning hard output bound");
        }
      }
    }
    if (pitch_commissioning_ && (std::abs(output_.i) > config_.pid.integral_limit + 1e-9 ||
                                  std::abs(output_.delta) > config_.pid.differential_limit + 1e-9)) {
      return fail("Pitch commissioning integral/differential bound");
    }
    link_->send(run_id_, record.sequence, record.motor, false);
    records_.push_back(record);
    motor_ = record.motor;
    reference_ = _reference;
    if(coupled_) yaw_reference_=yaw_command_;
    if (_segment.require_settled && segment_time_ >= 0) {
      settling_.update(output_.angle_deg - reference_.angle_deg, output_.rate_deg_s,
                       output_.saturated || (config_.pid.law == "ANGLE_RATE_PID" && output_.authority_limited));
      if(coupled_) yaw_settling_->update(coupled_output_.yaw.angle_deg-yaw_reference_.angle_deg,
        coupled_output_.yaw.rate_deg_s,coupled_output_.yaw.saturated || coupled_output_.yaw.authority_limited);
    }
    else {
      settling_.reset();
      if(coupled_) yaw_settling_->reset();
    }
    writeRecord(record, _segment, _stage);
    scheduled_ = lateness > kPeriodNs ? now + kPeriodNs : deadline + kPeriodNs;
    return true;
  }
  catch (const std::exception& error) {
    return fail(error.what());
  }
}

bool SessionRunner::playSegment(const Segment& _segment) noexcept
{
  settling_.reset();
  if(coupled_) yaw_settling_->reset();
  const Segment* yaw_segment=coupled_ ? &config_.coupled_yaw->segments[&_segment-config_.segments.data()] : nullptr;
  const Reference yaw_first=yaw_segment ? segmentReference(*yaw_segment,0):Reference{};
  segment_time_ = -1;
  auto first = segmentReference(_segment, 0);
  first.rate_deg_s = 0;  // Transition/collective-change stages hold this endpoint.
  if (config_.control_mode == "FIXED_INPUT") {
    const auto target = fixedInputReference(_segment, 0);
    double motor_distance = 0;
    const auto from_motor = fixedMotorCommands(config_.fixed_input_mix, fixed_input_);
    const auto to_motor = fixedMotorCommands(config_.fixed_input_mix, target);
    for (std::size_t i = 0; i < from_motor.size(); ++i) {
      motor_distance = std::max(motor_distance, std::abs(static_cast<double>(to_motor[i] - from_motor[i])));
    }
    if (motor_distance > 1e-8) {
      const double duration = std::max(1.0, 1.875 * motor_distance / config_.hardware.normal_ramp_rate);
      const auto initial = fixed_input_;
      for (double time = kStepSeconds; time < duration + kStepSeconds; time += kStepSeconds) {
        std::array<double, 4> demand{};
        const double scale = quintic(std::min(time / duration, 1.0));
        for (std::size_t i = 0; i < demand.size(); ++i) demand[i] = initial[i] + (target[i] - initial[i]) * scale;
        if (!tick(first, _segment, "FIXED_TRANSITION", demand[0], config_.hardware.normal_ramp_rate, demand)) return false;
      }
    }
    for (double elapsed = 0; elapsed + 1e-8 < _segment.duration_s; elapsed += kStepSeconds) {
      segment_time_ = elapsed;
      const auto demand = fixedInputReference(_segment, elapsed);
      if (!tick(first, _segment, "SEGMENT", demand[0], config_.hardware.normal_ramp_rate, demand)) return false;
    }
    acquiring_target_ = false;
    return true;
  }
  const double distance = first.angle_deg - reference_.angle_deg;
  const double ramp_rate = config_.hardware.normal_ramp_rate;
  const double yaw_distance=coupled_ ? yaw_first.angle_deg-yaw_reference_.angle_deg:0;
  if (std::abs(distance) > 1e-8 || std::abs(yaw_distance)>1e-8) {
    // 間の移動はモーター停止を挟まず、確認済み速度のquinticでつなぐ。
    const double duration = std::max(kStepSeconds, 1.875 * std::max(std::abs(distance),std::abs(yaw_distance)) / config_.transition_rate_deg_s);
    const double start = reference_.angle_deg;
    const double yaw_start=yaw_reference_.angle_deg;
    for (double time = 0; time < duration; time += kStepSeconds) {
      if(coupled_) yaw_command_={yaw_start+yaw_distance*quintic(time/duration),0,
        yaw_distance/duration*quinticDerivative(time/duration)};
      const Reference move{ start + distance * quintic(time / duration), 0,
                            distance / duration * quinticDerivative(time / duration) };
      if (!tick(move, _segment, "TRANSITION", _segment.collective, ramp_rate)) {
        return false;
      }
    }
    if(coupled_) yaw_command_=yaw_first;
    if (!tick(first, _segment, "TRANSITION", _segment.collective, ramp_rate)) {
      return false;
    }
  }
  if (_segment.collective_step_up && _segment.collective > collective_) {
    if (!tick(first, _segment, "COLLECTIVE_STEP_UP", _segment.collective, ramp_rate)) return false;
  }
  while (std::abs(collective_ - _segment.collective) > 1e-9) {
    if(coupled_) yaw_command_=yaw_first;
    if (!tick(first, _segment, "COLLECTIVE_CHANGE", _segment.collective, ramp_rate)) {
      return false;
    }
  }
  double elapsed = 0;
  for (;;) {
    segment_time_ = elapsed;
    const auto reference = segmentReference(_segment, std::min(elapsed, _segment.duration_s));
    if(coupled_) yaw_command_=segmentReference(*yaw_segment,std::min(elapsed,_segment.duration_s));
    const auto stage = elapsed < _segment.duration_s ? "SEGMENT" : "SETTLING_EXTENSION";
    if (!tick(reference, _segment, stage, _segment.collective, ramp_rate)) {
      return false;
    }
    elapsed += kStepSeconds;
    // Fixed commissioning measures equal motor commands, not attitude regulation.
    // Keep measured settling diagnostics; completion does not certify stability.
    if (elapsed + 1e-8 >= _segment.duration_s &&
        (fixed_commissioning_ || !_segment.require_settled ||
          (settling_.settled() && (!coupled_ || yaw_settling_->settled())))) {
      acquiring_target_ = false;
      return true;
    }
    if (_segment.require_settled && elapsed + 1e-8 >= config_.maximum_hold_s) {
      return fail("hold did not settle");
    }
  }
}

bool SessionRunner::stop(double _rate) noexcept
{
  shutdown_ = true;
  segment_time_ = -1;
  if (collective_ == 0) {
    return true;
  }
  const double initial = collective_;
  const Reference hold = reference_;
  const double rate = std::min(_rate, config_.hardware.normal_ramp_rate);
  const double duration = smoothStopDuration(initial, rate, config_.shutdown_minimum_duration_s);
  if (config_.control_mode == "FIXED_INPUT") {
    const auto initial_input = fixed_input_;
    for (double time = 0; time < duration; time += kStepSeconds) {
      std::array<double, 4> demand{};
      const double scale = 1 - quintic(std::min(time / duration, 1.0));
      for (std::size_t i = 0; i < demand.size(); ++i) demand[i] = initial_input[i] * scale;
      if (!tick(hold, config_.segments.back(), "RAMP_DOWN", demand[0], rate, demand)) return false;
    }
    return tick(hold, config_.segments.back(), "RAMP_DOWN", 0, rate, {});
  }
  if (config_.control_mode == "AXIS_PID" && !pitch_commissioning_) {
    // Scale the last actual motor commands, not an unattainable attitude
    // target. Duration is set by the largest motor, so every output respects
    // the configured maximum smooth-stop slew rate.
    const auto initial_motor = motor_;
    const double peak = static_cast<double>(*std::max_element(initial_motor.begin(), initial_motor.end()));
    const double motor_duration = smoothStopDuration(peak, rate, config_.shutdown_minimum_duration_s);
    for (double time = 0; time < motor_duration; time += kStepSeconds) {
      const double collective = smoothStopCollective(initial, time, motor_duration);
      if (!tick(hold, config_.segments.back(), "RAMP_DOWN", collective, rate,
                smoothStopMotors(initial_motor, time, motor_duration))) return false;
    }
    return tick(hold, config_.segments.back(), "RAMP_DOWN", 0, rate, {});
  }
  for (double time = 0; time < duration; time += kStepSeconds) {
    // 外部励振も徐々に消す。角度保持と積分補償は最後まで維持する。
    const double external = hold.external * (1 - quintic(time / config_.shutdown_external_fade_s));
    const Reference reference{ hold.angle_deg, external };
    const double collective = smoothStopCollective(initial, time, duration);
    if (!tick(reference, config_.segments.back(), "RAMP_DOWN", collective, rate)) {
      return false;
    }
  }
  return tick({ hold.angle_deg, 0 }, config_.segments.back(), "RAMP_DOWN", 0, rate);
}

bool SessionRunner::release() noexcept
{
  try {
    Record record;
    record.sequence = static_cast<std::uint32_t>(records_.size());
    record.release = true;
    record.sent_ns = nowNs();
    link_->send(run_id_, record.sequence, record.motor, true);
    records_.push_back(record);
    const auto deadline = nowNs() + 200'000'000LL;
    while (nowNs() < deadline && records_.back().ack_status < 0) {
      link_->poll(run_id_, records_);
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return records_.back().ack_status == FFTGYRO_MOTOR_SET_STATUS_RELEASED ?
             true :
             fail("release not acknowledged; verify RC Kill/PX4 timeout");
  }
  catch (const std::exception& error) {
    return fail(error.what());
  }
}

void SessionRunner::writeRecord(const Record& _record, const Segment& _segment, const char* _stage)
{
  const auto& feedback = link_->feedback();
  const bool analysis = segment_time_ >= _segment.analysis_start_s && segment_time_ < _segment.analysis_end_s;
  log_ << _record.sequence << ',' << scheduled_ << ',' << _record.sent_ns << ',' << feedback.received_ns << ','
       << feedback.boot_ms << ',' << _stage << ',' << _segment.condition_id << ',' << _segment.segment_id << ','
       << _segment.hold_id << ',' << segment_time_ << ',' << analysis << ',' << reference_.angle_deg << ','
       << reference_.external << ',' << collective_;
  for (double value : fixed_input_) log_ << ',' << value;
  log_ << ',' << output_.angle_deg << ',' << output_.wrapped_deg << ','
       << output_.rate_deg_s << ',' << output_.swing_deg;
  for (const double value : feedback.quaternion) {
    log_ << ',' << value;
  }
  for (const double value : feedback.body_rate_deg_s) {
    log_ << ',' << value;
  }
  log_ << ',' << output_.p << ',' << output_.i << ',' << output_.d << ',' << output_.delta << ',' << output_.saturated;
  for (const double value : output_.requested) {
    log_ << ',' << value;
  }
  for (const double value : output_.motor) {
    log_ << ',' << value;
  }
  log_ << ',' << output_.slew_limited << ',' << settling_.windowReady() << ',' << settling_.errorRms() << ','
       << settling_.rateRms() << ',' << settling_.dwell() << ',' << settling_.settled() << ','
       << (!acquiring_target_ && (!shutdown_ || pitch_commissioning_))
       << ',' << (config_.control_mode == "AXIS_PID" && (!shutdown_ || pitch_commissioning_) &&
                    collective_ >= config_.startup.integral_enable_collective)
       << ',' << (shutdown_ && config_.control_mode == "AXIS_PID" && !pitch_commissioning_ ?
                    "STOP_MOTOR_FADE" : config_.control_mode.c_str())
       << ',' << (shutdown_ && config_.control_mode == "AXIS_PID" && !pitch_commissioning_ ?
                    "NONE" : config_.pid.law.c_str()) << ',' << output_.rate_target_deg_s
       << ',' << output_.rate_error_deg_s << ',' << output_.angular_accel_filtered_deg_s2
       << ',' << output_.virtual_delta_requested << ',' << output_.actuator_gain_scale
       << ',' << output_.delta_requested << ',' << output_.delta_lower << ',' << output_.delta_upper
       << ',' << output_.authority_limited << ',' << _segment.preparation_only
       << ',' << reference_.rate_deg_s << ',' << config_.pid.cascade.trajectory_rate_feedforward
       << ',' << output_.actuator_gain_floor_active << ',' << output_.actuator_model_in_range
       << ',' << yaw_reference_.angle_deg << ',' << coupled_output_.yaw.angle_deg
       << ',' << coupled_output_.yaw.rate_deg_s << ',' << coupled_output_.yaw.p << ',' << coupled_output_.yaw.i
       << ',' << coupled_output_.yaw.d << ',' << coupled_output_.yaw.delta_requested << ',' << coupled_output_.yaw.delta
       << ',' << coupled_output_.allocation_scale << ',' << coupled_output_.roll_deg << '\n';
  if (_record.sequence % 25 == 0) {
    log_.flush();
  }
}

bool SessionRunner::finish() noexcept
{
  try {
    // ACKは遅れて到着するため、制御ログとは別に保存する。
    std::ofstream acks(directory_ / "ack.csv");
    acks.exceptions(std::ios::badbit | std::ios::failbit);
    acks << "sequence,ack_status,px4_rx_us,release\n";
    for (const auto& record : records_) {
      acks << record.sequence << ',' << record.ack_status << ',' << record.ack_rx_time_us << ',' << record.release
           << '\n';
    }
    acks.flush();
    log_.flush();
    const char* status = aborted_ ? "ABORTED" : "COMPLETE";
    result_file_ << "status=" << status << "\ndetail=" << error_.data() << '\n';
    if (first_failure_feedback_) {
      const auto& feedback = *first_failure_feedback_;
      result_file_ << std::setprecision(17)
                   << "failure_detected_ns=" << first_failure_ns_
                   << "\nfailure_feedback_received_ns=" << feedback.received_ns
                   << "\nfailure_px4_boot_ms=" << feedback.boot_ms
                   << "\nfailure_feedback_valid=" << feedback.valid
                   << "\nfailure_body_rate_deg_s=";
      for (double value : feedback.body_rate_deg_s) result_file_ << value << ',';
      result_file_ << "\nfailure_quaternion_wxyz=";
      for (double value : feedback.quaternion) result_file_ << value << ',';
      result_file_ << '\n';
    }
    result_file_.flush();
    std::cout << status << " log=" << directory_ << '\n';
    if (aborted_) {
      std::cerr << error_.data() << '\n';
    }
    return true;
  }
  catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return fail(error.what());
  }
}
}  // namespace

int sessionCommand(const std::string& _command, const std::filesystem::path& _config, const std::filesystem::path& _output)
{
  const auto config = loadSession(_config);
  if (_command == "session-pitch-hold-commissioning-validate") {
    validatePitchHoldCommissioning(config);
    std::cout << "VALID initial Pitch hold only; runtime_qualified=false\n";
    return 0;
  }
  if (_command == "session-pitch-hold-commissioning") {
    validatePitchHoldCommissioning(config);
    return SessionRunner(config, false, false, true).run();
  }
  if (_command == "session-yaw-hold-commissioning-validate") {
    validateYawHoldCommissioning(config);
    std::cout << "VALID initial Yaw hold only; runtime_qualified=false\n";
    return 0;
  }
  if (_command == "session-yaw-hold-commissioning") {
    validateYawHoldCommissioning(config);
    return SessionRunner(config, false, false, true).run();
  }
  if (_command == "session-fixed-commissioning-validate") {
    validateFixedCommissioning(config);
    std::cout << "VALID fixed commissioning only; runtime_qualified=false\n";
    return 0;
  }
  if (_command == "session-fixed-commissioning") {
    validateFixedCommissioning(config);
    return SessionRunner(config, false, true).run();
  }
  if (_command == "session-props-off-validate") {
    validatePropsOffSession(config);
    std::cout << "VALID props-off commissioning only; runtime_qualified=false\n";
    return 0;
  }
  if (_command == "session-validate") {
    std::cout << "VALID session=" << config.session_id << " axis=" << config.pid.axis
              << " segments=" << config.segments.size() << " runtime_qualified=" << config.runtime_qualified << '\n';
    return 0;
  }
  if (_command == "session-preview") {
    previewSession(config, _output);
    return 0;
  }
  if (_command == "session-run") {
    return SessionRunner(config).run();
  }
  if (_command == "session-props-off") {
    validatePropsOffSession(config);
    return SessionRunner(config, true).run();
  }
  throw std::runtime_error("unknown session command");
}
}  // namespace fftgyro
