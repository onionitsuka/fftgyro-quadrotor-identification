#include "fftgyro_executor/session.hpp"
#include "fftgyro_executor/settling_monitor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>

using namespace fftgyro;
namespace
{
void require(bool _ok, const char* _message)
{
  if (!_ok) {
    throw std::runtime_error(_message);
  }
}
bool near(double _a, double _b, double _eps = 1e-6)
{
  return std::abs(_a - _b) < _eps;
}
template <class F>
void rejects(F _fn, const char* _message)
{
  try {
    _fn();
  }
  catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error(_message);
}
template <typename Controller>
ControlOutput checkedStep(
  Controller&& _controller,
  const Feedback& _feedback,
  std::int64_t _now,
  const Reference& _reference,
  double _collective,
  double _dt,
  const std::array<float, 4>& _previous,
  ControlPolicy _policy = {})
{
  ControlOutput output;
  if (!_controller.update(_feedback, _now, _reference, _collective, _dt, _previous, output, _policy)) {
    throw std::runtime_error(output.error);
  }
  return output;
}
std::array<double, 4> product(std::array<double, 4> _a, std::array<double, 4> _b)
{
  return { _a[0] * _b[0] - _a[1] * _b[1] - _a[2] * _b[2] - _a[3] * _b[3],
           _a[0] * _b[1] + _a[1] * _b[0] + _a[2] * _b[3] - _a[3] * _b[2],
           _a[0] * _b[2] - _a[1] * _b[3] + _a[2] * _b[0] + _a[3] * _b[1],
           _a[0] * _b[3] + _a[1] * _b[2] - _a[2] * _b[1] + _a[3] * _b[0] };
}
Feedback feedback(const SessionConfig& _c, double _angle, double _rate = 0)
{
  Feedback f;
  f.valid = true;
  f.received_ns = 1'000'000'000;
  f.boot_ms = 1000;
  const double half = (_angle - _c.pid.zero_angle_deg) * std::numbers::pi / 360;
  std::array<double, 4> joint{ std::cos(half), 0, 0, 0 };
  joint[_c.pid.axis == "PITCH" ? 2 : 3] = std::sin(half);
  f.quaternion = product(_c.pid.zero_quaternion, joint);
  f.body_rate_deg_s[_c.pid.axis == "PITCH" ? 1 : 2] = _rate;
  return f;
}
void testWithoutAdditionalLimits(SessionConfig _c)
{
  require(!_c.pid.additional_limits_enabled, "working example still enables additional limits");
  require(_c.hardware.motor_min == 0 && _c.hardware.motor_max == .8, "working example output limits changed");
  _c.hardware.motor_max = 1;     // 設定可能な上限と追加制限無効化の独立性を確認する。
  _c.pid.motor_slew_time_s = 0;  // このテストでは差動・積分の上限無効化だけを確認する。
  _c.pid.kp = .05;
  _c.pid.ki = .1;
  AxisController controller(_c.hardware, _c.pid);
  std::array<float, 4> motors{ .5F, .5F, .5F, .5F };
  auto output = checkedStep(controller, feedback(_c, 0), 1'000'000'000, { 6, 0 }, .5, .01, motors);
  require(output.delta > .3 && !output.saturated, "differential limit still active");
  require(output.motor[0] > .8 && output.motor[1] < .2, "per-step limits still active");
  for (int n = 0; n < 100; ++n) {
    output = checkedStep(controller, feedback(_c, 0), 1'000'000'000, { 1, 0 }, .5, .01, output.motor);
  }
  require(output.i > .1, "integral limit still active");
  const double integral = output.i;
  output = checkedStep(controller, feedback(_c, 0), 1'000'000'000, { 1, 1 }, .5, .01, output.motor);
  require(output.saturated && near(output.delta, .5), "normalized output bounds lost");
  require(near(output.i, integral), "anti-windup lost");
  output = checkedStep(controller, feedback(_c, 0), 1'000'000'000, { 1, 0 }, 0, .01, output.motor);
  require(near(output.i, integral), "integral changed at zero collective");
  for (const float motor : output.motor) {
    require(motor == 0, "zero collective restarted motors");
  }
  _c.segments[1].external_amplitude = .1;
  _c.segments[1].external_frequency_hz = .35;
  validateSession(_c);
  _c.hardware.motor_max = 1.01;
  rejects([&] { validateSession(_c); }, "out-of-protocol motor range accepted");
}

void testMotorSlew(SessionConfig _c)
{
  require(
    !_c.pid.additional_limits_enabled && near(_c.pid.motor_slew_time_s, 0), "dynamic example still enables motor slew");
  // Disabling slew permits immediate differential reversal, but preserves the
  // requested collective, axis allocation, amplitude bounds and anti-windup.
  {
    auto unlimited = _c;
    unlimited.hardware.motor_max = .12;
    unlimited.pid.ki = 0;
    AxisController controller(unlimited.hardware, unlimited.pid);
    const std::array<float, 4> previous{ .09F, .03F, .09F, .03F };
    auto output = checkedStep(controller, feedback(unlimited, 0), 1'000'000'000,
                              { 0, -.03 }, .06, .01, previous);
    require(near(output.delta, -.03) && !output.slew_limited, "disabled slew delayed differential reversal");
    require(previous[0] - output.motor[0] > .05, "disabled slew retained previous step cap");
    require(near(output.motor[0], .03) && near(output.motor[1], .09) &&
              near(output.motor[0], output.motor[2]) && near(output.motor[1], output.motor[3]),
            "disabled slew changed collective or axis allocation");
    output = checkedStep(controller, feedback(unlimited, 0), 1'000'000'000,
                         { 0, .2 }, .06, .01, output.motor);
    require(output.saturated && near(output.delta, .06), "disabled slew removed amplitude bounds");
  }
  _c.pid.motor_slew_time_s = .2;  // Explicit opt-in still supports legacy profiles.
  _c.hardware.motor_max = 1;
  _c.pid.saturation_timeout_s = .02;
  AxisController controller(_c.hardware, _c.pid);
  std::array<float, 4> motors{ .5F, .5F, .5F, .5F };
  auto output = checkedStep(controller, feedback(_c, 0), 1'000'000'000, { 6, .3 }, .5, .01, motors);
  require(near(output.delta, .05) && output.slew_limited && !output.saturated, "slew counted as saturation");
  require(output.i == 0, "slew saturation not included in anti-windup");
  motors = output.motor;
  for (int n = 0; n < 20; ++n) {
    output = checkedStep(controller, feedback(_c, 0), 1'000'000'000, { 0, -.3 }, .5, .01, motors);
    for (int i = 0; i < 4; ++i) {
      require(std::abs(output.motor[i] - motors[i]) <= .0500001, "motor step exceeded slew limit");
    }
    require(near(output.motor[0], output.motor[2]) && near(output.motor[1], output.motor[3]), "axis allocation changed");
    motors = output.motor;
  }
  require(near(output.delta, -.3) && !output.saturated, "slew incorrectly caps final amplitude");
  output = checkedStep(controller, feedback(_c, 0), 1'000'000'000, { 0, .3 }, .5, .005, motors);
  require(near(output.delta, -.275), "slew not scaled to control period");
  // 長時間連続してslewに当たっても、飽和停止はしない。
  motors = output.motor;
  for (int n = 0; n < 100; ++n) {
    output = checkedStep(controller, feedback(_c, 0), 1'000'000'000, { 0, n % 2 ? -.3 : .3 }, .5, .01, motors);
    require(output.slew_limited && !output.saturated, "slew-only operation triggered saturation");
    motors = output.motor;
  }
  _c.pid.motor_slew_time_s = -1;
  rejects([&] { validateSession(_c); }, "negative slew time accepted");
  _c.pid.motor_slew_time_s = 5;
  rejects([&] { validateSession(_c); }, "infeasible ramp/slew combination accepted");
}

void testMotorSaturation(SessionConfig _c)
{
  _c.pid.motor_slew_time_s = .2;  // Exercise saturation independently of active slew.
  _c.pid.ki = 0;
  _c.pid.saturation_timeout_s = .02;
  AxisController controller(_c.hardware, _c.pid);
  std::array<float, 4> motors{ .4F, .4F, .4F, .4F };
  auto output = checkedStep(controller, feedback(_c, 0), 1'000'000'000, { 0, .4 }, .4, .01, motors);
  require(!output.saturated && output.slew_limited, "exact output boundary counted as saturation");
  output = checkedStep(controller, feedback(_c, 0), 1'000'000'000, { 0, .45 }, .4, .01, output.motor);
  require(output.saturated && output.slew_limited, "output saturation masked by slew");
  require(near(output.requested[0], .85), "unclipped request not retained");
  for (float motor : output.motor) {
    require(motor >= 0 && motor <= .8000001, "unsafe motor output sent");
  }
  // slewだけに戻ると飽和の連続時間はリセットされる。
  output = checkedStep(controller, feedback(_c, 0), 1'000'000'000, { 0, -.3 }, .4, .01, output.motor);
  require(!output.saturated && output.slew_limited, "saturation did not clear");
  motors = output.motor;
  rejects(
    [&]
    {
      for (int n = 0; n < 5; ++n) {
        output = checkedStep(controller, feedback(_c, 0), 1'000'000'000, { 0, .45 }, .4, .01, motors);
        motors = output.motor;
      }
    },
    "sustained unsafe request accepted");
  AxisController lower(_c.hardware, _c.pid);
  output = checkedStep(lower, feedback(_c, 0), 1'000'000'000, { 0, -.25 }, .2, .01, { .2F, .2F, .2F, .2F });
  require(output.saturated && output.requested[0] < 0, "lower output bound not monitored");
}

void testSettling(SessionConfig _c)
{
  require(near(_c.settled_window_s, .15) && near(_c.settled_dwell_s, .5), "settling timing defaults changed");
  SettlingMonitor monitor(_c);
  for (int n = 0; n < 14; ++n) {
    monitor.update(.7, 1.5, false);
    require(!monitor.windowReady() && monitor.dwell() == 0, "partial window counted as steady");
  }
  for (int n = 0; n < 49; ++n) {
    monitor.update(.7, 1.5, false);
    require(!monitor.settled(), "settling dwell ended early");
  }
  monitor.update(.7, 1.5, false);
  require(monitor.settled(), "steady window never settled");
  monitor.update(1.1, 2.1, false);
  require(monitor.settled(), "small one-sample excursion reset RMS settling");
  monitor.update(.7, 1.5, true);
  require(!monitor.settled() && monitor.dwell() == 0, "motor saturation did not reset dwell");
  monitor.reset();
  require(!monitor.windowReady() && monitor.errorRms() == 0 && monitor.dwell() == 0, "segment state leaked");
  const double excessive_error = _c.settled_error_deg + 1;
  const double excessive_rate = _c.settled_rate_deg_s + 1;
  for (int n = 0; n < 100; ++n) {
    monitor.update(n % 2 ? -excessive_error : excessive_error,
                   n % 2 ? -excessive_rate : excessive_rate, false);
  }
  require(
    !monitor.settled() && near(monitor.errorRms(), excessive_error) &&
      near(monitor.rateRms(), excessive_rate), "oscillation cancelled in average");
  // 角度が中心でもrateだけで未整定にできる。
  for (int n = 0; n < 100; ++n) {
    monitor.update(0, excessive_rate, false);
  }
  require(!monitor.settled(), "rate criterion ignored");
  for (int n = 0; n < 100; ++n) {
    monitor.update(.5, 1, false);
  }
  require(monitor.settled(), "RMS window did not recover");
  monitor.update(std::numeric_limits<double>::quiet_NaN(), 0, false);
  require(!monitor.windowReady() && !monitor.settled(), "invalid sample accepted");
  _c.settled_window_s = 0;
  rejects([&] { validateSession(_c); }, "empty RMS window accepted");
  _c.settled_window_s = .6;
  rejects([&] { validateSession(_c); }, "excessive RMS window accepted");
}

void testStartupTargetPolicy(SessionConfig _c)
{
  _c.pid.angle_min_deg = -30;
  _c.pid.angle_max_deg = 10;
  _c.pid.tracking_limit_deg = 8;  // This test deliberately exercises a tighter profile.
  AxisController controller(_c.hardware, _c.pid);
  const ControlPolicy low_speed{ .check_tracking = false, .enable_integral = false };
  auto o = checkedStep(controller, feedback(_c, 0), 1'000'000'000, { -15, 0 }, .01, .01, {}, low_speed);
  require(o.p < 0 && o.i == 0 && o.delta < 0, "startup does not drive toward final target with frozen I");
  o = checkedStep(controller, feedback(_c, 0, 2), 1'000'000'000, { -15, 0 }, .02, .01, o.motor, low_speed);
  require(o.d < 0 && o.i == 0, "startup damping absent or I accumulated");
  o = checkedStep(
    controller,
    feedback(_c, -14),
    1'000'000'000,
    { -15, 0 },
    .2,
    .01,
    { .2F, .2F, .2F, .2F },
    { .check_tracking = false, .enable_integral = true });
  require(o.i < 0, "integral did not enable at control-ready output");
  const double integral = o.i;
  o = checkedStep(controller, feedback(_c, -14), 1'000'000'000, { -15, 0 }, .2, .01, o.motor, low_speed);
  require(o.i == integral, "freeze erased or changed learned integral");
  rejects(
    [&] { checkedStep(controller, feedback(_c, 0), 1'000'000'000, { -15, 0 }, .2, .01, o.motor); },
    "normal tracking guard not restored");
  rejects(
    [&] { checkedStep(controller, feedback(_c, 11), 1'000'000'000, { -15, 0 }, .2, .01, o.motor, low_speed); },
    "startup bypassed upper angle safety limit");
  rejects(
    [&] { checkedStep(controller, feedback(_c, -31), 1'000'000'000, { -15, 0 }, .2, .01, o.motor, low_speed); },
    "startup bypassed lower angle safety limit");
  rejects(
    [&] { checkedStep(controller, feedback(_c, 0, 25), 1'000'000'000, { -15, 0 }, .2, .01, o.motor, low_speed); },
    "startup bypassed rate safety limit");
  _c.pid.saturation_timeout_s = .02;
  AxisController limited(_c.hardware, _c.pid);
  rejects(
    [&]
    {
      for (int n = 0; n < 5; ++n) {
        checkedStep(limited, feedback(_c, 0), 1'000'000'000, { -15, 0 }, .01, .01, {}, low_speed);
      }
    },
    "startup bypassed sustained saturation");
}

void testPid(SessionConfig _c)
{
  _c.pid.saturation_timeout_s = 5;
  AxisController pid(_c.hardware, _c.pid);
  const std::array<float, 4> previous{ .2F, .2F, .2F, .2F };
  auto o = checkedStep(pid, feedback(_c, 0), 1'000'000'000, { 2, 0 }, .2, .01, previous);
  require(o.p > 0 && o.i > 0 && o.d == 0, "PID sign and derivative on measurement");
  require(near(o.motor[0] - o.motor[1], 2 * o.delta), "single scalar target-axis allocation");
  require(near(o.motor[0], o.motor[2]) && near(o.motor[1], o.motor[3]), "non-target correction absent");
  const double integral = o.i;
  o = checkedStep(pid, feedback(_c, 2), 1'000'000'000, { 2, 0 }, .2, .01, o.motor);
  require(near(o.i, integral), "integral survives segment changes");
  o = checkedStep(pid, feedback(_c, 2, 3), 1'000'000'000, { 2, 0 }, .2, .01, o.motor);
  require(o.d < 0, "gyro damping sign");
  const auto i = o.i;
  for (int n = 0; n < 100; ++n) {
    o = checkedStep(pid, feedback(_c, 0), 1'000'000'000, { 8, .02 }, .2, .01, o.motor);
  }
  require(near(o.i, i), "external excitation included in anti-windup");
  require(!o.saturated && o.delta <= _c.pid.differential_limit, "optional differential cap counted as motor saturation");
  AxisController ramp(_c.hardware, _c.pid);
  o = checkedStep(ramp, feedback(_c, 0), 1'000'000'000, { 1, 0 }, .001, .01, {});
  require(o.i == 0 && o.motor[1] >= 0, "startup PID active but integral frozen without authority");
  o = checkedStep(ramp, feedback(_c, 0), 1'000'000'000, { 1, 0 }, 0, .01, o.motor);
  for (float v : o.motor) {
    require(v == 0, "PID cannot restart motors at zero collective");
  }
}
void testYawGeometry(SessionConfig _c)
{
  _c.pid.axis = "YAW";
  _c.pid.angle_min_deg = -200;
  _c.pid.angle_max_deg = 200;
  _c.pid.zero_quaternion = { std::cos(std::numbers::pi / 12), 0, -std::sin(std::numbers::pi / 12), 0 };
  AxisController yaw(_c.hardware, _c.pid);
  std::array<float, 4> motors{ .2F, .2F, .2F, .2F };
  for (double angle : { 170., 179., 181., 190., 180., 170. }) {
    auto f = feedback(_c, angle, 2);
    if (angle == 181) {
      for (auto& v : f.quaternion) {
        v = -v;
      }
    }
    const auto o = checkedStep(yaw, f, 1'000'000'000, { angle, 0 }, .2, .01, motors);
    require(near(o.angle_deg, angle), "yaw unwrap at 180 with quaternion sign flip");
    require(near(o.swing_deg, 0, 1e-5) && near(o.rate_deg_s, 2), "tilted outer pitch is not a yaw error");
    motors = o.motor;
  }
  // Starting at the calibrated zero then moving through -180 must preserve the negative branch.
  AxisController negative(_c.hardware, _c.pid);
  for (double angle : { 0., -90., -179., -181., -190. }) {
    const auto o = checkedStep(negative, feedback(_c, angle), 1'000'000'000, { angle, 0 }, .2, .01, motors);
    require(near(o.angle_deg, angle), "negative yaw branch");
  }
}
void testGuards(SessionConfig _c)
{
  const std::array<float, 4> motors{ .2F, .2F, .2F, .2F };
  auto f = feedback(_c, 0);
  rejects(
    [&] { checkedStep(AxisController(_c.hardware, _c.pid), f, 1'200'000'000, { 0, 0 }, .2, .01, motors); },
    "stale feedback accepted");
  f.valid = false;
  rejects(
    [&] { checkedStep(AxisController(_c.hardware, _c.pid), f, 1'000'000'000, { 0, 0 }, .2, .01, motors); },
    "missing feedback accepted");
  f = feedback(_c, 0);
  f.body_rate_deg_s[0] = 100;
  rejects(
    [&] { checkedStep(AxisController(_c.hardware, _c.pid), f, 1'000'000'000, { 0, 0 }, .2, .01, motors); },
    "non-target rate ignored");
  f = feedback(_c, 0);
  const double swing_half = (_c.pid.swing_limit_deg + 1) * std::numbers::pi / 360;
  f.quaternion = { std::cos(swing_half), std::sin(swing_half), 0, 0 };
  rejects(
    [&] { checkedStep(AxisController(_c.hardware, _c.pid), f, 1'000'000'000, { 0, 0 }, .2, .01, motors); },
    "non-target attitude ignored");
  f = feedback(_c, 0);
  f.quaternion[0] = std::numeric_limits<double>::quiet_NaN();
  rejects(
    [&] { checkedStep(AxisController(_c.hardware, _c.pid), f, 1'000'000'000, { 0, 0 }, .2, .01, motors); },
    "NaN accepted");
  f = feedback(_c, 0);
  rejects(
    [&] { checkedStep(AxisController(_c.hardware, _c.pid), f, 1'000'000'000, { 0, 0 }, .2, .03, motors); },
    "late control accepted");
  _c.pid.saturation_timeout_s = .02;
  AxisController pid(_c.hardware, _c.pid);
  rejects(
    [&]
    {
      for (int i = 0; i < 5; ++i) {
        checkedStep(pid, f, 1'000'000'000, { 8, .3 }, .2, .01, motors);
      }
    },
    "sustained saturation accepted");
}
void testStartupFeedbackGuards(SessionConfig _c)
{
  const ControlPolicy startup{ .check_tracking = false, .enable_integral = false };
  const std::array<float, 4> motors{ .1F, .1F, .1F, .1F };
  const auto rejects_feedback = [&](const Feedback& f, std::int64_t now)
  {
    rejects(
      [&] { checkedStep(AxisController(_c.hardware, _c.pid), f, now, { 0, 0 }, .1, .01, motors, startup); },
      "startup policy bypassed feedback safety");
  };
  auto f = feedback(_c, 0);
  rejects_feedback(f, f.received_ns - 1);
  rejects_feedback(f, f.received_ns + static_cast<std::int64_t>(_c.pid.feedback_timeout_s * 1e9) + 1);
  f.valid = false;
  rejects_feedback(f, 1'000'000'000);
  for (int axis = 0; axis < 3; ++axis) {
    for (double bad : { std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity() }) {
      f = feedback(_c, 0);
      f.body_rate_deg_s[axis] = bad;
      rejects_feedback(f, f.received_ns);
    }
  }
  f = feedback(_c, 0);
  f.quaternion = { 0, 0, 0, 0 };
  rejects_feedback(f, f.received_ns);
}

void testSettlingInvalidation(const SessionConfig& _c)
{
  SettlingMonitor monitor(_c);
  const int warmup = static_cast<int>(std::ceil((_c.settled_window_s + _c.settled_dwell_s) * kRateHz)) + 2;
  for (int signal = 0; signal < 2; ++signal) {
    for (double bad : { std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity() }) {
      for (int n = 0; n < warmup; ++n) {
        monitor.update(0, 0, false);
      }
      require(monitor.settled(), "settling invalidation precondition");
      monitor.update(signal == 0 ? bad : 0, signal == 1 ? bad : 0, false);
      require(!monitor.settled() && !monitor.windowReady() && monitor.dwell() == 0,
              "invalid measurement retained settling history");
      monitor.update(0, 0, false);
      require(!monitor.settled() && !monitor.windowReady(), "settling recovered without fresh window");
    }
  }
}

void testReferences(SessionConfig _c)
{
  Segment move;
  move.kind = "MOVE";
  move.start_deg = -15;
  move.end_deg = -25;
  move.duration_s = 1.875 * 10 / 30;
  require(near(segmentReference(move, 0).angle_deg, -15), "move starts at origin");
  require(near(segmentReference(move, move.duration_s).angle_deg, -25), "move endpoint");
  const double mid = move.duration_s / 2, eps = 1e-5;
  const double rate =
    (segmentReference(move, mid + eps).angle_deg - segmentReference(move, mid - eps).angle_deg) / (2 * eps);
  require(near(rate, -30, 1e-5), "quintic peak rate");
  move.kind = "SCAN";
  move.start_deg = 0;
  move.end_deg = 45;
  move.duration_s = 23.5;
  require(near(segmentReference(move, 23.5).angle_deg, 45), "scan endpoint");
  for (double t = .01; t < 23.5; t += .01) {
    require(
      (segmentReference(move, t).angle_deg - segmentReference(move, t - .01).angle_deg) / .01 <= 2.000001,
      "scan exceeds 2 deg/s");
  }
  move.duration_s = 6.625;
  require(near(segmentReference(move, move.duration_s).angle_deg, 45), "fast scan endpoint");
  for (double t = .01; t < move.duration_s; t += .01) {
    require(
      (segmentReference(move, t).angle_deg - segmentReference(move, t - .01).angle_deg) / .01 <= 8.000001,
      "fast scan exceeds 8 deg/s");
  }
  const auto& s = _c.segments[3];
  require(
    near(segmentReference(s, 0).external, 0) && near(segmentReference(s, s.duration_s).external, 0),
    "external smooth endpoints");
  _c.segments[0].segment_id = _c.segments[1].segment_id;
  rejects([&] { validateSession(_c); }, "duplicate segment accepted");
}
void testClosedLoopBias(SessionConfig _c)
{
  // Simple synthetic one-axis rig: unknown constant nose-up torque + damping.
  // This is a regression test for disturbance rejection, not an X500 model.
  _c.pid.kp = .005;
  _c.pid.ki = .001;
  _c.pid.kd = .006;
  _c.pid.differential_limit = .08;
  _c.pid.integral_limit = .05;
  _c.pid.saturation_timeout_s = 5;
  _c.pid.tracking_limit_deg = 9;
  AxisController controller(_c.hardware, _c.pid);
  double angle = 0, rate = 0;
  std::array<float, 4> motor{ .2F, .2F, .2F, .2F };
  ControlOutput out;
  for (int n = 0; n < 3000; ++n) {
    out = checkedStep(controller, feedback(_c, angle, rate), 1'000'000'000, { 0, 0 }, .2, .01, motor);
    motor = out.motor;
    rate += (100 * out.delta + 1 - .8 * rate) * .01;
    angle += rate * .01;
  }
  require(std::abs(angle) < .15 && std::abs(rate) < .05, "PID fails to reject synthetic nose-up bias");
  require(out.i < -.008, "integrator should learn nose-down compensation");
}

void testShutdownAndStartupConfig(SessionConfig _c)
{
  const double dt = .01;
  const double duration = smoothStopDuration(.2, .1, 2);
  require(near(duration, 3.75), "shutdown exceeds configured peak rate");
  double collective = .2;
  for (int n = 0; n <= 375; ++n) {
    const double next = smoothStopCollective(.2, n * dt, duration);
    require(
      next <= collective + 1e-12 && next >= 0 && collective - next <= .0010001, "shutdown not monotonic/rate bounded");
    collective = next;
  }
  require(collective == 0, "shutdown did not reach zero");
  require(.2 - smoothStopCollective(.2, dt, duration) < 1e-6, "shutdown starts abruptly");
  require(smoothStopCollective(.2, duration - dt, duration) < 1e-6, "shutdown ends abruptly");
  for (double time = 0; time < duration; time += dt) {
    require(smoothStopCollective(.2, time, duration) >= 0, "floating-point endpoint produced negative motor command");
  }

  // The loaded Pitch shutdown must fade the four *actual* outputs, including
  // a holding differential, without a first-step jump or a motor slew spike.
  const std::array<float, 4> initial_motor{ .245F, .355F, .245F, .355F };
  const double motor_duration = smoothStopDuration(.355, .1, 6);
  auto previous_motor = smoothStopMotors(initial_motor, 0, motor_duration);
  for (int n = 1; n <= static_cast<int>(std::ceil(motor_duration / dt)); ++n) {
    const auto next_motor = smoothStopMotors(initial_motor, n * dt, motor_duration);
    for (std::size_t i = 0; i < next_motor.size(); ++i) {
      require(next_motor[i] >= 0 && next_motor[i] <= previous_motor[i] + 1e-12,
              "loaded shutdown motor fade not monotonic");
      require(previous_motor[i] - next_motor[i] <= .001001,
              "loaded shutdown exceeded configured peak motor slew");
    }
    previous_motor = next_motor;
  }
  require(std::all_of(previous_motor.begin(), previous_motor.end(),
                      [](double value) { return value == 0; }),
          "loaded shutdown did not reach four zero motor commands");
  require(smoothStopMotors(initial_motor, dt, motor_duration)[1] > .354999,
          "loaded shutdown motor fade starts abruptly");

  _c.runtime_qualified = true;
  rejects([&] { validateSession(_c); }, "unverified control-ready range accepted for execution");
  _c.startup.control_ready_verified = true;
  validateSession(_c);

  _c.segments.front().collective = 0;
  rejects([&] { validateSession(_c); }, "zero startup collective accepted");
}

void testTelemetryWatchdog()
{
  Feedback f;
  auto message = [](std::uint32_t boot, int system = 1)
  {
    mavlink_attitude_quaternion_t a{};
    a.q1 = 1;
    a.time_boot_ms = boot;
    a.yawspeed = 1;
    mavlink_message_t m{};
    mavlink_msg_attitude_quaternion_encode(system, 1, &m, &a);
    return m;
  };
  updateAttitudeFeedback(f, message(100, 2), 1, 1, 1000);
  require(!f.valid, "wrong MAVLink source accepted");
  updateAttitudeFeedback(f, message(100), 1, 1, 2000);
  require(f.valid && near(f.body_rate_deg_s[2], 180 / std::numbers::pi), "telemetry rate units");
  updateAttitudeFeedback(f, message(100), 1, 1, 9000);
  require(f.received_ns == 2000, "duplicate timestamp refreshed freshness");
  updateAttitudeFeedback(f, message(1), 1, 1, 10000);
  require(f.fault && !f.valid, "PX4 clock restart not latched");
  updateAttitudeFeedback(f, message(200), 1, 1, 11000);
  require(!f.valid, "faulted clock recovered during same session");
  f = {};
  updateAttitudeFeedback(f, message(0xfffffff0U), 1, 1, 1);
  updateAttitudeFeedback(f, message(10), 1, 1, 2);
  require(f.valid && !f.fault && f.boot_ms == 10, "uint32 boot-clock rollover rejected");
}
}  // namespace
void testFixedHoldMode(SessionConfig c)
{
  c.role = "PILOT";
  c.control_mode = "FIXED_HOLD";
  c.runtime_qualified = false;
  c.pid.kp = c.pid.ki = c.pid.kd = 0;
  c.hardware.motor_min = 0;
  c.hardware.motor_max = .12;
  c.hardware.startup_ramp_rate = .1;
  c.startup.integral_enable_collective = .1;
  c.startup.control_ready_verified = false;
  c.maximum_duration_s = 30;
  c.maximum_hold_s = 5;
  Segment hold;
  hold.condition_id = "fixed";
  hold.segment_id = "fixed-hold";
  hold.hold_id = "hold";
  hold.kind = "HOLD";
  hold.duration_s = 5;
  hold.start_deg = hold.end_deg = c.pid.zero_angle_deg;
  hold.collective = .1;
  hold.require_settled = true;
  c.segments = {hold};
  validateSession(c);
  auto bad = c;
  bad.pid.kp = .001;
  rejects([&] { validateSession(bad); }, "fixed mode accepted PID");
  bad = c;
  bad.segments[0].end_deg += .5;
  rejects([&] { validateSession(bad); }, "fixed mode accepted angle motion");
  bad = c;
  bad.control_mode = "TYPO";
  rejects([&] { validateSession(bad); }, "unknown mode accepted");
  bad = c;
  bad.control_mode = "AXIS_PID";
  rejects([&] { validateSession(bad); }, "axis PID accepted zero proportional gain");
  c.runtime_qualified = true;
  rejects([&] { validateSession(c); }, "fixed mode accepted missing mechanical lock");
  c.mechanically_locked_verified = true;
  validateSession(c);  // Fixed input does not claim dynamic control authority.
}

void testPx4PitchMoment(SessionConfig c)
{
  require(c.hardware.axis_mix == std::array<double, 4>{1, -1, 1, -1}, "wrong PX4 Pitch example mix");
  c.pid.zero_quaternion = {1, 0, 0, 0};
  c.pid.zero_angle_deg = 0;
  c.pid.kp = .001;
  c.pid.ki = 0;
  c.pid.kd = .0005;
  c.pid.motor_slew_time_s = 0;
  for (double direction : {-1., 1.}) {
    AxisController controller(c.hardware, c.pid);
    const auto output = checkedStep(controller, feedback(c, direction, direction),
      1'000'000'000, {0, 0}, .1, .01, {.1F, .1F, .1F, .1F});
    const auto& m = output.motor;
    // Unit arm lengths and thrust proportional to command for sign checks only.
    const double pitch_moment = m[0] + m[2] - m[1] - m[3];
    const double roll_moment = m[1] + m[2] - m[0] - m[3];
    require(direction * pitch_moment < 0, "Pitch correction does not oppose deviation");
    require(near(roll_moment, 0), "Pitch correction introduces Roll moment");
    require(near(m[0], m[2]) && near(m[1], m[3]), "front/rear motor pairs disagree");
  }
}

void testPitchWideDifferential(SessionConfig c)
{
  c.hardware.motor_min = 0;
  c.hardware.motor_max = .12;
  c.pid.additional_limits_enabled = true;
  c.pid.differential_limit = .3;
  c.pid.ki = .0001;
  c.pid.integral_limit = .002;
  for (double collective : {.005, .04, .06, .1}) {
    for (double sign : {-1., 1.}) {
      AxisController controller(c.hardware, c.pid);
      const auto output = checkedStep(controller, feedback(c, 0), 1'000'000'000,
                                      {sign, sign * .3}, collective, .01, {});
      const double feasible = std::min(collective, .12 - collective);
      require(near(output.delta, sign * feasible), "wide differential ignored physical motor headroom");
      require(output.i == 0 && output.saturated, "wide differential lost output anti-windup");
      for (float motor : output.motor) {
        require(motor >= 0 && motor <= .1200001, "wide differential exceeded individual motor limits");
      }
    }
  }
}

void testPitchIntegralLimits(SessionConfig c)
{
  c.hardware.motor_max = .12;
  c.pid.additional_limits_enabled = true;
  c.pid.kp = .001;
  c.pid.ki = .0001;
  c.pid.kd = .0005;
  c.pid.integral_limit = .002;
  c.pid.differential_limit = .005;
  AxisController controller(c.hardware, c.pid);
  const ControlPolicy frozen{.check_tracking = false, .enable_integral = false};
  auto output = checkedStep(controller, feedback(c, 0), 1'000'000'000,
                            {1, 0}, .05, .01, {}, frozen);
  require(output.i == 0, "low-collective Pitch I accumulated");
  for (int n = 0; n < 1000; ++n) {
    output = checkedStep(controller, feedback(c, 0), 1'000'000'000,
                         {1, 0}, .1, .01, output.motor);
  }
  require(near(output.i, .001, 1e-9), "Pitch I scaling is not Ki*error*time");
  const double learned = output.i;
  output = checkedStep(controller, feedback(c, 0), 1'000'000'000,
                       {1, 0}, .05, .01, output.motor, frozen);
  require(output.i == learned, "freeze erased learned Pitch trim");
  for (int n = 0; n < 2500; ++n) {
    output = checkedStep(controller, feedback(c, 0), 1'000'000'000,
                         {1, 0}, .1, .01, output.motor);
  }
  require(near(output.i, .002, 1e-9), "Pitch integral limit not enforced");
  // Unwind from the I cap before testing the differential anti-windup, so an
  // unchanged I cannot be explained merely by clipping at integral_limit.
  output = checkedStep(controller, feedback(c, 0), 1'000'000'000,
                       {-1, .008}, .1, .01, output.motor);
  require(output.i < .002 && near(output.delta, .005), "saturated Pitch I did not unwind");
  const double before = output.i;
  output = checkedStep(controller, feedback(c, 0), 1'000'000'000,
                       {1, .006}, .1, .01, output.motor);
  require(output.i == before && near(output.delta, .005), "Pitch I accumulated into differential saturation");
}

void testWideAngleGuards(SessionConfig c)
{
  require(c.pid.angle_min_deg == -90 && c.pid.angle_max_deg == 90 &&
            c.pid.tracking_limit_deg == 90, "working Pitch bounds are not 90 degrees");
  c.pid.zero_quaternion = {1, 0, 0, 0};
  c.pid.zero_angle_deg = 0;
  c.pid.ki = 0;
  const std::array<float, 4> motors{.1F, .1F, .1F, .1F};
  const ControlPolicy startup{.check_tracking = false, .enable_integral = false};
  for (double sign : {-1., 1.}) {
    // Stay just inside the angle boundary to avoid floating-point round-off
    // from the quaternion conversion; exact tracking boundary uses theta=0.
    checkedStep(AxisController(c.hardware, c.pid), feedback(c, sign * 89.99),
                1'000'000'000, {0, 0}, .1, .01, motors);
    rejects([&] { checkedStep(AxisController(c.hardware, c.pid), feedback(c, sign * 90.01),
                              1'000'000'000, {0, 0}, .1, .01, motors, startup); },
            "startup accepted angle outside 90 degrees");
    checkedStep(AxisController(c.hardware, c.pid), feedback(c, 0),
                1'000'000'000, {sign * 90, 0}, .1, .01, motors);
    // Actual and reference angles are individually valid, but error is 91 deg.
    checkedStep(AxisController(c.hardware, c.pid), feedback(c, sign * 46),
                1'000'000'000, {-sign * 45, 0}, .1, .01, motors, startup);
    rejects([&] { checkedStep(AxisController(c.hardware, c.pid), feedback(c, sign * 46),
                              1'000'000'000, {-sign * 45, 0}, .1, .01, motors); },
            "tracking error outside 90 degrees accepted after acquisition");
  }

  // A freely rotating Yaw fixture can start on the far side of the registered
  // front.  Only Yaw may use the full circular range; Pitch remains capped at
  // the established +/-90 deg envelope.
  c.pid.axis = "YAW";
  c.pid.angle_min_deg = -180;
  c.pid.angle_max_deg = 180;
  c.pid.swing_limit_deg = 180;
  c.pid.tracking_limit_deg = 180;
  validateSession(c);
  c.pid.axis = "PITCH";
  rejects([&] { validateSession(c); }, "Pitch accepted Yaw-only 180 degree swing envelope");
}

void testYawStartupSaturationPolicy(SessionConfig c)
{
  c.pid.axis = "YAW";
  c.pid.law = "ANGLE_RATE_PID";
  c.pid.cascade.attitude_kp = 2;
  c.pid.cascade.kp = .14;
  c.pid.cascade.ki = .07;
  c.pid.cascade.kd = 0;
  c.pid.cascade.target_rate_limit_deg_s = 15;
  c.pid.additional_limits_enabled = true;
  c.pid.motor_slew_time_s = 0;
  c.pid.differential_limit = .15;
  c.pid.saturation_timeout_s = .02;
  c.pid.rate_limit_deg_s = 60;
  c.pid.angle_min_deg = -90;
  c.pid.angle_max_deg = 90;
  c.hardware.motor_min = 0;
  c.hardware.motor_max = .8;
  c.hardware.axis_mix = {1, 1, -1, -1};
  AxisController controller(c.hardware, c.pid);
  const ControlPolicy startup{.check_tracking = false, .enable_integral = false,
                              .check_saturation_timeout = false};
  std::array<float, 4> previous{};
  for (int n = 0; n < 10; ++n) {
    const auto output = checkedStep(controller, feedback(c, 5), 1'000'000'000,
                                    {0, 0}, .01, .01, previous, startup);
    require(output.saturated && output.authority_limited,
            "Yaw startup did not retain low-thrust saturation diagnostics");
    require(output.i == 0, "Yaw startup integrated while authority was unavailable");
    for (const float motor : output.motor) {
      require(motor >= 0 && motor <= .8F, "Yaw startup bypassed motor output bounds");
    }
    previous = output.motor;
  }
  rejects([&] { checkedStep(controller, feedback(c, 0, 61), 1'000'000'000,
                             {0, 0}, .01, .01, previous, startup); },
          "Yaw startup bypassed the measured angular-rate limit");
  rejects([&] { checkedStep(controller, feedback(c, 91), 1'000'000'000,
                             {0, 0}, .01, .01, previous, startup); },
          "Yaw startup bypassed the measured joint-angle limit");
  rejects([&] {
    for (int n = 0; n < 5; ++n) {
      previous = checkedStep(controller, feedback(c, 5, -55), 1'000'000'000,
                             {10, .2}, .3, .01, previous).motor;
    }
  }, "Yaw saturation timeout was not restored after the startup ramp");
}

void testFixedInputMix()
{
  const std::array<std::array<double, 4>, 4> mix{{
    {{1, 1, 1, 1}}, {{1, -1, -1, 1}}, {{1, 1, -1, -1}}, {{1, -1, 1, -1}}
  }};
  const auto motors = fixedMotorCommands(mix, {.3, .01, .02, .03});
  require(near(motors[0], .36, 1e-7) && near(motors[1], .30, 1e-7) &&
            near(motors[2], .26, 1e-7) && near(motors[3], .28, 1e-7),
          "fixed C/P/R/Y Quad-X allocation is wrong");
  Segment sine;
  sine.kind = "SINE";
  sine.duration_s = 10;
  sine.frequency_hz = .1;
  sine.phase_rad = std::numbers::pi / 2;
  sine.fixed_input = {.3, 0, 0, 0};
  sine.fixed_amplitude = {0, .01, 0, 0};
  const auto input = fixedInputReference(sine, 0);
  require(near(input[0], .3) && near(input[1], .01), "fixed sine reference is wrong");
}

void testFixedInputReferenceCapture(SessionConfig c)
{
  c.role = "T";
  c.repetition = 1;
  c.control_mode = "FIXED_INPUT";
  c.runtime_qualified = true;
  c.mechanically_locked_verified = true;
  c.fixed_input_mix = {{{{1, 1, 1, 1}}, {{1, -1, -1, 1}},
                        {{1, 1, -1, -1}}, {{1, -1, 1, -1}}}};
  c.pid.kp = c.pid.ki = c.pid.kd = 0;
  c.pid.zero_angle_deg = 0;
  c.pid.integral_limit = 0;
  c.hardware.motor_min = 0;
  c.hardware.motor_max = .8;
  c.hardware.normal_ramp_rate = .3;
  c.startup.capture_reference = true;
  c.startup.control_ready_verified = false;
  c.maximum_duration_s = 30;
  c.maximum_hold_s = 5;
  Segment hold;
  hold.condition_id = "fixed-reference";
  hold.segment_id = "fixed-reference-hold";
  hold.hold_id = "hold";
  hold.kind = "HOLD";
  hold.duration_s = 5;
  hold.collective = .3;
  hold.fixed_input = {.3, 0, 0, 0};
  hold.require_settled = false;
  c.segments = {hold};
  validateSession(c);
}

void testReferenceOnlySlewAllowance(SessionConfig c)
{
  c.pid.rate_limit_deg_s = 60;  // Measured body/joint stop remains unchanged.
  c.pid.reference_rate_limit_deg_s = 100;
  c.pid.cascade.trajectory_rate_feedforward = 0;
  Segment baseline = c.segments.front();
  baseline.start_deg = 0;
  baseline.collective = .3;
  baseline.kind = "HOLD";
  baseline.duration_s = 5;
  baseline.analysis_start_s = baseline.analysis_end_s = 0;
  baseline.require_settled = true;
  Segment sine;
  sine.condition_id = baseline.condition_id;
  sine.segment_id = "reference-only-slew-sine";
  sine.kind = "SINE";
  sine.duration_s = 10;
  sine.start_deg = 0;
  sine.amplitude_deg = 30;
  sine.frequency_hz = .5;
  sine.fade_s = 1;
  sine.collective = .3;
  sine.analysis_start_s = 1;
  sine.analysis_end_s = 9;
  Segment final_hold = baseline;
  final_hold.segment_id = "reference-only-slew-final";
  final_hold.hold_id = final_hold.segment_id;
  c.segments = {baseline, sine, final_hold};
  validateSession(c);
  c.pid.reference_rate_limit_deg_s = 60;
  rejects([&] { validateSession(c); }, "94 deg/s sine accepted by default 60 deg/s reference limit");
  c.pid.reference_rate_limit_deg_s = 101;
  rejects([&] { validateSession(c); }, "reference slew allowance above 100 deg/s accepted");
}

int main(int _argc, char** _argv)
{
  try {
    if (_argc != 2) {
      throw std::runtime_error("test_session requires example config path");
    }
    auto c = loadSession(_argv[1]);
    testPitchWideDifferential(c);
    testPitchIntegralLimits(c);
    testWideAngleGuards(c);
    testFixedInputMix();
    testFixedInputReferenceCapture(c);
    testReferenceOnlySlewAllowance(c);
    testYawStartupSaturationPolicy(c);
    testPx4PitchMoment(c);
    testFixedHoldMode(c);
    testWithoutAdditionalLimits(c);
    testMotorSlew(c);
    testMotorSaturation(c);
    testSettling(c);
    testStartupTargetPolicy(c);
    testStartupFeedbackGuards(c);
    testSettlingInvalidation(c);
    testShutdownAndStartupConfig(c);
    // 追加制限を有効にした既存プロファイルも引き続き検証する。
    c.pid.additional_limits_enabled = true;
    c.hardware.motor_min = .1;
    c.hardware.motor_max = .4;
    testPid(c);
    testYawGeometry(c);
    testGuards(c);
    testReferences(c);
    testClosedLoopBias(c);
    testTelemetryWatchdog();
    std::cout << "all session controller tests passed\n";
    return 0;
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
