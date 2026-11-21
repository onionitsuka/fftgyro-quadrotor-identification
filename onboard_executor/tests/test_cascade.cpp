#include "fftgyro_executor/session.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>

using namespace fftgyro;
namespace
{
void require(bool ok, const char* message)
{
  if (!ok) throw std::runtime_error(message);
}
Feedback sample(double angle, double rate, int n)
{
  Feedback f;
  f.valid = true;
  f.boot_ms = 1000 + n * 10;
  f.received_ns = 1'000'000'000LL + n * 10'000'000LL;
  const double half = angle * std::numbers::pi / 360;
  f.quaternion = {std::cos(half), 0, std::sin(half), 0};
  f.body_rate_deg_s = {0, rate, 0};
  return f;
}
ControlOutput step(AxisController& controller, const Feedback& f, double target,
                   double collective = .1, ControlPolicy policy = {})
{
  ControlOutput o;
  if (!controller.update(f, f.received_ns, {target, 0}, collective, .01, {}, o, policy)) {
    throw std::runtime_error(o.error);
  }
  return o;
}
void testTerms(SessionConfig c)
{
  AxisController controller(c.hardware, c.pid);
  auto o = step(controller, sample(1, 0, 0), 0);
  require(o.rate_target_deg_s < -3.99 && o.rate_target_deg_s > -4.01, "angle P conversion wrong");
  require(o.p < -.0068 && o.p > -.0071 && o.i < 0 && o.d == 0, "rate PI terms wrong");
  require(o.motor[0] == o.motor[2] && o.motor[1] == o.motor[3] && o.motor[0] < o.motor[1], "wrong Pitch mapping");
  const double integral = o.i;
  o = step(controller, sample(1, 0, 1), 0, .05, {.check_tracking = false, .enable_integral = false});
  require(o.i == integral, "freeze reset learned wind compensation");
  o = step(controller, sample(0, 0, 2), 90);
  require(o.rate_target_deg_s == 30, "rate target not bounded separately from stop guard");
}
void testDerivative(SessionConfig c)
{
  c.pid.cascade.kd = .004;
  AxisController controller(c.hardware, c.pid);
  auto o = step(controller, sample(0, 0, 0), 0);
  require(o.d == 0, "first sample derivative kick");
  const auto f = sample(0, 1, 1);
  o = step(controller, f, 0);
  require(o.d < 0 && o.angular_accel_filtered_deg_s2 > 0 && o.angular_accel_filtered_deg_s2 < 100,
          "measurement derivative/filter wrong");
  const double derivative = o.d;
  o = step(controller, f, 5);
  require(o.d == derivative, "duplicate feedback or target step changed derivative");
}
void testAuthority(SessionConfig c)
{
  AxisController controller(c.hardware, c.pid);
  const auto o = step(controller, sample(5, 0, 0), 0, .001);
  require(o.authority_limited && o.i == 0 && std::abs(o.delta) <= .001000001, "anti-windup lost at low collective");
  bool stopped = false;
  for (int n = 1; n < 250; ++n) {
    const auto f = sample(5, 0, n);
    ControlOutput out;
    if (!controller.update(f, f.received_ns, {0, 0}, .001, .01, {}, out)) {
      require(std::string(out.error) == "sustained saturation", "wrong authority failure");
      stopped = true;
      break;
    }
  }
  require(stopped, "sustained impossible request did not stop");
}
void testCollectiveGainCompensation(SessionConfig c)
{
  auto& gain = c.pid.cascade;
  gain.collective_gain_compensation = true;
  gain.effectiveness_reference_collective = .1;
  gain.thrust_curve_linear = 4.70821757;
  gain.thrust_curve_quadratic = 9.661238;
  const ControlPolicy no_integral{.check_tracking = true, .enable_integral = false};
  AxisController low(c.hardware, c.pid), high(c.hardware, c.pid);
  const auto low_output = step(low, sample(1, 0, 0), 0, .1, no_integral);
  const auto high_output = step(high, sample(1, 0, 0), 0, .3, no_integral);
  const auto slope = [&](double collective) {
    return gain.thrust_curve_linear + 2 * gain.thrust_curve_quadratic * collective;
  };
  require(std::abs(low_output.actuator_gain_scale - 1) < 1e-12, "reference collective scale is not unity");
  require(high_output.actuator_gain_scale > .63 && high_output.actuator_gain_scale < .64,
          "measured thrust-slope compensation scale wrong");
  require(std::abs(low_output.delta * slope(.1) - high_output.delta * slope(.3)) < 1e-10,
          "collective changed normalized differential thrust");
}
void testNormalizedLimits(SessionConfig c)
{
  auto& gain = c.pid.cascade;
  gain.collective_gain_compensation = true;
  gain.thrust_curve_linear = 4.70821757;
  gain.thrust_curve_quadratic = 9.661238;
  const auto force = [&](double command) {
    return gain.thrust_curve_linear * command + gain.thrust_curve_quadratic * command * command;
  };
  const ControlPolicy frozen{.check_tracking = true, .enable_integral = false};
  // Finite positive/negative differential: exact for the stated symmetric quadratic model.
  for (double target : {-5., -1., 1., 5.}) {
    AxisController reference_controller(c.hardware, c.pid);
    const auto base = step(reference_controller, sample(0, 0, 0), target, .1, frozen);
    const double force_difference = force(.1 + base.delta) - force(.1 - base.delta);
    for (double collective : {.1, .2, .3, .5, .7}) {
      AxisController controller(c.hardware, c.pid);
      const auto o = step(controller, sample(0, 0, 0), target, collective, frozen);
      require(!o.authority_limited && !o.slew_limited, "unexpected clipping in model test");
      require(std::abs(force(collective + o.delta) - force(collective - o.delta) - force_difference) < 1e-12,
              "quadratic differential thrust varies with collective or sign");
      require(o.actuator_model_in_range == (collective - std::abs(o.delta) >= .1), "wrong model domain flag");
    }
  }
  for (double collective : {0., .001, .05, .099, .1}) {
    AxisController controller(c.hardware, c.pid);
    const auto o = step(controller, sample(0, 0, 0), 1, collective, frozen);
    require(o.actuator_gain_scale == 1, "startup control boosted below identified reference");
    require(o.actuator_gain_floor_active == (collective < .1), "wrong reference floor flag");
    if (collective == 0) require(o.delta == 0 && o.motor[0] == 0 && o.motor[1] == 0, "nonzero stop output");
  }
  // Physical excitation bypasses PID normalization, but shares the real output limits.
  for (double external : {-.01, .01}) {
    AxisController controller(c.hardware, c.pid);
    const auto f = sample(0, 0, 0);
    ControlOutput o;
    require(controller.update(f, f.received_ns, {0, external}, .3, .01, {}, o, frozen), "external update failed");
    require(o.delta == external && o.virtual_delta_requested == 0, "physical excitation was rescaled");
    require(controller.update(f, f.received_ns, {1, external}, .3, .01, {}, o, frozen), "mixed update failed");
    require(std::abs(o.delta_requested - (o.actuator_gain_scale * (o.p + o.i + o.d) + external)) < 1e-14,
            "mixed demand units wrong");
  }
  AxisController integral_controller(c.hardware, c.pid);
  ControlOutput learned;
  for (int n = 0; n < 50; ++n) learned = step(integral_controller, sample(0, 0, n), 1, .3);
  require(learned.i > 0, "normalized integral did not learn");
  auto f = sample(0, 0, 50);
  ControlOutput o;
  require(integral_controller.update(f, f.received_ns, {1, .5}, .3, .01, {}, o), "saturated update failed");
  require(o.authority_limited && o.i == learned.i && o.delta <= .3, "windup at external/output saturation");
  f = sample(0, 0, 51);
  require(integral_controller.update(f, f.received_ns, {-1, .5}, .3, .01, {}, o), "unwind update failed");
  require(o.i < learned.i && o.authority_limited, "saturated I cannot unwind");
  c.pid.motor_slew_time_s = 1;
  AxisController slew_controller(c.hardware, c.pid);
  require(slew_controller.update(f, f.received_ns, {5, 0}, .3, .01, {.3F, .3F, .3F, .3F}, o), "slew update failed");
  require(o.slew_limited && o.i == 0 && std::abs(o.delta) <= .0100001, "real slew/AW limits lost");
  c.pid.motor_slew_time_s = 0;
  for (int invalid = 0; invalid < 7; ++invalid) {
    auto bad = c;
    auto& bad_gain = bad.pid.cascade;
    if (invalid == 0) bad_gain.thrust_curve_linear = 0;
    if (invalid == 1) bad_gain.thrust_curve_quadratic = -1;
    if (invalid == 2) bad_gain.thrust_curve_linear = std::numeric_limits<double>::infinity();
    if (invalid == 3) bad_gain.thrust_curve_quadratic = std::numeric_limits<double>::max();
    if (invalid == 4) bad.pid.axis = "YAW";
    if (invalid == 5) bad.pid.law = "ANGLE_PID";
    if (invalid == 6) bad_gain.thrust_curve_command_max = .4;
    bool rejected = false;
    try { validateSession(bad); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "invalid compensation configuration accepted");
    AxisController bad_controller(bad.hardware, bad.pid);
    require(!bad_controller.update(f, f.received_ns, {0, 0}, .3, .01, {}, o), "invalid model reached motors");
  }
}

void testTrajectoryFeedforward(SessionConfig c)
{
  for (const auto* kind : {"MOVE", "SCAN", "SINE", "HOLD"}) {
    Segment segment;
    segment.kind = kind;
    segment.start_deg = -1;
    segment.end_deg = 4;
    segment.duration_s = 6;
    segment.amplitude_deg = 2;
    segment.frequency_hz = .3;
    segment.phase_rad = .4;
    segment.fade_s = 1.5;
    for (int n = 1; n < 600; ++n) {
      const double time = n * .01;
      constexpr double epsilon = 1e-5;
      const double numerical = (segmentReference(segment, time + epsilon).angle_deg -
                                segmentReference(segment, time - epsilon).angle_deg) / (2 * epsilon);
      require(std::abs(segmentReference(segment, time).rate_deg_s - numerical) < 1e-7,
              "analytic trajectory derivative mismatch (including fade)");
    }
    require(segmentReference(segment, 0).rate_deg_s == 0 &&
              segmentReference(segment, segment.duration_s).rate_deg_s == 0, "nonzero trajectory endpoint rate");
  }
  const auto f = sample(0, 0, 0);
  ControlOutput o;
  AxisController legacy(c.hardware, c.pid);
  require(legacy.update(f, f.received_ns, {0, 0, 3}, .3, .01, {}, o) && o.rate_target_deg_s == 0,
          "feedforward changed default legacy behavior");
  c.pid.cascade.trajectory_rate_feedforward = 1;
  AxisController controller(c.hardware, c.pid);
  require(controller.update(f, f.received_ns, {0, 0, 3}, .3, .01, {}, o) && o.rate_target_deg_s == 3,
          "trajectory rate not fed forward at zero angle error");
  require(controller.update(f, f.received_ns, {90, 0, 3}, .3, .01, {}, o) &&
            o.rate_target_deg_s == c.pid.cascade.target_rate_limit_deg_s, "FF bypasses rate limit");
  require(!controller.update(f, f.received_ns, {0, 0, std::numeric_limits<double>::quiet_NaN()}, .3, .01, {}, o),
          "nonfinite trajectory rate accepted");
}

void testNormalizedWindModel(SessionConfig c)
{
  c.pid.cascade.collective_gain_compensation = true;
  c.pid.cascade.thrust_curve_linear = 4.70821757;
  c.pid.cascade.thrust_curve_quadratic = 9.661238;
  AxisController varying(c.hardware, c.pid), fixed(c.hardware, c.pid);
  double angle = 0, rate = 0, fixed_angle = 0, fixed_rate = 0;
  ControlOutput o;
  for (int n = 0; n < 4000; ++n) {
    const double collective = .1 + .2 * quintic(n / 1000.0);
    const double target = n >= 1000 && n < 2000 ? 2 : 0;
    o = step(varying, sample(angle, rate, n), target, collective);
    const auto base = step(fixed, sample(fixed_angle, fixed_rate, n), target, .1);
    const double slope_ratio = (4.70821757 + 2 * 9.661238 * collective) / (4.70821757 + 2 * 9.661238 * .1);
    // Only an exact static model, not validation of the real motor lag/rig.
    rate += (500 * (slope_ratio * o.delta + .02) - rate - 2 * angle) * .01;
    fixed_rate += (500 * (base.delta + .02) - fixed_rate - 2 * fixed_angle) * .01;
    angle += rate * .01;
    fixed_angle += fixed_rate * .01;
    require(std::abs(angle - fixed_angle) < 1e-9 && std::abs(o.i - base.i) < 1e-9,
            "collective ramp changes exact-model response or integral memory");
  }
  require(std::abs(angle) < .05 && std::abs(rate) < .05 && std::abs(o.i + .02) < .001,
          "normalized constant-disturbance model failed to settle");
}

void testWindModel(SessionConfig c)
{
  // Illustrative passive one-axis models, not measured rig parameters or wind speeds.
  for (double effectiveness : {200., 500., 1000.}) {
    for (double wind : {-.04, .04}) {
      AxisController controller(c.hardware, c.pid);
      double angle = 0, rate = 0;
      ControlOutput o;
      for (int n = 0; n < 4000; ++n) {
        const double target = n >= 1000 && n < 2000 ? 2 : 0;
        o = step(controller, sample(angle, rate, n), target);
        require(std::abs(o.delta) <= .1000001 && std::abs(o.i) <= .1000001, "model exceeded available authority");
        rate += (effectiveness * (o.delta + wind) - rate - 2 * angle) * .01;
        angle += rate * .01;
      }
      require(std::abs(angle) < .5 && std::abs(rate) < .5, "constant-wind model failed to settle");
      require(std::abs(o.i + wind) < .01, "steady wind bias not learned across target changes");
    }
  }
}
void testConfiguration(SessionConfig c)
{
  c.motor_mapping_verified = c.feedback_frame_verified = true;
  validatePitchHoldCommissioning(c);
  const auto rejects = [](const SessionConfig& config) {
    try { validateSession(config); } catch (const std::runtime_error&) { return; }
    throw std::runtime_error("invalid cascade config accepted");
  };
  auto bad = c;
  bad.pid.kp = .001;
  rejects(bad);
  bad = c;
  bad.pid.cascade.target_rate_limit_deg_s = c.pid.rate_limit_deg_s;
  rejects(bad);
  bad = c;
  bad.pid.cascade.derivative_cutoff_hz = 0;
  rejects(bad);
  bad = c;
  bad.segments[0].analysis_end_s = .5;
  rejects(bad);
  bad = c;
  bad.startup.minimum_duration_s = 20;
  rejects(bad);
}
}
int main(int argc, char** argv)
{
  try {
    require(argc == 2, "supply cascade config");
    const auto c = loadSession(argv[1]);
    testConfiguration(c);
    testTerms(c);
    testDerivative(c);
    testAuthority(c);
    testCollectiveGainCompensation(c);
    testNormalizedLimits(c);
    testTrajectoryFeedforward(c);
    testNormalizedWindModel(c);
    testWindModel(c);
    std::cout << "cascade, normalized thrust/limits/AW/domain, trajectory FF and illustrative wind models passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
