#pragma once

#include <array>

namespace fftgyro
{
double transitionCollective(double current, double target, double max_step, bool step_up) noexcept;
double quintic(double _fraction) noexcept;
double quinticDerivative(double _fraction) noexcept;
double smoothStopDuration(double _collective, double _maximum_rate, double _minimum_duration) noexcept;
double smoothStopCollective(double _initial, double _time, double _duration) noexcept;
std::array<double, 4> smoothStopMotors(
  const std::array<float, 4>& _initial, double _time, double _duration) noexcept;
}  // namespace fftgyro
