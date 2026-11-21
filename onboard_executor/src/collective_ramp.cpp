#include "fftgyro_executor/collective_ramp.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>

namespace fftgyro
{
double transitionCollective(double current, double target, double max_step, bool step_up) noexcept
{
  if (step_up && target > current) return target;
  return current + std::clamp(target - current, -max_step, max_step);
}

double quintic(double _fraction) noexcept
{
  const double x = std::clamp(_fraction, 0.0, 1.0);
  return std::clamp(x * x * x * (10.0 + x * (-15.0 + 6.0 * x)), 0.0, 1.0);
}

double quinticDerivative(double _fraction) noexcept
{
  const double x = std::clamp(_fraction, 0.0, 1.0);
  return 30 * x * x * (1 - x) * (1 - x);
}

double smoothStopDuration(double _collective, double _maximum_rate, double _minimum_duration) noexcept
{
  // quinticの最大傾きは1.875。設定rateは平均でなく最大減速速度。
  return std::max(_minimum_duration, 1.875 * _collective / _maximum_rate);
}

double smoothStopCollective(double _initial, double _time, double _duration) noexcept
{
  return _initial * (1 - quintic(_time / _duration));
}

std::array<double, 4> smoothStopMotors(
  const std::array<float, 4>& _initial, double _time, double _duration) noexcept
{
  const double scale = 1 - quintic(_time / _duration);
  std::array<double, 4> result{};
  for (std::size_t i = 0; i < result.size(); ++i) {
    result[i] = static_cast<double>(_initial[i]) * scale;
  }
  return result;
}
}  // namespace fftgyro
