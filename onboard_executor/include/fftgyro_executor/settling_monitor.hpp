#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "fftgyro_executor/session.hpp"

namespace fftgyro
{
/** @brief 保持区間専用のRMS判定。PID・異常停止用の信号は変更しない。 */
class SettlingMonitor
{
public:
  explicit SettlingMonitor(const SessionConfig& _config)
    : config_(_config), samples_(static_cast<std::size_t>(std::ceil(_config.settled_window_s * kRateHz)))
  {
  }

  void reset() noexcept
  {
    std::fill(samples_.begin(), samples_.end(), std::array<double, 2>{});
    next_ = count_ = 0;
    error_rms_ = rate_rms_ = dwell_ = 0;
  }

  void update(double _error_deg, double _rate_deg_s, bool _saturated) noexcept
  {
    if (!std::isfinite(_error_deg) || !std::isfinite(_rate_deg_s)) {
      reset();
      return;
    }
    samples_[next_] = { _error_deg * _error_deg, _rate_deg_s * _rate_deg_s };
    next_ = (next_ + 1) % samples_.size();
    count_ = std::min(count_ + 1, samples_.size());
    double error_squared = 0, rate_squared = 0;
    for (const auto& sample : samples_) {
      error_squared += sample[0];
      rate_squared += sample[1];
    }
    error_rms_ = std::sqrt(error_squared / static_cast<double>(count_));
    rate_rms_ = std::sqrt(rate_squared / static_cast<double>(count_));
    const bool steady = windowReady() && error_rms_ <= config_.settled_error_deg &&
                        rate_rms_ <= config_.settled_rate_deg_s && !_saturated;
    dwell_ = steady ? dwell_ + 1.0 / kRateHz : 0;
  }

  bool windowReady() const noexcept
  {
    return count_ == samples_.size();
  }
  bool settled() const noexcept
  {
    return windowReady() && dwell_ + 1e-8 >= config_.settled_dwell_s;
  }
  double errorRms() const noexcept
  {
    return error_rms_;
  }
  double rateRms() const noexcept
  {
    return rate_rms_;
  }
  double dwell() const noexcept
  {
    return dwell_;
  }

private:
  const SessionConfig& config_;
  std::vector<std::array<double, 2>> samples_;
  std::size_t next_ = {}, count_ = {};
  double error_rms_ = {}, rate_rms_ = {}, dwell_ = {};
};
}  // namespace fftgyro
