#pragma once
#include "fftgyro_executor/axis_controller.hpp"

namespace fftgyro {
// Measured G4.0 nesting: R = R_base Ry(pitch) Rz(yaw); product Frame2 locked.
struct CoupledOutput {
  ControlOutput pitch, yaw;
  double allocation_scale{1}, roll_deg{};
};
class CoupledController {
public:
  CoupledController(const Config& pitch_hardware, const PidConfig& pitch,
                    const Config& yaw_hardware, const PidConfig& yaw,
                    const std::array<double,4>& base);
  bool read(const Feedback&, std::int64_t, CoupledOutput&) noexcept;
  bool update(const Feedback&, std::int64_t, const Reference&, const Reference&,
              double collective, double dt, CoupledOutput&, ControlPolicy) noexcept;
private:
  Config hp_, hy_;
  PidConfig pp_, py_;
  std::array<double,4> base_;
  AxisController pitch_, yaw_;
  Feedback fp_, fy_;
  double saturation_s_{};
  bool decompose(const Feedback&, CoupledOutput&) noexcept;
};
}
