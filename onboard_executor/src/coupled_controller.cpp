#include "fftgyro_executor/coupled_controller.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>

namespace fftgyro {
namespace {
constexpr double rad = std::numbers::pi/180;
PidConfig jointConfig(PidConfig p) {
  p.zero_quaternion = {1,0,0,0}; p.zero_angle_deg = 0;
  // Slew is applied to common startup/shutdown, never separately to each axis.
  p.motor_slew_time_s = 0;
  return p;
}
}
CoupledController::CoupledController(const Config& hp, const PidConfig& pp,
 const Config& hy, const PidConfig& py, const std::array<double,4>& base)
 : hp_(hp), hy_(hy), pp_(jointConfig(pp)), py_(jointConfig(py)), base_(base),
 pitch_(hp_,pp_), yaw_(hy_,py_) {}

bool CoupledController::decompose(const Feedback& f, CoupledOutput& out) noexcept {
  auto a=base_, b=f.quaternion;
  if (!normalizeQuaternion(a) || !normalizeQuaternion(b)) {
    out.pitch.error="invalid coupled quaternion"; return false;
  }
  const double w=a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3];
  const double x=a[0]*b[1]-a[1]*b[0]-a[2]*b[3]+a[3]*b[2];
  const double y=a[0]*b[2]+a[1]*b[3]-a[2]*b[0]-a[3]*b[1];
  const double z=a[0]*b[3]-a[1]*b[2]+a[2]*b[1]-a[3]*b[0];
  // Intrinsic Y-Z-X decomposition for the physical Frame3(Pitch)-Frame4(Yaw)
  // nesting.  The previous Z-Y-X order generated a fictitious roll while Yaw
  // moved at nonzero Pitch (observed in e9811 on 2026-09-17).
  const double h=std::asin(std::clamp(2*(x*y+w*z),-1.0,1.0));
  const double p=std::atan2(-2*(x*z-w*y),1-2*(y*y+z*z));
  const double r=std::atan2(-2*(y*z-w*x),1-2*(x*x+z*z));
  out.roll_deg=r/rad;
  if (std::abs(p)>75*rad || std::abs(out.roll_deg)>std::min(pp_.swing_limit_deg,py_.swing_limit_deg)) {
    out.pitch.error="coupled rig geometry limit"; return false;
  }
  for (double v:f.body_rate_deg_s) {
    if (!std::isfinite(v) || std::abs(v)>std::min(pp_.rate_limit_deg_s,py_.rate_limit_deg_s)) {
      out.pitch.error="body rate limit"; return false;
    }
  }
  fp_=fy_=f;
  fp_.quaternion={std::cos(p/2),0,std::sin(p/2),0};
  fy_.quaternion={std::cos(h/2),0,0,std::sin(h/2)};
  const double pitch_rate=(std::cos(r)*f.body_rate_deg_s[1]-std::sin(r)*f.body_rate_deg_s[2])/std::cos(h);
  const double yaw_rate=std::sin(r)*f.body_rate_deg_s[1]+std::cos(r)*f.body_rate_deg_s[2];
  fp_.body_rate_deg_s={0,pitch_rate,0};
  fy_.body_rate_deg_s={0,0,yaw_rate};
  return true;
}
bool CoupledController::read(const Feedback& f, std::int64_t now, CoupledOutput& out) noexcept {
  if (!decompose(f,out)) return false;
  if (!pitch_.readJoint(fp_,now,out.pitch)) return false;
  if (!yaw_.readJoint(fy_,now,out.yaw)) {out.pitch.error=out.yaw.error; return false;}
  return true;
}
bool CoupledController::update(const Feedback& f, std::int64_t now, const Reference& p,
 const Reference& y, double c, double dt, CoupledOutput& out, ControlPolicy policy) noexcept {
  if (!decompose(f,out)) return false;
  std::array<float,4> previous{}; previous.fill(static_cast<float>(c));
  const auto shared_policy=policy;
  policy.check_saturation_timeout=false;
  if (!pitch_.update(fp_,now,p,c,dt,previous,out.pitch,policy)) return false;
  if (!yaw_.update(fy_,now,y,c,dt,previous,out.yaw,policy)) {out.pitch.error=out.yaw.error; return false;}
  const double floor=c<hp_.motor_min ? 0:hp_.motor_min;
  std::array<double,4> d{};
  double scale=1;
  for (int i=0;i<4;++i) {
    d[i]=hp_.axis_mix[i]*out.pitch.delta+hy_.axis_mix[i]*out.yaw.delta;
    if (d[i]>0) scale=std::min(scale,(hp_.motor_max-c)/d[i]);
    if (d[i]<0) scale=std::min(scale,(floor-c)/d[i]);
  }
  scale=std::clamp(scale,0.0,1.0); out.allocation_scale=scale;
  if (scale<1-1e-10) {pitch_.rejectIntegralStep(); yaw_.rejectIntegralStep();}
  const bool limited=scale<1-1e-10 || out.pitch.authority_limited || out.yaw.authority_limited;
  saturation_s_=shared_policy.check_saturation_timeout && limited ? saturation_s_+dt:0;
  if (saturation_s_>std::min(pp_.saturation_timeout_s,py_.saturation_timeout_s)) {
    out.pitch.error="coupled sustained saturation"; return false;
  }
  out.pitch.delta*=scale; out.yaw.delta*=scale;
  out.pitch.saturated|=limited; out.yaw.saturated|=limited;
  for(int i=0;i<4;++i) {
    const double sent=c+scale*d[i];
    out.pitch.requested[i]=c+hp_.axis_mix[i]*out.pitch.delta_requested+hy_.axis_mix[i]*out.yaw.delta_requested;
    // Exact cancellation at the lower bound can leave a negative roundoff bit.
    out.pitch.motor[i]=static_cast<float>(std::clamp(sent,floor,hp_.motor_max));
    if(!std::isfinite(sent) || sent<floor-1e-7 || sent>hp_.motor_max+1e-7) {
      out.pitch.error="coupled allocation bounds"; return false;
    }
  }
  return true;
}
}
