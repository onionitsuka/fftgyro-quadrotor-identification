#include "fftgyro_executor/session.hpp"
#include "fftgyro_executor/coupled_controller.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

static void check(bool v) {if(!v) throw std::runtime_error("coupled assertion failed");}
int main(int argc,char** argv) {
  check(argc==2);
  auto c=fftgyro::loadSession(argv[1]); check(bool(c.coupled_yaw));
  constexpr double rad=3.141592653589793/180;
  fftgyro::Feedback f{}; f.valid=true; f.received_ns=1'000'000'000; f.boot_ms=100;
  const double p=-30*rad,y=45*rad;
  f.quaternion={std::cos(y/2)*std::cos(p/2),std::sin(y/2)*std::sin(p/2),
                std::cos(y/2)*std::sin(p/2),std::sin(y/2)*std::cos(p/2)};
  // For R=Ry(pitch)Rz(yaw), true Pitch=3 and Yaw=7 joint rates.
  f.body_rate_deg_s={std::sin(y)*3,std::cos(y)*3,7};
  fftgyro::CoupledController controller(c.hardware,c.pid,c.coupled_yaw->hardware,c.coupled_yaw->pid,{1,0,0,0});
  fftgyro::CoupledOutput out;
  check(controller.read(f,f.received_ns,out));
  check(std::abs(out.pitch.angle_deg+30)<1e-6 && std::abs(out.yaw.angle_deg-45)<1e-6);
  check(std::abs(out.pitch.rate_deg_s-3)<1e-6 && std::abs(out.yaw.rate_deg_s-7)<1e-6);
  check(!controller.read(f,f.received_ns+200'000'000,out));
  f.body_rate_deg_s={0,0,0};
  check(controller.update(f,f.received_ns,{-25,0,0},{50,0,0},.3,.01,out,{}));
  for(float v:out.pitch.motor) check(v>=0 && v<=.8);
  // Force joint allocator saturation at low collective with independent mixes.
  c.pid.cascade.kp=20;c.coupled_yaw->pid.cascade.kp=20;
  fftgyro::CoupledController saturated(c.hardware,c.pid,c.coupled_yaw->hardware,c.coupled_yaw->pid,{1,0,0,0});
  check(saturated.update(f,f.received_ns,{-15,0,0},{60,0,0},.01,.01,out,{}));
  check(out.allocation_scale<1 && out.pitch.saturated);
  for(float v:out.pitch.motor) check(v>=-1e-7 && v<=.8);
  // Each axis alone fits, but their sum does not: reject BOTH integral steps.
  auto pp=c.pid,py=c.coupled_yaw->pid;
  pp.law=py.law="ANGLE_PID"; pp.kp=py.kp=.008;pp.ki=py.ki=.01;pp.kd=py.kd=0;
  pp.cascade.collective_gain_compensation=py.cascade.collective_gain_compensation=false;
  fftgyro::CoupledController windup(c.hardware,pp,c.coupled_yaw->hardware,py,{1,0,0,0});
  check(windup.update(f,f.received_ns,{-10,0,0},{65,0,0},.2,.01,out,{}));
  check(out.allocation_scale<1);
  check(windup.update(f,f.received_ns,{-30,0,0},{45,0,0},.2,.01,out,{}));
  check(std::abs(out.pitch.i)<1e-12 && std::abs(out.yaw.i)<1e-12);
  bool aborted=false;
  for(int i=0;i<250;++i) {
    if(!windup.update(f,f.received_ns,{-10,0,0},{65,0,0},.2,.01,out,{})) {aborted=true;break;}
  }
  check(aborted);
  f.quaternion={std::cos(80*rad/2),0,std::sin(80*rad/2),0};
  check(!controller.read(f,f.received_ns,out));
  std::cout<<"coupled geometry, rates, allocation, stale feedback and singularity passed\n";
}
