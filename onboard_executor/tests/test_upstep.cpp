#include "fftgyro_executor/collective_ramp.hpp"
#include <cassert>
#include <cmath>
int main() {
  using fftgyro::transitionCollective;
  const auto near=[](double a,double b){ return std::abs(a-b)<1e-10; };
  assert(near(transitionCollective(.45,.55,.0004,true),.55));
  assert(near(transitionCollective(.55,.45,.0004,true),.5496));
  assert(near(transitionCollective(.45,.55,.0004,false),.4504));
  assert(near(transitionCollective(.65,.70,.0004,true),.70));
  double c=.70;
  for(int i=0;i<125;++i)c=transitionCollective(c,.65,.0004,true);
  assert(near(c,.65));
  assert(near(transitionCollective(.45,.45,.0004,true),.45));
}
