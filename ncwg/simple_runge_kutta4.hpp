#include <vector>

namespace ncwg {

class runge_kutta4 {
public:
  typedef std::vector<double> state_type;

  runge_kutta4(size_t N)
    : N(N), x_tmp(N), k1(N), k2(N), k3(N), k4(N) { }

  template<typename System>
  void do_step(System system, state_type &x, double t, double dt)
  {
    const double dt2 = dt / 2;
    const double dt3 = dt / 3;
    const double dt6 = dt / 6;

    system(x, k1, t);
    for(size_t i = 0; i < N; ++i)
      x_tmp[i] = x[i] + dt2 * k1[i];

    system(x_tmp, k2, t + dt2);
    for(size_t i = 0 ; i < N; ++i)
      x_tmp[i] = x[i] + dt2 * k2[i];

    system(x_tmp, k3, t + dt2);
    for(size_t i = 0; i < N; ++i)
      x_tmp[i] = x[i] + dt * k3[i];

    system(x_tmp, k4, t + dt);
    for(size_t i = 0; i < N; ++i)
      x[i] += dt6*k1[i] + dt3*k2[i] + dt3*k3[i] + dt6*k4[i];
  }

private:
  const size_t N;
  state_type x_tmp, k1, k2, k3, k4;
};

}
