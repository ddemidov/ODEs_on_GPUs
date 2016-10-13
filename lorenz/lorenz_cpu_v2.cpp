/* The improved CPU version of the Lorenz ensemble example.
 *
 * State type is std::array<double,3>, and the odeint stepper is called in an
 * outer loop which is iterating across attractors. This version is
 * cache-friendly, which result in tenfold acceleration w.r.t. the Thrust
 * version with OpenMP backend.
 */
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <utility>
#include <cstdlib>

#include <boost/timer/timer.hpp>

#ifdef USE_VANILLA_ODEINT
#  include <boost/numeric/odeint.hpp>
#  define container_algebra range_algebra
   namespace odeint = boost::numeric::odeint;
#else
#  include "default_operations.hpp"
#  include "container_algebra.hpp"
#  include "runge_kutta4.hpp"
   namespace odeint = ncwg;
#endif

#include "log.hpp"

typedef boost::array<double, 3> state_type;

//---------------------------------------------------------------------------
struct lorenz_system {
    double R, sigma, b;

    lorenz_system(double R, double sigma = 10.0, double b = 8.0 / 3.0)
        : R(R), sigma(sigma), b(b) { }

    void operator()(const state_type &x, state_type &dxdt, double t) const {
        dxdt[0] = sigma * ( x[1] - x[0] );
        dxdt[1] = R * x[0] - x[1] - x[0] * x[2];
        dxdt[2] = -b * x[2] + x[0] * x[1];
    }
};

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
const size_t n = argc > 1 ? atoi(argv[1]) : 1024;
const double dt = 0.01;
const double t_max = 10.0;

std::vector<double> R(n);
std::vector<state_type> x(n);

const double Rmin = 0.1, Rmax = 50.0;

#pragma omp parallel for
for(size_t i = 0; i < n; ++i) {
    R[i] = Rmin + i * (Rmax - Rmin) / (n - 1);
    x[i][0] = 10.0;
    x[i][1] = 10.0;
    x[i][2] = 10.0;
}

boost::timer::cpu_timer timer;

#pragma omp parallel for
for(size_t i = 0; i < n; ++i) {
    odeint::runge_kutta4_classic<
        state_type, double, state_type, double,
        odeint::container_algebra, odeint::default_operations
        > stepper;

    lorenz_system sys(R[i]);
    for(double t = 0; t < t_max; t += dt)
        stepper.do_step(sys, x[i], t, dt);
}

log_perf("cpu_v2", n, static_cast<size_t>(t_max / dt), timer.elapsed());

#ifdef SHOW_OUTPUT
std::cout << "x = {" << std::setprecision(6);
for(size_t i = 0; i < n; ++i) {
    if (i % 2 == 0) std::cout << "\n" << std::setw(6) << i << ":";
    std::cout << std::scientific << " (";
    for(size_t j = 0; j < 3; ++j)
        std::cout << std::setw(14) << x[i][j];
    std::cout << ")";
}
std::cout << "\n}" << std::endl;
#endif

return 0;
}
