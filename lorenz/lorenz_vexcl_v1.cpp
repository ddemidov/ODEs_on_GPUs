/* VexCL version 1 of the Lorenz ensemble example.
 *
 * This version uses vex::vector<double> as state type. The performance is
 * similar to the Thrust version.
 */
#include <iostream>
#include <vector>
#include <tuple>

#include <vexcl/vexcl.hpp>

#include <boost/timer/timer.hpp>

#ifdef USE_VANILLA_ODEINT
#  include <boost/numeric/odeint.hpp>
#  include <boost/numeric/odeint/external/vexcl/vexcl.hpp>
   namespace odeint = boost::numeric::odeint;
#else
#  include "default_operations.hpp"
#  include "vector_space_algebra.hpp"
#  include "vexcl_resize.hpp"
#  include "runge_kutta4.hpp"
   namespace odeint = ncwg;
#endif

#include "log.hpp"

typedef vex::multivector<double, 3> state_type;
struct lorenz_system {
    const vex::vector<double> &R;
    double sigma, b;

    lorenz_system(const vex::vector<double> &R, double sigma = 10.0, double b = 8.0/3)
        : R(R), sigma(sigma), b(b) { }

    void operator()(const state_type &x, state_type &dxdt,
                        double t) const
    {
	dxdt = std::tie(
		sigma * (x(1) - x(0)),
		R * x(0) - x(1) - x(0) * x(2),
		x(0) * x(1) - b * x(2)         );
    }
};

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    const size_t n = argc > 1 ? atoi(argv[1]) : 1024;
    const double dt = 0.01;
    const double t_max = 10.0;

    vex::Context ctx( vex::Filter::Env && vex::Filter::DoublePrecision );
    std::cout << ctx << std::endl;

    double Rmin = 0.1, Rmax = 50.0, dR = (Rmax - Rmin) / (n - 1);

    vex::vector<double> R(ctx, n);
    R = Rmin + dR * vex::element_index();

    state_type X(ctx, n);
    X = 10.0;

    odeint::runge_kutta4_classic<
        state_type, double, state_type, double,
        odeint::vector_space_algebra, odeint::default_operations
        > stepper;

    ctx.finish();
    boost::timer::cpu_timer timer;

    lorenz_system sys(R);
    for(double t = 0; t < t_max; t += dt)
        stepper.do_step(sys, X, t, dt);

    ctx.finish();
    log_perf("vexcl_v1", n, t_max / dt, timer.elapsed());

#ifdef SHOW_OUTPUT
    std::cout << "x = " << X << std::endl;
#endif
}
