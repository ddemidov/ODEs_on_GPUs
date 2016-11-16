#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <cmath>

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

typedef vex::vector<double> state_type;
struct phase_oscillators {
    const state_type &omega;

    phase_oscillators(const state_type &omega) : omega(omega) {}

    void operator()(const state_type &phi, state_type &dxdt,
            double t) const
    {
        VEX_FUNCTION(size_t, left,  (size_t, i),
                return (i > 0) ? i - 1 : 0;
                );
        VEX_FUNCTION(size_t, right, (size_t, i)(size_t, n),
                return (i >= n) ? n : i + 1;
                );

        auto idx = vex::element_index();
        auto phi_l=vex::permutation(left(idx))(phi);
        auto phi_r=vex::permutation(right(idx,phi.size()-1))(phi);

        dxdt = omega + sin(phi_r - phi) + sin(phi - phi_l);
    }
};

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    const size_t n = ( argc > 1 ) ? atoi(argv[1]) : 1024;
    const double dt = 0.01;
    const double t_max = 10.0;
    const double epsilon = 6.0 / ( n * n ); // should be < 8/N^2 to see phase locking

    vex::Context ctx( vex::Filter::Env && vex::Filter::DoublePrecision );
    std::cout << ctx << std::endl;

    // initialize omega and the state of the lattice

    std::vector< double > omega( n );
    std::vector< double > x( n );
    for(size_t i = 0; i < n; ++i) {
        x[i] = 2.0 * M_PI * drand48();
        omega[i] = (n - i) * epsilon; // decreasing frequencies
    }

    state_type X(ctx, x);
    state_type Omega(ctx, omega);

    odeint::runge_kutta4_classic<
        state_type, double, state_type, double,
        odeint::vector_space_algebra, odeint::default_operations
        > stepper;

    ctx.finish();
    boost::timer::cpu_timer timer;

    phase_oscillators sys(Omega);
    for(double t = 0; t < t_max; t += dt)
        stepper.do_step(sys, X, t, dt);

    ctx.finish();
    log_perf("vexcl", n, t_max / dt, timer.elapsed());

#ifdef SHOW_OUTPUT
    std::cout << "x = " << X << std::endl;
#endif
}
