/* The Thrust version of the Lorenz ensemble example.
   The state type is thrust::device_vector<double>.
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <utility>
#include <cstdlib>

#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include <boost/timer/timer.hpp>
#include <boost/typeof/typeof.hpp>

#ifdef USE_VANILLA_ODEINT
#  include <boost/numeric/odeint.hpp>
#  include <boost/numeric/odeint/external/vexcl/vexcl.hpp>
   namespace odeint = boost::numeric::odeint;
#else
#  include "thrust_operations.hpp"
#  include "thrust_algebra.hpp"
#  include "runge_kutta4.hpp"
   namespace odeint = ncwg;
#endif

#include "log.hpp"

//---------------------------------------------------------------------------
typedef thrust::device_vector<double> state_type;
struct lorenz_system {
    size_t n;
    double sigma, b;
    const state_type &R;

    lorenz_system(size_t n, const state_type &R,
            double sigma = 10.0, double b = 8.0 / 3.0
            ) : n(n), R(R), sigma(sigma), b(b) { }

    struct lorenz_functor {
        double sigma, b;
        lorenz_functor(double sigma, double b)
            : sigma(sigma), b(b) {}

        template<class T>
        __host__ __device__ void operator()(T t) const {
            double x = thrust::get<0>( t );
            double y = thrust::get<1>( t );
            double z = thrust::get<2>( t );
            double R = thrust::get<3>( t );

            thrust::get<4>( t ) = sigma * ( y - x );
            thrust::get<5>( t ) = R * x - y - x * z;
            thrust::get<6>( t ) = -b * z + x * y;
        }
    };

    template<class State, class Deriv>
    void operator()(const State &x, Deriv &dxdt, double t) const {
        BOOST_AUTO(start,
                thrust::make_zip_iterator( thrust::make_tuple(
                        x.begin(),
                        x.begin() + n,
                        x.begin() + 2 * n,
                        R.begin(),
                        dxdt.begin(),
                        dxdt.begin() + n,
                        dxdt.begin() + 2 * n
                        ) )
                );

        thrust::for_each(start, start+n, lorenz_functor(sigma, b));
    }
};

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    const size_t n = argc > 1 ? atoi(argv[1]) : 1024;
    const double dt = 0.01;
    const double t_max = 10.0;

    thrust::host_vector<double> r(n);
    const double Rmin = 0.1, Rmax = 50.0;
    for(size_t i = 0; i < n; ++i)
        r[i] = Rmin + i * (Rmax - Rmin) / (n - 1);

    state_type R = r;
    state_type x(3 * n);

    // initialize x,y,z
    thrust::fill(x.begin(), x.end(), 10.0);

    odeint::runge_kutta4_classic<
        state_type, double, state_type, double,
        odeint::thrust_algebra, odeint::thrust_operations
        > stepper;

    lorenz_system sys(n, R);

#ifndef CPU_RUN
    cudaThreadSynchronize();
#endif
    boost::timer::cpu_timer timer;

    for(double t = 0; t < t_max; t += dt)
        stepper.do_step(sys, x, t, dt);

#ifdef CPU_RUN
    log_perf("cpu_v1", n, t_max / dt, timer.elapsed());
#else
    cudaThreadSynchronize();
    log_perf("thrust_v1", n, t_max / dt, timer.elapsed());
#endif

#ifdef SHOW_OUTPUT
    thrust::host_vector<double> res = x;
    std::cout << "x = {" << std::setprecision(6);
    for(size_t i = 0; i < n; ++i) {
        if (i % 2 == 0) std::cout << "\n" << std::setw(6) << i << ":";
        std::cout << std::scientific << " (";
        for(size_t j = 0; j < 3; ++j)
            std::cout << std::setw(14) << res[j * 3 + i];
        std::cout << ")";
    }
    std::cout << "\n}" << std::endl;
#endif

    return 0;
}
