#include <iostream>
#include <iomanip>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>

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

typedef thrust::device_vector< double > state_type;
struct phase_oscillators {
    const state_type &omega;
    const size_t n;

    phase_oscillators(const state_type &omega)
        : omega(omega), n(omega.size()) { }

    struct left_nbr : thrust::unary_function<size_t, size_t> {
        __host__ __device__ size_t operator()(size_t i) const {
            return (i > 0) ? i - 1 : 0;
        }
    };

    struct right_nbr : thrust::unary_function<size_t, size_t> {
        size_t back;
        right_nbr(size_t back) : back(back) {}
        __host__ __device__ size_t operator()(size_t i) const {
            return (i >= back) ? back : i + 1;
        }
    };

    struct sys_functor {
        template< class Tuple >
        __host__ __device__ void operator()( Tuple t ) {
            double phi_c = thrust::get<0>(t);
            double phi_l = thrust::get<1>(t);
            double phi_r = thrust::get<2>(t);
            double omega = thrust::get<3>(t);

            thrust::get<4>(t) = omega +
                sin(phi_r - phi_c) + sin(phi_c - phi_l);
        }
    };

    void operator() (const state_type &x, state_type &dxdt,
            double dt)
    {
        BOOST_AUTO(start, thrust::make_zip_iterator(
                    thrust::make_tuple(
                        x.begin(),
                        thrust::make_permutation_iterator(
                            x.begin(),
                            thrust::make_transform_iterator(
                                thrust::counting_iterator<size_t>(0),
                                left_nbr()
                                )
                            ),
                        thrust::make_permutation_iterator(
                            x.begin(),
                            thrust::make_transform_iterator(
                                thrust::counting_iterator<size_t>(0),
                                right_nbr(n - 1)
                                )
                            ),
                        omega.begin(),
                        dxdt.begin()
                        )
                    )
                );

        thrust::for_each(start, start + n, sys_functor());
    }
};

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    const size_t n = argc > 1 ? atoi(argv[1]) : 1024;

    const double dt = 0.01;
    const double t_max = 10.0;
    const double epsilon = 6.0 / ( n * n ); // should be < 8/N^2 to see phase locking

    std::vector<double> x_host( n );
    std::vector<double> omega_host( n );

    for(size_t i = 0; i < n; ++i) {
        x_host[i] = 2.0 * M_PI * drand48();
        omega_host[i] = (n - i) * epsilon; // decreasing frequencies
    }

    state_type x = x_host;
    state_type omega = omega_host;

    odeint::runge_kutta4_classic<
        state_type, double, state_type, double,
        odeint::thrust_algebra, odeint::thrust_operations
        > stepper;

    phase_oscillators sys(omega);

#ifndef CPU_RUN
    cudaThreadSynchronize();
#endif
    boost::timer::cpu_timer timer;

    for(double t = 0; t < t_max; t += dt)
        stepper.do_step(sys, x, t, dt);

#ifdef CPU_RUN
    log_perf("cpu", n, t_max / dt, timer.elapsed());
#else
    cudaThreadSynchronize();
    log_perf("thrust", n, t_max / dt, timer.elapsed());
#endif

#ifdef SHOW_OUTPUT
    thrust::host_vector<double> res = x;
    std::cout << "x = {" << std::setprecision(6);
    for(size_t i = 0; i < n; ++i) {
        if (i % 6 == 0) std::cout << "\n" << std::setw(6) << i << ":";
        std::cout << std::scientific << std::setw(14) << res[i];
    }
    std::cout << "\n}" << std::endl;
#endif
}
