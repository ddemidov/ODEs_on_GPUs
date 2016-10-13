#include <iostream>
#include <vector>
#include <tuple>

#include <vexcl/vexcl.hpp>

#include <boost/timer/timer.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/vexcl/vexcl.hpp>

#include "log.hpp"

namespace odeint = boost::numeric::odeint;

typedef vex::vector<double>         vector_type;
typedef vex::multivector<double, 3> state_type;

struct lorenz_system {
    const vector_type &R;
    double sigma, b;

    lorenz_system(const vector_type &R, double sigma = 10.0, double b = 8.0/3)
        : R(R), sigma(sigma), b(b) { }

    auto operator()(const state_type &x) const {
        return std::make_tuple(
                sigma * (x(1) - x(0)),
                R * x(0) - x(1) - x(0) * x(2),
                x(0) * x(1) - b * x(2)
                );
    }
};

int main(int argc, char *argv[]) {
    const size_t n = argc > 1 ? atoi(argv[1]) : 1024;
    const double dt = 0.01;
    const double t_max = 10.0;

    vex::Context ctx( vex::Filter::Env && vex::Filter::DoublePrecision );
    std::cout << ctx << std::endl;

    double Rmin = 0.1, Rmax = 50.0, dR = (Rmax - Rmin) / (n - 1);

    vector_type R(ctx, n);
    R = Rmin + dR * vex::element_index();

    state_type X(ctx, n);
    X = 10.0;

    odeint::runge_kutta4_classic< state_type > stepper;

    lorenz_system rhs(R);

    ctx.finish();
    boost::timer::cpu_timer timer;

    double dt2 = dt / 2;
    double dt3 = dt / 3;
    double dt6 = dt / 6;

    state_type k1(ctx, n);
    state_type k2(ctx, n);
    state_type k3(ctx, n);

    state_type xtmp(ctx, n);

    for(double t = 0; t < t_max; t += dt) {
        using namespace vex;

        {
            auto tmp = rhs(X);
            auto sys = std::make_tuple(
                    make_temp<0>( std::get<0>(tmp) ),
                    make_temp<1>( std::get<1>(tmp) ),
                    make_temp<2>( std::get<2>(tmp) )
                    );

            vex::tie(
                    k1(0),
                    k1(1),
                    k1(2),
                    xtmp(0),
                    xtmp(1),
                    xtmp(2)
                    ) =
                std::tie(
                    std::get<0>(sys),
                    std::get<1>(sys),
                    std::get<2>(sys),
                    X(0) + dt2 * std::get<0>(sys),
                    X(1) + dt2 * std::get<1>(sys),
                    X(2) + dt2 * std::get<2>(sys)
                    );
        }

        {
            auto tmp = rhs(xtmp);
            auto sys = std::make_tuple(
                    make_temp<0>( std::get<0>(tmp) ),
                    make_temp<1>( std::get<1>(tmp) ),
                    make_temp<2>( std::get<2>(tmp) )
                    );

            vex::tie(
                    k2(0),
                    k2(1),
                    k2(2),
                    xtmp(0),
                    xtmp(1),
                    xtmp(2)
                    ) =
                std::tie(
                    std::get<0>(sys),
                    std::get<1>(sys),
                    std::get<2>(sys),
                    X(0) + dt2 * std::get<0>(sys),
                    X(1) + dt2 * std::get<1>(sys),
                    X(2) + dt2 * std::get<2>(sys)
                    );
        }

        {
            auto tmp = rhs(xtmp);
            auto sys = std::make_tuple(
                    make_temp<0>( std::get<0>(tmp) ),
                    make_temp<1>( std::get<1>(tmp) ),
                    make_temp<2>( std::get<2>(tmp) )
                    );

            vex::tie(
                    k3(0),
                    k3(1),
                    k3(2),
                    xtmp(0),
                    xtmp(1),
                    xtmp(2)
                    ) =
                std::tie(
                    std::get<0>(sys),
                    std::get<1>(sys),
                    std::get<2>(sys),
                    X(0) + dt2 * std::get<0>(sys),
                    X(1) + dt2 * std::get<1>(sys),
                    X(2) + dt2 * std::get<2>(sys)
                    );
        }

        {
            auto sys = rhs(xtmp);
            X += std::tie(
                    dt6 * k1(0) + dt3 * k2(0) + dt3 * k3(0) + dt6 * std::get<0>(sys),
                    dt6 * k1(1) + dt3 * k2(1) + dt3 * k3(1) + dt6 * std::get<1>(sys),
                    dt6 * k1(2) + dt3 * k2(2) + dt3 * k3(2) + dt6 * std::get<2>(sys)
                    );
        }
    }

    ctx.finish();
    log_perf("vexcl_v3", n, t_max / dt, timer.elapsed());

#ifdef SHOW_OUTPUT
    std::cout << "x = " << X << std::endl;
#endif
}
