/* Molecular dynamics example
 * CPU implementation
 * Bucket sort version
 */

#include <boost/numeric/odeint.hpp>

#include <type_traits>
#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <iostream>

#include <boost/timer/timer.hpp>
#include <boost/io/ios_state.hpp>

#include "point_type.hpp"
#include "log.hpp"

//---------------------------------------------------------------------------
template <class T, size_t N>
std::ostream& operator<<(std::ostream &o, const point<T, N> &p) {
    boost::io::ios_all_saver stream_state(o);
    o << "(";
    for(int i = 0; i < 2; ++i) {
        if (i)                          o << ", ";
        if (std::is_integral<T>::value) o << std::setw(6);
        else                            o << std::scientific;
        o << p[0];
    }
    return o << ")";
}

//---------------------------------------------------------------------------
template<class T>
std::ostream &operator<<(std::ostream &o, const std::vector<T> &data) {
    boost::io::ios_all_saver stream_state(o);
    const size_t chunk = std::is_integral<T>::value ? 10 : 5;

    o << "{" << std::setprecision(6);
    for(size_t i = 0 ; i < data.size() ; i++) {
        if (i % chunk == 0) {
             o << "\n" << std::setw(6) << i << ":";
        }
        if (std::is_integral<T>::value)
            o << " " << std::setw(6) << data[i];
        else
            o << std::scientific << " " << data[i];
    }
    return o << "\n}\n";
}


//---------------------------------------------------------------------------
struct local_force {
    double gamma; // friction

    local_force(double gamma = 0.0)
        : gamma(gamma) { }

    template<class Point>
    Point operator()(const Point &x, const Point &v) const {
        return -gamma * v;
    }
};

//---------------------------------------------------------------------------
struct lennard_jones {
    double sigma;
    double eps;

    lennard_jones(double sigma = 1.0, double eps = 0.1)
        : sigma(sigma), eps(eps) { }

    double operator()(double r) const {
        double c = sigma / r;
        double c3 = c * c * c;
        double c6 = c3 * c3;
        return eps * c6 * (24.0 - 12.0 * c6) / r;
    }
};

//---------------------------------------------------------------------------
template<class F>
struct conservative_interaction {
    F f;

    conservative_interaction(F f = F())
        : f(f) { }

    template<class Point>
    Point operator()(const Point &x, const Point &y) const {
        Point  d = x - y;
        double r = abs(d);
        return r == 0 ? 0 : -d / r * f(r);
    }
};

//---------------------------------------------------------------------------
template<class F>
conservative_interaction<F> make_conservative_interaction(F f) {
    return conservative_interaction<F>(f);
}

//---------------------------------------------------------------------------
// force = interaction(x1, x2)
// force = local_force(x, v)
template<typename LocalForce, typename Interaction>
class md_system_bs {
    public:
        typedef point<double, 2>        point_type;
        typedef std::vector<point_type> point_vector;
        typedef LocalForce              local_force_type;
        typedef Interaction             interaction_type;

        struct params {
            size_t n;
            size_t n_cell_x, n_cell_y, n_cells;
            double x_max, y_max, cell_size;
            interaction_type interaction;
            local_force_type local_force;
        } prm;

        mutable point_vector x_bc;

        md_system_bs(
                size_t n,
                local_force_type local_force = local_force_type(),
                interaction_type interaction = interaction_type(),
                double xmax = 100.0, double ymax = 100.0,
                double cell_size = 2.0
                ) : x_bc(n)
        {
            prm.n = n;

            prm.x_max = xmax;
            prm.y_max = ymax;

            prm.interaction = interaction;
            prm.local_force = local_force;

            prm.n_cell_x = static_cast<size_t>(xmax / cell_size);
            prm.n_cell_y = static_cast<size_t>(ymax / cell_size);
            prm.n_cells  = prm.n_cell_x * prm.n_cell_y;

            prm.cell_size = cell_size;
        }

        size_t num_points() const { return prm.n; }

        void operator()(
                point_vector const &x,
                point_vector const &v,
                point_vector       &a,
                double t
                ) const
        {
#pragma omp parallel for schedule(dynamic, 512)
            for(size_t i = 0; i < prm.n; ++i)
                x_bc[i] = periodic_bc(x[i]);

#pragma omp parallel for schedule(dynamic, 512)
            for(size_t i = 0; i < prm.n; ++i) {
                point_type X = x_bc[i];
                point_type A = prm.local_force(X, v[i]);

                for(size_t j = 0; j < prm.n; ++j)
                    if (j != i) A += prm.interaction(X, x_bc[j]);

                a[i] = A;
            }
        }

        void bc(point_vector &x) const {
#pragma omp parallel for schedule(dynamic, 512)
            for(size_t i = 0; i < prm.n; ++i)
                x[i] = periodic_bc(x[i]);
        }

        static inline double periodic_bc(double x, double xmax) {
            double tmp = x - xmax * static_cast<int>(x / xmax);
            return tmp >= 0.0 ? tmp : tmp + xmax;
        }


        inline point_type periodic_bc(const point_type &x) const {
            return point_type(
                    periodic_bc(x[0], prm.x_max),
                    periodic_bc(x[1], prm.y_max)
                    );
        }
};

template<class LocalForce, class Interaction>
md_system_bs<LocalForce, Interaction> make_md_system_bs(
        size_t n, LocalForce f, Interaction i,
        double xmax = 100.0, double ymax = 100.0,
        double cell_size = 2.0
        )
{
    return md_system_bs<LocalForce, Interaction>(n, f, i, xmax, ymax, cell_size);
}


int main( int argc , char *argv[] ) {
    using namespace boost::numeric::odeint;

    const size_t n1 = argc > 1 ? std::stoi(argv[1]) : 32;
    const size_t m  = argc > 2 ? std::stoi(argv[2]) : 1000;

    const size_t n2 = n1;
    const size_t n  = n1 * n2;

    const double cell_size = 2.0;
    const double domain_size = 50.0 * (n1 / 32.0);
    const double dx = 2.0 * domain_size / 50.0;
    const double mx = 10.0 * domain_size / 50.0;

    auto sys = make_md_system_bs(
            n,
            local_force(),
            make_conservative_interaction( lennard_jones() ),
            domain_size, domain_size, cell_size
            );

    typedef decltype( sys ) system_type;
    typedef system_type::point_vector point_vector;

    point_vector x(sys.num_points());
    point_vector v(sys.num_points());

    for( size_t i=0 ; i<n1 ; ++i ) {
        for( size_t j=0 ; j<n2 ; ++j ) {
            size_t index = i * n2 + j;
            x[index][0] = mx + i * dx;
            x[index][1] = mx + j * dx;
            v[index][0] = drand48();
            v[index][1] = drand48();
        }
    }

    velocity_verlet< point_vector > stepper;
    const double dt = 0.025;
    double t = 0.0;

    // std::cout << "set term x11" << endl;
    boost::timer::cpu_timer timer;
    for( size_t oi = 0 ; oi < m ; ++oi ) {
        for( size_t ii = 0 ; ii < 5 ; ++ii, t+=dt )
            stepper.do_step( sys, std::make_pair(std::ref(x), std::ref(v) ), t, dt );

#ifdef SHOW_OUTPUT
        sys.bc( x );
        std::cout << "set size square" << "\n";
        std::cout << "unset key" << "\n";
        std::cout << "p [0:" << sys.prm.x_max << "][0:" << sys.prm.y_max << "] '-' pt 7 ps 0.5" << "\n";
        for( size_t i=0 ; i<n ; ++i )
            std::cout << x[i][0] << " " << x[i][1] << " " << v[i][0] << " " << v[i][1] << "\n";
        std::cout << "e" << std::endl;
#else
        std::cout << "."  << std::flush;
        if ((oi + 1) % 50 == 0)
            std::cout << " " << oi << std::endl;
#endif
    }

    log_perf("cpu_v1", n, m * 5, timer.elapsed());
    std::cout << std::endl;
}
