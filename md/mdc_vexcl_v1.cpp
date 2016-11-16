#include <vexcl/vexcl.hpp>

#include <boost/timer/timer.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/vexcl/vexcl.hpp>

#include "log.hpp"

//---------------------------------------------------------------------------
struct local_force {
    double gamma;

    local_force(double gamma = 0.0) : gamma(gamma) { }

    VEX_FUNCTION(cl_double2, impl, (cl_double, x)(cl_double2, v)(double, gamma),
            return -gamma * v;
            );

    template <class X, class V>
    auto operator()(X &&x, V &&v) const {
        return impl(x, v, gamma);
    }
};

//---------------------------------------------------------------------------
struct lennard_jones {
    double sigma;
    double eps;

    lennard_jones(double sigma = 1.0, double eps = 0.1)
        : sigma(sigma), eps(eps) { }

    VEX_FUNCTION(double, impl, (double, r)(double, sigma)(double, eps),
            double c = sigma / r, c3 = c * c * c, c6 = c3 * c3;
            return eps * c6 * (24.0 - 48.0 * c6) / r;
            );

    template <class R>
    auto operator()(R &&r) const {
        return impl(r, sigma, eps);
    }
};

//---------------------------------------------------------------------------
template< typename F >
struct conservative_interaction {
    F f;

    conservative_interaction( F const &f = F() ) : f( f ) { }

    template <class X1, class X2>
    auto operator()(X1 &&x1, X2 &&x2) const {
        VEX_CONSTANT(zero, 0);

        auto diff = vex::make_temp<1001>(x1 - x2);
        auto r    = vex::make_temp<1002, double>( length(diff) );
        return if_else( r == zero(), zero(), -diff / r * f(r) );
    }
};

//---------------------------------------------------------------------------
template<typename F>
conservative_interaction<F> make_conservative_interaction(F const &f) {
    return conservative_interaction<F>(f);
}

//---------------------------------------------------------------------------
template<typename local_force_type, typename interaction_type>
struct md_system {
    typedef cl_double2              point_type;
    typedef vex::vector<point_type> point_vector;
    // ...

    struct params {
        unsigned n;
        unsigned n_cell_x, n_cell_y, n_cells;
        double x_max, y_max, cell_size;
    } prm;

    struct periodic_bc_type {
        double xmax, ymax;

        periodic_bc_type(double xmax, double ymax) : xmax(xmax), ymax(ymax) {}

        VEX_FUNCTION(double, bc, (double,x)(double,y),
                double t = x - m * (int)(x / m);
                return t >= 0.0 ? t : t + m;
                );
        VEX_FUNCTION_D(cl_double2, impl, (cl_double2, p)(double, xmax)(double, ymax), (bc),
                return (double2)( bc(p.x, xmax), bc(p.y, ymax) );
                );

        template <class P>
        auto operator()(P &&p) const {
            return impl(p, xmax, ymax);
        }
    } periodic_bc;

    interaction_type interaction;
    local_force_type local_force;

    mutable point_vector x_bc;

    md_system(
            const vex::Context &ctx, size_t n,
            local_force_type const& local_force = local_force_type(),
            interaction_type const& interaction = interaction_type(),
            double xmax = 100.0, double ymax = 100.0, double cell_size = 2.0
            )
        : periodic_bc(xmax, ymax),
          interaction(interaction), local_force(local_force),
          x_bc(ctx, n)
    {
        prm.n = n;
        prm.x_max = xmax;
        prm.y_max = ymax;
        prm.n_cell_x = size_t( xmax / cell_size );
        prm.n_cell_y = size_t( ymax / cell_size );
        prm.n_cells = prm.n_cell_x * prm.n_cell_y;
        prm.cell_size = cell_size;
    }

    size_t num_points() const { return prm.n; }

    void bc(point_vector &x) const {
        x = periodic_bc(x);
    }

    void operator()(
            const point_vector &x, const point_vector &v,
            point_vector &a, double t) const
    {
        using namespace vex;

        x_bc = periodic_bc(x);

        auto x_i = make_temp<1>(x_bc);

        a = local_force(x_i, v)
          + reduce<SUM>(extents[1][prm.n], interaction(x_i, x_bc), 1);
    }
};

//---------------------------------------------------------------------------
template< typename LocalForce , typename Interaction >
md_system<LocalForce, Interaction> make_md_system(
        const vex::Context &ctx,
        size_t n, LocalForce const &f1, Interaction const &f2,
        double xmax = 100.0, double ymax = 100.0, double cell_size = 2.0
        )
{
    return md_system<LocalForce, Interaction>(ctx, n, f1, f2, xmax, ymax, cell_size);
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    using namespace boost::numeric::odeint;

    const size_t n1 = argc > 1 ? std::stoi(argv[1]) : 32;
    const size_t n2 = n1;
    const size_t n = n1 * n2;
    const size_t m = argc > 2 ? std::stoi(argv[2]) : 1000;

    const double cell_size = 2.0;
    const double domain_size = 50.0 * (n1 / 32.0);
    const double dx = 2.0 * domain_size / 50.0;
    const double mx = 10.0 * domain_size / 50.0;

    vex::Context ctx(
            vex::Filter::Env             &&
            vex::Filter::DoublePrecision &&
            vex::Filter::Count(1)
            );
    std::cout << ctx << std::endl;

    auto sys = make_md_system(
            ctx, n,
            local_force(),
            make_conservative_interaction( lennard_jones() ),
            domain_size, domain_size, cell_size
            );

    typedef decltype( sys ) system_type;
    typedef system_type::point_vector point_vector;

    point_vector x(ctx, sys.num_points());
    point_vector v(ctx, sys.num_points());

    {
        std::vector<system_type::point_type> X(sys.num_points());
        std::vector<system_type::point_type> V(sys.num_points());

        for( size_t i=0 ; i<n1 ; ++i ) {
            for( size_t j=0 ; j<n2 ; ++j ) {
                size_t index = i * n2 + j;
                X[index].s[0] = mx + i * dx;
                X[index].s[1] = mx + j * dx;
                V[index].s[0] = drand48();
                V[index].s[1] = drand48();
            }
        }

        vex::copy(X, x);
        vex::copy(V, v);
    }

    velocity_verlet< point_vector > stepper;
    const double dt = 0.025;
    double t = 0.0;

    ctx.finish();
    boost::timer::cpu_timer timer;

    for( size_t oi = 0; oi < m; ++oi ) {
        for( size_t ii = 0; ii < 5; ++ii, t+=dt )
            stepper.do_step( std::ref(sys), std::make_pair(std::ref(x), std::ref(v) ), t, dt );

#ifdef SHOW_OUTPUT
        sys.bc( x );
        std::cout << "x: " << x << std::endl;
        std::cout << "v: " << v << std::endl;
#else
        std::cout << "."  << std::flush;
        if ((oi + 1) % 50 == 0)
            std::cout << " " << oi << std::endl;
#endif
    }

    ctx.finish();
    log_perf("vexcl_v1", n, m * 5, timer.elapsed());
    std::cout << std::endl;
}
