#include <vexcl/vexcl.hpp>

#include <boost/timer/timer.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/vexcl/vexcl.hpp>

#include "log.hpp"

//---------------------------------------------------------------------------
struct local_force {
    double gamma;

    local_force(double gamma = 0.0) : gamma(gamma) { }

    void define(vex::backend::source_generator &src, const std::string &name) const
    {
        src <<
            "double2 " << name << "(double2 x, double2 v) {\n"
            "    return " << -gamma << " * v;\n"
            "}\n";
    }
};

//---------------------------------------------------------------------------
struct lennard_jones {
    double sigma;
    double eps;

    lennard_jones(double sigma = 1.0, double eps = 0.1)
        : sigma(sigma), eps(eps) { }

    void define(vex::backend::source_generator &src, const std::string &name) const
    {
        src <<
            "double " << name << "(double r) {\n"
            "    double c = " << sigma << " / r;\n"
            "    double c3 = c * c * c;\n"
            "    double c6 = c3 * c3;\n"
            "    return " << eps << " * c6 * (24.0 - 48.0 * c6) / r;\n"
            "}\n";
    }
};

//---------------------------------------------------------------------------
template< typename Func >
struct conservative_interaction {
    Func func;
    conservative_interaction( Func const &func = Func() )
        : func( func ) { }

    void define(vex::backend::source_generator &src, const std::string &name) const
    {
        std::string fname = name + "_helper";

        func.define(src, fname);

        src <<
            "double2 " << name << "(double2 x1, double2 x2) {\n"
            "    double2 d = x1 - x2;\n"
            "    double  r = length(d);\n"
            "    return r == 0 ? 0 : -d / r * " << fname << "(r);\n"
            "}\n";
    }
};

//---------------------------------------------------------------------------
template<typename F>
conservative_interaction<F> make_conservative_interaction(F const &f) {
    return conservative_interaction<F>(f);
}

template<class local_force_type, class interaction_type>
struct md_system_bs {
    // ...
    typedef cl_double2              point_type;
    typedef cl_int2                 index_type;
    typedef cl_uint                 hash_type;

    typedef vex::vector<double>     vector_type;
    typedef vex::vector<point_type> point_vector;
    typedef vex::vector<index_type> index_vector;
    typedef vex::vector<hash_type>  hash_vector;

    struct params {
        unsigned n;
        unsigned n_cell_x, n_cell_y, n_cells;
        double x_max, y_max, cell_size;
        interaction_type interaction;
        local_force_type local_force;
    } prm;

    VEX_FUNCTION(double, BC, (double, x)(double, m),
            double t = x - m * (int)(x / m);
            return t >= 0.0 ? t : t + m;
            );

    VEX_FUNCTION_D(point_type, periodic_bc, (point_type, p)(double, xmax)(double, ymax), (BC),
            return (double2)( BC(p.x, xmax), BC(p.y, ymax) );
            );

    VEX_FUNCTION(int, check, (int, k)(int, n),
            int tmp = k % n;
            return tmp >= 0 ? tmp : tmp + n;
            );
    VEX_FUNCTION_D(hash_type, get_cell_idx, (index_type, i)(cl_uint, nx)(cl_uint, ny), (check),
            int i1 = check(i.x, nx);
            int i2 = check(i.y, ny);
            return i1 * ny + i2;
            );

    VEX_FUNCTION(hash_type, lo_bound, (hash_type*, x)(size_t, n)(size_t, v),
            size_t begin = 0;
            size_t end   = n;
            while(end > begin) {
                size_t mid = begin + (end - begin) / 2;
                if (x[mid] < v)
                    begin = ++mid;
                else
                    end = mid;
            }
            return begin;
            );

    VEX_FUNCTION(hash_type, hi_bound, (hash_type*, x)(size_t, n)(size_t, v),
            size_t begin = 0;
            size_t end   = n;
            while(end > begin) {
                size_t mid = begin + (end - begin) / 2;
                if (x[mid] <= v)
                    begin = ++mid;
                else
                    end = mid;
            }
            return begin;
            );

    mutable index_vector cell_coo;
    mutable hash_vector  cell_idx;
    mutable hash_vector  part_ord;
    mutable hash_vector  cell_begin;
    mutable hash_vector  cell_end;
    mutable point_vector x_bc;

    md_system_bs(
            const vex::Context &ctx, size_t n,
            local_force_type const& local_force = local_force_type(),
            interaction_type const& interaction = interaction_type(),
            double xmax = 100.0, double ymax = 100.0, double cell_size = 2.0
            )
        : cell_coo(ctx, n), cell_idx(ctx, n), part_ord(ctx, n), x_bc(ctx, n)
    {
        prm.n = n;
        prm.x_max = xmax;
        prm.y_max = ymax;
        prm.interaction = interaction;
        prm.local_force = local_force;
        prm.n_cell_x = size_t( xmax / cell_size );
        prm.n_cell_y = size_t( ymax / cell_size );
        prm.n_cells = prm.n_cell_x * prm.n_cell_y;
        prm.cell_size = cell_size;

        cell_begin.resize(ctx, prm.n_cells);
        cell_end.resize(ctx, prm.n_cells);
    }

    size_t num_points() const { return prm.n; }

    void bc(point_vector &x) const {
        x = periodic_bc(x, prm.x_max, prm.y_max);
    }

    std::string interaction_kernel_source(const vex::command_queue &queue) const {
        vex::backend::source_generator src(queue);

        prm.local_force.define(src, "local_force");
        prm.interaction.define(src, "interaction");
        get_cell_idx.define   (src, "get_cell_idx");

        src << "\n" << VEX_STRINGIZE_SOURCE(
            kernel void global_interaction(
                uint n, uint nx, uint ny,
                double xmax, double ymax,
                global int2    const *cell_coo,
                global uint    const *part_ord,
                global uint    const *cell_begin,
                global uint    const *cell_end,
                global double2 const *X,
                global double2 const *V,
                global double2       *A
                )
            {
                for(size_t idx = get_global_id(0); idx < n; idx += get_global_size(0)) {
                    double2 x     = X[idx];
                    double2 a     = local_force(x, V[idx]);
                    int2    index = cell_coo[idx];
                    for(int i = -1; i <= 1; ++i) {
                        for(int j = -1; j <= 1; ++j) {
                            int2 cell_index = index + (int2)(i, j);
                            uint cell_hash  = get_cell_idx(cell_index, nx, ny);
                            for(uint ii = cell_begin[cell_hash], ee = cell_end[cell_hash]; ii < ee; ++ii) {
                                uint jj = part_ord[ii];
                                if (jj == idx) continue;
                                double2 y = X[jj];
                                if( cell_index.x >= nx ) y.x += xmax;
                                if( cell_index.x <  0  ) y.x -= xmax;
                                if( cell_index.y >= ny ) y.y += ymax;
                                if( cell_index.y <  0  ) y.y -= ymax;
                                a += interaction(x, y);
                            }
                        }
                    }
                    A[idx] = a;
                }
            }
        );

        return src.str();
    }

    vex::backend::kernel interaction_kernel(const vex::command_queue &queue, size_t n) const {
        static vex::backend::kernel krn(queue, interaction_kernel_source(queue), "global_interaction");
        size_t block = 256;
        krn.config((n + block - 1) / block, block);
        return krn;
    }

    void operator()(const point_vector &x, const point_vector &v,
            point_vector &a, double t) const
    {
        using namespace vex;
        auto ctx = x.queue_list();

        VEX_FUNCTION(cl_int2, make_idx, (cl_int, x)(cl_int, y), return (int2)(x,y););
        VEX_FUNCTION(double, get_x, (cl_double2,p), return p.x;);
        VEX_FUNCTION(double, get_y, (cl_double2,p), return p.y;);

        // Assign each particle to a cell, reset ordering.
        auto index = make_temp<1>(
                make_idx(get_x(x) / prm.cell_size, get_y(x) / prm.cell_size));
        vex::tie(cell_coo, cell_idx, part_ord) = std::tie(index,
                get_cell_idx(index, prm.n_cell_x, prm.n_cell_y), element_index());

        // Sort particle numbers in part_ord by cell numbers.
        sort_by_key(cell_idx, part_ord);

        // Find range of each cell in cell_idx array.
        vex::tie(cell_begin, cell_end) = std::tie(
                lo_bound(raw_pointer(cell_idx), prm.n, element_index(0, prm.n_cells)),
                hi_bound(raw_pointer(cell_idx), prm.n, element_index(0, prm.n_cells)));

        // Handle boundary conditions
        x_bc = periodic_bc(x, prm.x_max, prm.y_max);

        // Call custom interaction kernel.
        auto interaction = interaction_kernel(ctx[0], prm.n);
        interaction(ctx[0], prm.n, prm.n_cell_x, prm.n_cell_y, prm.x_max,
                prm.y_max, cell_coo(0), part_ord(0), cell_begin(0),
                cell_end(0), x_bc(0), v(0), a(0));
    }
};

template< typename LocalForce , typename Interaction >
md_system_bs<LocalForce, Interaction> make_md_system_bs(
        const vex::Context &ctx,
        size_t n, LocalForce const &f1, Interaction const &f2,
        double xmax = 100.0, double ymax = 100.0, double cell_size = 2.0
        )
{
    return md_system_bs<LocalForce, Interaction>(ctx, n, f1, f2, xmax, ymax, cell_size);
}

using namespace boost::numeric::odeint;

int main(int argc, char *argv[]) {
    const size_t n1 = argc > 1 ? std::stoi(argv[1]) : 32;
    const size_t n2 = n1;
    const size_t n = n1 * n2;
    const size_t m = argc > 2 ? std::stoi(argv[2]) : 1000;
    const double dt = 0.025;

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

    auto sys = make_md_system_bs(
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

    ctx.finish();
    boost::timer::cpu_timer timer;
    double t = 0.0;
    for(size_t oi = 0; oi < m; ++oi) {
        for(size_t ii = 0; ii < 5; ++ii, t+=dt)
            stepper.do_step( std::ref(sys), std::make_pair(std::ref(x), std::ref(v) ), t, dt);

#ifdef SHOW_OUTPUT
        sys.bc( x );
        std::cout << "set size square" << "\n";
        std::cout << "unset key" << "\n";
        std::cout << "p [0:" << sys.prm.x_max << "][0:" << sys.prm.y_max << "] '-' pt 7 ps 0.5" << "\n";
        std::cout << "x: " << x << std::endl;
        std::cout << "v: " << v << std::endl;
#else
        std::cout << "."  << std::flush;
        if ((oi + 1) % 50 == 0)
            std::cout << " " << oi << std::endl;
#endif
    }

    ctx.finish();
    log_perf("vexcl_v2", n, m * 5, timer.elapsed());
    std::cout << std::endl;
}
