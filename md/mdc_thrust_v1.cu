#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>

#include <boost/timer/timer.hpp>
#include <boost/typeof/typeof.hpp>
#include <boost/ref.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>

#include "log.hpp"
#include "point_type.hpp"



//---------------------------------------------------------------------------
struct local_force {
    double gamma;        // friction
    local_force( double gamma = 0.0 ) : gamma( gamma ) { }

    template< typename Point >
    __host__ __device__ Point operator()(Point x, Point v) const {
        return -gamma * v;
    }
};

//---------------------------------------------------------------------------
struct lennard_jones {
    double sigma;
    double eps;

    lennard_jones( double sigma = 1.0 , double eps = 0.1 )
        : sigma( sigma ), eps( eps ) { }

    __host__ __device__ double operator()( double r ) const {
        double c = sigma / r;
        double c3 = c * c * c;
        double c6 = c3 * c3;
        return eps * c6 * ( 24.0 - 48.0 * c6) / r;
    }
};

//---------------------------------------------------------------------------
template< typename F >
struct conservative_interaction {
    F f;

    conservative_interaction( F f = F() ) : f( f ) { }

    template< typename Point >
    __host__ __device__ Point operator()( Point const& x1 , Point const& x2 ) const {
        Point diff = x1 - x2;
        double r = abs( diff );
        return r == 0 ? 0 : -diff / r * f(r);
    }
};

//---------------------------------------------------------------------------
template< typename F >
conservative_interaction< F > make_conservative_interaction( F const &f ) {
    return conservative_interaction< F >( f );
}

//---------------------------------------------------------------------------
template<class local_force_type, class interaction_type>
struct md_system {
    typedef point<double, 2>                  point_type;
    typedef thrust::device_vector<point_type> point_vector;

    struct params {
        size_t n;
        double x_max, y_max;
        interaction_type interaction;
        local_force_type local_force;
    } prm;
    // ...

    mutable point_vector x_bc;

    md_system(
            size_t n ,
            local_force_type const& local_force = local_force_type() ,
            interaction_type const& interaction = interaction_type() ,
            double xmax = 100.0 , double ymax = 100.0
            ) : x_bc(n)
    {
        prm.n = n;
        prm.x_max = xmax;
        prm.y_max = ymax;
        prm.interaction = interaction;
        prm.local_force = local_force;
    }

    size_t num_points() const { return prm.n; }

    struct interaction_functor {
        // ...
        local_force_type local_force;
        interaction_type interaction;

        size_t n;
        point_type const *x;
        point_type const *v;


        interaction_functor(
                point_vector const &x, point_vector const &v, params const &p
                ) : local_force(p.local_force), interaction(p.interaction), n(p.n),
                    x( thrust::raw_pointer_cast(&x[0]) ),
                    v( thrust::raw_pointer_cast(&v[0]) )
        { }

        template< typename Tuple >
        __host__ __device__ void operator()( Tuple const &t ) const {
            point_type X = thrust::get<0>(t);
            point_type V = thrust::get<1>(t);
            size_t i     = thrust::get<3>(t);

            point_type A = local_force(X, V);

            for(size_t j = 0; j < n; ++j)
                if (j != i) A += interaction(X, x[j]);

            thrust::get<2>(t) = A;
        }
    };

    void operator()(point_vector const &x, point_vector const &v,
            point_vector &a, double t) const
    {
        thrust::transform(x.begin(), x.end(), x_bc.begin(), bc_functor(prm));

        BOOST_AUTO(start, thrust::make_zip_iterator(
                    thrust::make_tuple(
                        x_bc.begin(),
                        v.begin(),
                        a.begin(),
                        thrust::counting_iterator<size_t>(0)
                        )
                    ) );

        thrust::for_each(start, start + prm.n, interaction_functor(x_bc, v, prm));
    }
    // ...

    struct bc_functor {
        size_t xmax, ymax;

        bc_functor(const params &p) : xmax(p.x_max), ymax(p.y_max) {}

        __host__ __device__ point_type operator()(point_type p) const {
            return periodic_bc(p, xmax, ymax);
        }
    };

    void bc( point_vector &x )
    {
        thrust::transform(x.begin(), x.end(), x.begin(), bc_functor(prm));
    }

    __host__ __device__ static inline double periodic_bc( double x , double xmax )
    {
        double tmp = x - xmax * int( x / xmax );
        return tmp >= 0.0 ? tmp : tmp + xmax;
    }


    __host__ __device__ static inline point_type periodic_bc( point_type x, double xmax, double ymax)
    {
        return point_type( periodic_bc( x[0] , xmax ) , periodic_bc( x[1] , ymax ) );
    }
};


template< typename LocalForce , typename Interaction >
md_system< LocalForce , Interaction > make_md_system(
        size_t n , LocalForce const &f1 , Interaction const &f2 ,
        double xmax = 100.0 , double ymax = 100.0
        )
{
    return md_system< LocalForce , Interaction >( n , f1 , f2 , xmax , ymax );
}


//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    using namespace boost::numeric::odeint;

    const size_t n1 = argc > 1 ? atoi(argv[1]) : 32;
    const size_t n2 = n1;
    const size_t n = n1 * n2;
    const size_t m = argc > 2 ? atoi(argv[2]) : 1000;

    const double cell_size = 2.0;
    const double domain_size = 50.0 * (n1 / 32.0);
    const double dx = 2.0 * domain_size / 50.0;
    const double mx = 10.0 * domain_size / 50.0;

    typedef
        md_system<local_force, conservative_interaction<lennard_jones> >
        system_type;

    typedef
        system_type::point_vector
        point_vector;

    typedef
        system_type::point_type
        point_type;

    system_type sys = system_type(n,
        local_force(), make_conservative_interaction( lennard_jones() ),
        domain_size, domain_size);


    point_vector x(n);
    point_vector v(n);

    {
        std::vector<point_type> X(n);
        std::vector<point_type> V(n);

        for( size_t i=0 ; i<n1 ; ++i ) {
            for( size_t j=0 ; j<n2 ; ++j ) {
                size_t index = i * n2 + j;
                X[index][0] = mx + i * dx;
                X[index][1] = mx + j * dx;
                V[index][0] = drand48();
                V[index][1] = drand48();
            }
        }

        thrust::copy(X.begin(), X.end(), x.begin());
        thrust::copy(V.begin(), V.end(), v.begin());
    }

    velocity_verlet< point_vector > stepper;
    const double dt = 0.025;
    double t = 0.0;

    // std::cout << "set term x11" << endl;
    cudaThreadSynchronize();
    boost::timer::cpu_timer timer;
    for(size_t oi = 0; oi < m; ++oi) {
      for( size_t ii = 0 ; ii < 5 ; ++ii, t+=dt )
        stepper.do_step(sys, std::make_pair(boost::ref(x), boost::ref(v)), t, dt);

#ifdef SHOW_OUTPUT
        sys.bc( x );
        std::cout << "set size square" << "\n";
        std::cout << "unset key" << "\n";
        std::cout << "p [0:" << sys.prm.x_max << "][0:" << sys.prm.y_max << "] '-' pt 7 ps 0.5" << "\n";
        for( size_t i=0 ; i<n ; ++i ) {
            point_type X = x[i];
            point_type V = v[i];
            std::cout << X[0] << " " << X[1] << " " << V[0] << " " << V[1] << "\n";
        }
        std::cout << "e" << std::endl;
#else
        std::cout << "."  << std::flush;
        if ((oi + 1) % 50 == 0)
            std::cout << " " << oi << std::endl;
#endif
    }

    cudaThreadSynchronize();
    log_perf("thrust_v1", n, m * 5, timer.elapsed());
    std::cout << std::endl;
}
