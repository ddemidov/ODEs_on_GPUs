#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>

#include <boost/timer/timer.hpp>
#include <boost/ref.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>

#include "log.hpp"
#include "point_type.hpp"



struct local_force {
  double m_gamma;        // friction
  local_force( double gamma = 0.0 ) : m_gamma( gamma ) { }

  template< typename Point >
  __host__ __device__ Point operator()(Point x, Point v) const {
    return - m_gamma * v;
  }
};


struct lennard_jones {
  double m_sigma;
  double m_eps;

  lennard_jones( double sigma = 1.0 , double eps = 0.1 ) : m_sigma( sigma ) , m_eps( eps ) { }

  __host__ __device__ double operator()( double r ) const {
    double c = m_sigma / r;
    double c3 = c * c * c;
    double c6 = c3 * c3;
    return 4.0 * m_eps * ( -12.0 * c6 * c6 / r + 6.0 * c6 / r );
  }
};

template< typename F >
struct conservative_interaction {
  F f;

  conservative_interaction( F const &f = F() ) : f(f) { }

  template< typename Point >
  __host__ __device__ Point operator()( Point const& x , Point const& y ) const {
    Point d = x - y;
    double r = abs(d);
    return r == 0 ? 0 : -d / r * f(r);
  }
};

template< typename F >
conservative_interaction< F > make_conservative_interaction( F const &f ) {
  return conservative_interaction< F >( f );
}


// force = interaction( x1 , x2 )
// force = local_force( x , v )
template< typename LocalForce , typename Interaction >
struct md_system_bs {
  typedef point<double, 2>                  point_type;
  typedef point<int, 2>                     index_type;
  typedef thrust::device_vector<point_type> point_vector;
  typedef thrust::device_vector<index_type> index_vector;
  typedef thrust::device_vector<size_t>     hash_vector;
  typedef LocalForce                        local_force_type;
  typedef Interaction                       interaction_type;


  struct params {
    size_t n;
    size_t n_cell_x , n_cell_y , n_cells;
    double x_max , y_max , cell_size;
    double eps , sigma;    // interaction strength, interaction radius
    interaction_type interaction;
    local_force_type local_force;
  } prm;

  mutable index_vector cell_coo;
  mutable hash_vector  cell_idx;
  mutable hash_vector  part_ord;
  mutable hash_vector  cells_begin;
  mutable hash_vector  cells_end;
  mutable point_vector x_bc;

  struct fill_index_n_hash {
    size_t nx, ny;
    double cell_size;

    fill_index_n_hash(const params &p)
      : nx(p.n_cell_x), ny(p.n_cell_y), cell_size(p.cell_size) { }

    template< typename Tuple >
    __host__ __device__ void operator()( Tuple const& t ) const
    {
      // Input: x, index, hash
      point_type point = thrust::get< 0 >( t );
      size_t i1 = size_t( point[0] / cell_size );
      size_t i2 = size_t( point[1] / cell_size );
      thrust::get< 1 >( t ) = index_type( i1 , i2 );
      thrust::get< 2 >( t ) = get_cell_idx( thrust::get< 1 >( t ) , nx, ny );
    }
  };

  md_system_bs(size_t n ,
      local_force_type const& local_force = local_force_type() ,
      interaction_type const& interaction = interaction_type() ,
      double xmax = 100.0 , double ymax = 100.0 , double cell_size = 2.0 )
    : cell_coo(n), cell_idx(n), part_ord(n), x_bc(n)
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

    cells_begin.resize(prm.n_cells);
    cells_end.resize(prm.n_cells);
  }

  size_t num_points() const { return prm.n; }

  struct interaction_functor {
    size_t nx, ny;
    double xmax, ymax;
    local_force_type local_force;
    interaction_type interaction;

    size_t     const *cells_begin;
    size_t     const *cells_end;
    size_t     const *order;
    point_type const *x;
    point_type const *v;


    interaction_functor(
        hash_vector  const &cells_begin,
        hash_vector  const &cells_end,
        hash_vector  const &part_ord,
        point_vector const &x,
        point_vector const &v,
        params       const &p
        )
      : nx( p.n_cell_x ), ny(p.n_cell_y), xmax(p.x_max), ymax(p.y_max),
        local_force(p.local_force), interaction(p.interaction),
        cells_begin( thrust::raw_pointer_cast(&cells_begin[0]) ),
        cells_end  ( thrust::raw_pointer_cast(&cells_end[0]) ),
        order      ( thrust::raw_pointer_cast(&part_ord[0]) ),
        x          ( thrust::raw_pointer_cast(&x[0]) ),
        v          ( thrust::raw_pointer_cast(&v[0]) )
    { }

    template< typename Tuple >
    __host__ __device__ void operator()( Tuple const &t ) const {
      point_type X     = thrust::get<0>( t );
      point_type V     = thrust::get<1>( t );
      index_type index = thrust::get<2>( t );
      size_t cell_idx   = thrust::get<3>( t );

      point_type A     = local_force(X, V);

      for(int i = -1; i <= 1; ++i) {
        for(int j = -1; j <= 1; ++j) {
          index_type cell_index = index + index_type(i, j);
          size_t cell_hash = get_cell_idx(cell_index, nx, ny);
          for(size_t ii = cells_begin[cell_hash],
                     ee = cells_end[cell_hash]; ii < ee; ++ii)
          {
            size_t jj = order[ii];

            if( jj == cell_idx ) continue;
            point_type Y = x[jj];

            if( cell_index[0] >= nx ) Y[0] += xmax;
            if( cell_index[0] < 0   ) Y[0] -= xmax;
            if( cell_index[1] >= ny ) Y[1] += ymax;
            if( cell_index[1] < 0   ) Y[1] -= ymax;

            A += interaction(X, Y);
          }
        }
      }

      thrust::get<4>( t ) = A;
    }
  };

  void operator()(point_vector const &x, point_vector const &v,
          point_vector &a, double t) const
  {
    typedef thrust::counting_iterator< size_t > ci;

    // Reset the ordering.
    thrust::copy(ci(0), ci(prm.n), part_ord.begin());

    // Assign each particle to a cell.
    thrust::for_each(
        thrust::make_zip_iterator( thrust::make_tuple(
            x.begin(), cell_coo.begin(), cell_idx.begin()
            ) ) ,
        thrust::make_zip_iterator( thrust::make_tuple(
            x.end(), cell_coo.end(), cell_idx.end()
            ) ) ,
        fill_index_n_hash( prm ));

    // Sort particle numbers in part_ord by cell numbers.
    thrust::sort_by_key(cell_idx.begin(), cell_idx.end(),
            part_ord.begin());

    // Find range of each cell in cell_idx array.
    thrust::lower_bound(cell_idx.begin(), cell_idx.end(),
            ci(0), ci(prm.n_cells), cells_begin.begin());

    thrust::upper_bound(cell_idx.begin(), cell_idx.end(),
            ci(0), ci(prm.n_cells), cells_end.begin());

    // Handle boundary conditions
    thrust::transform(x.begin(), x.end(), x_bc.begin(),
            bc_functor(prm));

    // Calculate the local and interacttion forces.
    thrust::for_each(
        thrust::make_zip_iterator( thrust::make_tuple(
            x_bc.begin(), v.begin(), cell_coo.begin(),
            ci(0), a.begin()
            ) ),
        thrust::make_zip_iterator( thrust::make_tuple(
            x_bc.end(), v.end(), cell_coo.end(),
            ci(prm.n), a.end()
            ) ),
        interaction_functor(cells_begin, cells_end, part_ord,
            x, v, prm)
        );
  }

  struct bc_functor {
    size_t xmax, ymax;

    bc_functor(const params &p) : xmax(p.x_max), ymax(p.y_max) {}

    __host__ __device__ point_type operator()(point_type p) const {
      return periodic_bc(p, xmax, ymax);
    }
  };

  void bc( point_vector &x ) {
    thrust::transform(x.begin(), x.end(), x.begin(), bc_functor(prm));
  }

  __host__ __device__ static double periodic_bc( double x , double xmax ) {
    double tmp = x - xmax * int( x / xmax );
    return tmp >= 0.0 ? tmp : tmp + xmax;
  }


  __host__ __device__ static point_type periodic_bc( point_type x, double xmax, double ymax) {
    return point_type( periodic_bc( x[0] , xmax ) , periodic_bc( x[1] , ymax ) );
  }


  __host__ __device__ static int check_interval( int i , int max ) {
    int tmp = i % max;
    return tmp >= 0 ? tmp : tmp + max;
  }


  __host__ __device__ static size_t get_cell_idx(index_type index, size_t nx, size_t ny) {
    size_t i1 = check_interval( index[0] , nx );
    size_t i2 = check_interval( index[1] , ny );
    return i1 * ny + i2;
  }
};


template< typename LocalForce , typename Interaction >
md_system_bs< LocalForce , Interaction > make_md_system_bs(
    size_t n , LocalForce const &f1 , Interaction const &f2 ,
    double xmax = 100.0 , double ymax = 100.0 , double cell_size = 2.0
    )
{
  return md_system_bs< LocalForce , Interaction >( n , f1 , f2 , xmax , ymax , cell_size );
}


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
    md_system_bs<local_force, conservative_interaction<lennard_jones> >
    system_type;

  typedef
    system_type::point_vector
    point_vector;

  typedef
    system_type::point_type
    point_type;

  system_type sys = make_md_system_bs(
      n,
      local_force(),
      make_conservative_interaction( lennard_jones() ),
      domain_size, domain_size, cell_size
      );

  point_vector x(sys.num_points());
  point_vector v(sys.num_points());

  {
    std::vector<point_type> X(sys.num_points());
    std::vector<point_type> V(sys.num_points());

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

  cudaThreadSynchronize();
  boost::timer::cpu_timer timer;
  for( size_t oi = 0 ; oi < m ; ++oi ) {
    for( size_t ii = 0 ; ii < 5 ; ++ii, t+=dt )
      stepper.do_step( sys, std::make_pair(boost::ref(x), boost::ref(v) ), t, dt );

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
  log_perf("thrust_v2", n, m * 5, timer.elapsed());
  std::cout << std::endl;
}
