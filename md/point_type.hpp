/* Boost libs/numeric/odeint/examples/point_type.hpp

 Copyright 2009-2012 Karsten Ahnert
 Copyright 2009-2012 Mario Mulansky

 solar system example for Hamiltonian stepper

 Distributed under the Boost Software License, Version 1.0.
(See accompanying file LICENSE_1_0.txt or
 copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef POINT_TYPE_HPP_INCLUDED
#define POINT_TYPE_HPP_INCLUDED


#include <boost/operators.hpp>
#include <ostream>

#ifdef __CUDACC__
#  define HD __host__ __device__
#else
#  define HD
#endif

//[ point_type
/*the point type */
template< class T , size_t Dim >
class point {
    public:
        const static size_t dim = Dim;
        typedef T value_type;
        typedef point< value_type , dim > point_type;

        // ...
        // constructors
        //<-
        HD point( void )
        {
#pragma unroll
            for( size_t i=0 ; i<dim ; ++i ) m_val[i] = 0.0;
        }

        HD point( value_type val )
        {
#pragma unroll
            for( size_t i=0 ; i<dim ; ++i ) m_val[i] = val;
        }

        HD point( value_type x , value_type y )
        {
            if( dim > 0 ) m_val[0] = x;
            if( dim > 1 ) m_val[1] = y;
        }
        //->

        // ...
        // operators
        //<-
        HD T operator[]( size_t i ) const { return m_val[i]; }
        HD T& operator[]( size_t i ) { return m_val[i]; }

        HD point_type& operator+=( point_type p ) {
#pragma unroll
            for( size_t i=0 ; i<dim ; ++i )
                m_val[i] += p[i];
            return *this;
        }

        HD point_type& operator-=( point_type p ) {
#pragma unroll
            for( size_t i=0 ; i<dim ; ++i )
                m_val[i] -= p[i];
            return *this;
        }

        HD point_type& operator+=( value_type val ) {
#pragma unroll
            for( size_t i=0 ; i<dim ; ++i )
                m_val[i] += val;
            return *this;
        }

        HD point_type& operator-=( value_type val ) {
#pragma unroll
            for( size_t i=0 ; i<dim ; ++i )
                m_val[i] -= val;
            return *this;
        }

        HD point_type& operator*=( value_type val ) {
#pragma unroll
            for( size_t i=0 ; i<dim ; ++i )
                m_val[i] *= val;
            return *this;
        }

        HD point_type& operator/=( value_type val ) {
#pragma unroll
            for( size_t i=0 ; i<dim ; ++i )
                m_val[i] /= val;
            return *this;
        }

        //->

    private:

        T m_val[dim];
    };

template <class T, size_t N>
HD point<T, N> operator+(point<T, N> a, point<T, N> b) {
    a += b;
    return a;
}

template <class T, size_t N>
HD point<T, N> operator-(point<T, N> a, point<T, N> b) {
    a -= b;
    return a;
}

template <class T, size_t N>
HD point<T, N> operator-(point<T, N> a) {
#pragma unroll
    for(size_t i = 0; i < N; ++i) a[i] = -a[i];
    return a;
}

template <class T, size_t N>
HD point<T, N> operator*(T a, point<T, N> b) {
    b *= a;
    return b;
}

template <class T, size_t N>
HD point<T, N> operator*(point<T, N> a, T b) {
    a *= b;
    return a;
}

template <class T, size_t N>
HD point<T, N> operator/(point<T, N> a, T b) {
    a /= b;
    return a;
}

template<class T, size_t N>
HD T abs( point<T, N> a ) {
    T sum = 0;
#pragma unroll
    for(size_t i = 0; i < N; ++i) sum += a[i] * a[i];
    return sqrt(sum);
}

#endif //POINT_TYPE_HPP_INCLUDED
