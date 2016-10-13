#ifndef THRUST_OPERATIONS_HPP
#define THRUST_OPERATIONS_HPP

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

namespace ncwg {

struct thrust_operations {
    template<class Fac1 = double, class Fac2 = Fac1>
    struct scale_sum2 {
        const Fac1 m_alpha1;
        const Fac2 m_alpha2;

        scale_sum2(const Fac1 alpha1, const Fac2 alpha2)
            : m_alpha1(alpha1), m_alpha2(alpha2) { }

        template< class Tuple >
        __host__ __device__ void operator()(Tuple t) const {
            thrust::get<0>(t) = m_alpha1 * thrust::get<1>(t) +
                                m_alpha2 * thrust::get<2>(t);
        }
    };

    template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2, class Fac4 = Fac3, class Fac5 = Fac4>
    struct scale_sum5 {
        const Fac1 m_alpha1;
        const Fac2 m_alpha2;
        const Fac3 m_alpha3;
        const Fac4 m_alpha4;
        const Fac5 m_alpha5;

        scale_sum5(const Fac1 alpha1, const Fac2 alpha2, const Fac3 alpha3,
                const Fac4 alpha4, const Fac5 alpha5)
            : m_alpha1(alpha1), m_alpha2(alpha2), m_alpha3(alpha3),
              m_alpha4(alpha4), m_alpha5(alpha5)
        { }

        template< class Tuple >
        __host__ __device__ void operator()(Tuple t) const {
            thrust::get<0>(t) = m_alpha1 * thrust::get<1>(t) +
                                m_alpha2 * thrust::get<2>(t) +
                                m_alpha3 * thrust::get<3>(t) +
                                m_alpha4 * thrust::get<4>(t) +
                                m_alpha5 * thrust::get<5>(t);
        }
    };
};

}

#endif
