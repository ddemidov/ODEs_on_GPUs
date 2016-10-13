#ifndef THRUST_ALGEBRA_HPP
#define THRUST_ALGEBRA_HPP

namespace ncwg {

struct thrust_algebra {
    template<class S1, class S2, class S3, class Op>
    static void for_each3(S1 &s1, S2 &s2, S3 &s3, Op op) {
        thrust::for_each(
                thrust::make_zip_iterator( thrust::make_tuple(
                        s1.begin(), s2.begin(),  s3.begin() ) ),
                thrust::make_zip_iterator( thrust::make_tuple(
                        s1.end(), s2.end(), s3.end() ) ),
                op);
    }

    template<class S1, class S2, class S3, class S4,
             class S5, class S6, class Op>
    static void for_each6(S1 &s1, S2 &s2, S3 &s3, S4 &s4,
                          S5 &s5, S6 &s6, Op op)
    {
        thrust::for_each(
                thrust::make_zip_iterator( thrust::make_tuple(
                        s1.begin(),
                        s2.begin(),
                        s3.begin(),
                        s4.begin(),
                        s5.begin(),
                        s6.begin() ) ),
                thrust::make_zip_iterator( thrust::make_tuple(
                        s1.end(),
                        s2.end(),
                        s3.end(),
                        s4.end(),
                        s5.end(),
                        s6.end() ) ),
                op);
    }
};

}

#endif
