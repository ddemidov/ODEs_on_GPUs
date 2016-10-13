#ifndef VECTOR_SPACE_ALGEBRA_HPP
#define VECTOR_SPACE_ALGEBRA_HPP

namespace ncwg {

struct vector_space_algebra {
    template<class S1, class S2, class S3, class Op>
    static void for_each3(S1 &s1, S2 &s2, S3 &s3, Op op) {
        op(s1, s2, s3);
    }
    template<class S1, class S2, class S3, class S4, class S5, class S6, class Op>
    static void for_each6(S1 &s1, S2 &s2, S3 &s3, S4 &s4, S5 &s5, S6 &s6, Op op) {
        op(s1, s2, s3, s4, s5, s6);
    }
};

}

#endif
