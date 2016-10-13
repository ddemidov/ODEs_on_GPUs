#ifndef CONTAINER_ALGEBRA_HPP
#define CONTAINER_ALGEBRA_HPP

namespace ncwg {

struct container_algebra {
    template<class S1, class S2, class S3, class Op>
    static void for_each3(S1 &s1, S2 &s2, S3 &s3, Op op) {
	const size_t dim = s1.size();
	for(size_t n = 0; n < dim; ++n)
	    op(s1[n], s2[n], s3[n]);
    }

    template<class S1, class S2, class S3, class S4, class S5, class S6, class Op>
    static void for_each6(S1 &s1, S2 &s2, S3 &s3, S4 &s4, S5 &s5, S6 &s6, Op op) {
	const size_t dim = s1.size();
	for(size_t n = 0; n < dim; ++n)
	    op(s1[n], s2[n], s3[n], s4[n], s5[n], s6[n]);
    }
};

}

#endif
