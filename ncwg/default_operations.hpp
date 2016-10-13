#ifndef DEFAULT_OPERATIONS_HPP
#define DEFAULT_OPERATIONS_HPP

namespace ncwg {

struct default_operations {
    template<class Fac1 = double, class Fac2 = Fac1>
    struct scale_sum2 {
        typedef void result_type;

        const Fac1 alpha1;
        const Fac2 alpha2;

        scale_sum2(Fac1 alpha1, Fac2 alpha2)
	    : alpha1(alpha1), alpha2(alpha2) { }

        template<class T0, class T1, class T2>
        void operator()(T0 &t0, const T1 &t1, const T2 &t2) const {
            t0 = alpha1 * t1 + alpha2 * t2;
        }
    };

    template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2,
	     class Fac4 = Fac3,   class Fac5 = Fac4>
    struct scale_sum5 {
        typedef void result_type;

        const Fac1 alpha1;
        const Fac2 alpha2;
        const Fac3 alpha3;
        const Fac4 alpha4;
        const Fac5 alpha5;

        scale_sum5(Fac1 alpha1, Fac2 alpha2, Fac3 alpha3,
		   Fac4 alpha4, Fac5 alpha5)
            : alpha1(alpha1), alpha2(alpha2), alpha3(alpha3) ,
              alpha4(alpha4), alpha5(alpha5)
        { }

        template<class T0, class T1, class T2, class T3, class T4, class T5>
        void operator()(T0 &t0, const T1 &t1, const T2 &t2, const T3 &t3,
			const T4 &t4, const T5 &t5) const
        {
            t0 = alpha1*t1 + alpha2*t2 + alpha3*t3 + alpha4*t4 + alpha5*t5;
        }

    };
};

}

#endif
