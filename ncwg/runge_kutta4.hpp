#ifndef RUNGE_KUTTA4_HPP
#define RUNGE_KUTTA4_HPP

#include "resize.hpp"
#include "container_algebra.hpp"
#include "default_operations.hpp"

namespace ncwg {

template<class state_type, class value_type = double,
         class deriv_type = state_type, // to be interchangeable with odeint
         class time_type = value_type,
         class algebra = container_algebra,
         class operations = default_operations>
class runge_kutta4 {
public:
    template<typename System>
    void do_step(System &system, state_type &x,
                     time_type t, time_type dt)
    {
        adjust_size( x );
        const value_type one = 1;
        const time_type dt2 = dt/2, dt3 = dt/3, dt6 = dt/6;

        typedef typename operations::template scale_sum2<
                    value_type, time_type> scale_sum2;

        typedef typename operations::template scale_sum5<
                    value_type, time_type, time_type,
                    time_type, time_type> scale_sum5;

        system(x, k1, t);
        algebra::for_each3(x_tmp, x, k1, scale_sum2(one, dt2));

        system(x_tmp, k2, t + dt2);
        algebra::for_each3(x_tmp, x, k2, scale_sum2(one, dt2));

        system(x_tmp, k3, t + dt2);
        algebra::for_each3(x_tmp, x, k3, scale_sum2(one, dt));

        system(x_tmp, k4, t + dt);
        algebra::for_each6(x, x, k1, k2, k3, k4,
                           scale_sum5(one, dt6, dt3, dt3, dt6));
    }
private:
    state_type x_tmp, k1, k2, k3, k4;

    void adjust_size(const state_type &x) {
        resize(x, x_tmp);
        resize(x, k1); resize(x, k2);
        resize(x, k3); resize(x, k4);
    }
};

#define runge_kutta4_classic runge_kutta4

}

//typedef runge_kutta4< vector<double> , double , double ,
//                      container_algebra , default_operations > rk_stepper;
// equivalent shorthand definition using the default parameters:
//typedef runge_kutta4< vector<double> > rk_stepper;

#endif
