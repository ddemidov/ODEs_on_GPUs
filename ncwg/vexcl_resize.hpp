#ifndef VEXCL_RESIZE_HPP
#define VEXCL_RESIZE_HPP

#include <vexcl/vector.hpp>
#include <vexcl/multivector.hpp>

#include "resize.hpp"

namespace ncgw {

template<class T>
void resize(const vex::vector<T> &in, vex::vector<T> &out) {
    out.resize(in.queue_list(), in.size());
}

template<class T, size_t N>
void resize(const vex::multivector<T,N> &in,
        vex::multivector<T,N> &out)
{
    out.resize(in.queue_list(), in.size());
}

}

#endif
