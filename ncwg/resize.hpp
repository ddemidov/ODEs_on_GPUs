#ifndef RESIZE_HPP
#define RESIZE_HPP

#include <boost/array.hpp>

namespace ncwg {

template< class State >
void resize( const State &in , State &out )
{
    // standard implementation works for containers
    out.resize( in.size() );
}

// specialization for std::array
template< class T , size_t N >
void resize( const boost::array< T , N > & , boost::array< T , N > & ) {
    /* arrays don't need resizing */
}

}

#endif
