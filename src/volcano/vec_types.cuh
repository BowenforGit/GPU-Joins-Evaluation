#pragma once

#include <cuda.h>

template<typename P>
struct to_vec4_t {
};

template<>
struct to_vec4_t<int32_t> {
    using type = int4;
};

template<>
struct to_vec4_t<long> {
    using type = long4;
};

template<typename ScalarT, typename VectorT = typename to_vec4_t<ScalarT>::type>
union vec4{
    VectorT    vec ;
    ScalarT i[sizeof(VectorT)/sizeof(ScalarT)];
};