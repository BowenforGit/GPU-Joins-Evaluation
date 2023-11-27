#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>

#include "utils.cuh"
#include "experiment_util.cuh"

#define GET_DATA_TYPE(p) using p ## _t = std::remove_pointer_t<decltype(p)>;
#define DEF_DATA_TYPE(p,type) using p ## _t = type;

#define TPC_DATA_PREFIX "/scratch/wubo/vldb2024-tpc/"

#define LOAD_COL(p, N) \
  long* p; \
  GET_DATA_TYPE(p) \
  alloc_load_tpch_column(TPC_DATA_PREFIX"tpch_sf10/"#p".bin", p, (N));

#define READ_COL_SHUFFLE(p, N, from, to, seed) \
  to* p; \
  GET_DATA_TYPE(p) \
  read_col<from,to>(TPC_DATA_PREFIX"tpch_sf10/"#p".bin", p, (N), true, (seed));

#define RET_COL(m) auto c ## m = t.template get_typed_ptr<m>();

template<class T>
void read_col(const std::string& file_name, T*& dst, const int N, bool shuffle=false, const int seed=42) {
    dst = new T[N];
    
    auto shuffle_path = file_name+".shuffled";
    if(shuffle) {
        std::ifstream file(shuffle_path);
        if(file.good()) {
            file.read(reinterpret_cast<char*>(dst), N*sizeof(T));
            file.close();
            return;
        }
    }
    
    alloc_load_column(file_name, dst, N);

    if(!shuffle) return;

    std::default_random_engine e(seed);
    std::shuffle(dst, dst+N, e);

    write_binary_file(dst, N, shuffle_path);
}

template<class T, class U>
void read_col(const std::string& file_name, U*& dst, const int N, bool shuffle=false, const int seed=42) {
    T* temp;
    read_col(file_name, temp, N, shuffle, seed);
    
    if constexpr (std::is_same<T, U>::value) {
        dst = temp;
        return;
    }

    dst = new U[N];
    for(int i = 0; i < N; i++) {
        dst[i] = static_cast<U>(temp[i]);
    }
    delete[] temp;
}