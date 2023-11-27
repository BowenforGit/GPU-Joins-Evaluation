#pragma once
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>
#include <vector>
#include <tuple>
#include <string>
#include <type_traits>

#include <cuda.h>
#include <cub/cub.cuh> 

#include "tuple.cuh"
#include "utils.cuh"

using namespace std;

class Operator {
public:
    float next_this;
    std::string op_name;
    Operator(std::string&& str) : op_name(str), next_this(0) {}
    virtual float get_op_time() = 0;
};

template<typename Tuple>
class ScanOperator : public Operator {
public:
    ScanOperator(typename Tuple::row_type&& src, size_t num_items, int vec_size)
    : Operator("ScanOperator"),
      in_(std::move(src)), 
      num_items_(num_items), 
      num_cols_(Tuple::num_cols), 
      vec_size_(vec_size),
      start_idx_(0) {}

    virtual void open() {}
    virtual void close() {}
    virtual float get_op_time() override {
        return next_this;
    }

    Tuple next() {
        Tuple ret;
        SETUP_TIMING();
        TIME_FUNC_ACC(ret=next_(), next_this);
        return ret;
    }

protected:
    virtual Tuple next_ () {
        CHECK_LAST_CUDA_ERROR();
        size_t num_items_out = (start_idx_+vec_size_ < num_items_) ? vec_size_ : num_items_-start_idx_;
        if(num_items_out == 0) {
            Tuple ret;
            ret.set_stream_end();
            return ret;
        }
        
        auto pack = std::tuple_cat(std::make_tuple(num_items_out, start_idx_), in_);
        
        start_idx_ += num_items_out;
        return std::make_from_tuple<Tuple>(std::move(pack));
    }

protected:
    using tuple_row_type = typename Tuple::row_type;
    tuple_row_type in_;
    const size_t num_items_;
    const int num_cols_;
    const int vec_size_;
    size_t start_idx_;
};