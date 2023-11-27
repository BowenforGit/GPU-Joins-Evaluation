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
#include "join_base.hpp"
#include "partitioned_hash_join.cuh"
#include "sort_merge_join.cuh"
#include "sort_hash_join.cuh"
#include "experiment_util.cuh"

using namespace std;

template<typename Tuple, typename Functor>
__global__ void gpu_select_tuple(Tuple t) {
    Functor func;

    for(int i = get_cuda_tid(); i < t.num_items; i += nthreads()) {
        if(t.select_vec[i]) t.select_vec[i] = func(t, i);
    }
}

template<typename Tuple, typename T, typename Functor>
__global__ void project_column(T* out, Tuple t) {
    int tid = get_cuda_tid();
    Functor func;

    for(int i = tid; i < t.num_items; i += nthreads()) {
        out[i] = func(t.data, i);
    }
}

template<int SRC>
struct ProjMoveFrom {
    static constexpr bool move = true;
    static constexpr int src_col = SRC;
};

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

template<typename Tuple, 
         typename Functor, 
         typename Upstream>
class SelectTupleOperator : public Operator {
public:
    SelectTupleOperator(Upstream upstream) : Operator("SelectTupleOperator") {
        upstream_ = upstream;
    }
    void open() { CHECK_LAST_CUDA_ERROR(); upstream_->open(); }
    void close() { CHECK_LAST_CUDA_ERROR(); upstream_->close(); }
    virtual float get_op_time() override {
        return next_this - upstream_->next_this;
    }
    Tuple next() {
        Tuple ret;
        SETUP_TIMING();
        TIME_FUNC_ACC(ret=next_(), next_this);
        return ret;
    }

private:
    Tuple next_() {
        CHECK_LAST_CUDA_ERROR();
        Tuple t = upstream_->next();
        if(t.is_end() || t.empty()) return t;

        // it might happen that t does not have allocated a select vector,
        // if so, allocate one now
        if(t.select_vec == nullptr) t.allocate_select_vec();

        gpu_select_tuple<Tuple, Functor><<<num_tb(t.num_items), 1024>>>(t);
        CHECK_LAST_CUDA_ERROR();

        t.set_consolidated(false);

        return t;
    }

private:
    Upstream upstream_;
};

template<typename Tuple, 
         typename Upstream, 
         int BLOCK_THREADS=1024, 
         int ITEMS_PER_THREAD=4>
class MaterializeOperator : public Operator {
public:
    MaterializeOperator(Upstream upstream): Operator("MaterializeOperator") {
        upstream_ = upstream;
    }
    void open() { CHECK_LAST_CUDA_ERROR(); upstream_->open(); }
    void close() { CHECK_LAST_CUDA_ERROR(); upstream_->close(); }
    virtual float get_op_time() override {
        return next_this - upstream_->next_this;
    }
    Tuple next() {
        Tuple ret;
        SETUP_TIMING();
        TIME_FUNC_ACC(ret=next_(), next_this);
        return ret;
    }
private:
    Tuple next_() {
        CHECK_LAST_CUDA_ERROR();
        Tuple t = upstream_->next();
#ifndef NO_MATERIALIZATION
        if(t.is_end() || t.empty()) return t;
        else {
            auto ret = t.consolidate();
            return ret;
        }
#endif
        return t;
    }

private:
    Upstream upstream_;
};

template<typename InputTuple,
         typename OutputTuple,
         typename Upstream,
         typename... Fs>
class ProjectOperator : public Operator {
    static_assert(OutputTuple::num_cols == (sizeof(Fs) + ... + 0));

public:
    ProjectOperator(Upstream upstream) 
    : Operator("ProjectOperator"), 
      upstream_(upstream) {}
    void open() { CHECK_LAST_CUDA_ERROR(); upstream_->open(); }
    void close() { CHECK_LAST_CUDA_ERROR(); upstream_->close(); }
    virtual float get_op_time() override {
        return next_this - upstream_->next_this;
    }
    OutputTuple next() {
        OutputTuple ret;
        SETUP_TIMING();
        TIME_FUNC_ACC(ret=next_(), next_this);
        return ret;
    }

private:
    OutputTuple next_() {
        CHECK_LAST_CUDA_ERROR();
        InputTuple t = upstream_->next();
        OutputTuple ret;
        ret.set_consolidated(t.is_consolidated()); // FIXME Too many states to maintain
        ret.set_stream_end(t.is_end());
        if(t.is_end() || t.empty()) return ret;

        std::vector<int> skip(t.num_cols, 0);
        for_<ret.num_cols>([&] (auto i) {
            using col_t = decltype(ret.template get_typed_ptr<i.value>());
            using func_t = NthTypeOf<i.value, Fs...>;
            col_t new_col = nullptr;
            if constexpr (func_t::move) {
                new_col = (col_t)(t.data[func_t::src_col]);
                skip[func_t::src_col] = 1;
            }
            else {
                allocate_mem(&new_col, false, sizeof(*new_col)*t.num_items);
                project_column<InputTuple, std::remove_pointer_t<col_t>, func_t><<<num_tb(t.num_items), 1024>>>(new_col, t);
                CHECK_LAST_CUDA_ERROR();
            }
            ret.add_column(new_col);
        });
        ret.select_vec = t.select_vec;
        ret.num_items = t.num_items;

        cudaStreamSynchronize(0);
        t.free_mem(false, &skip);

        return ret;
    }

private:
    Upstream upstream_;
    template<int N, typename... Ts> using NthTypeOf = typename std::tuple_element<N, std::tuple<Ts...>>::type;
};

template<typename LeftTuple,
         typename RightTuple,
         typename OutputTuple,
         typename LeftUpstream,
         typename RightUpstream>
class InnerEqJoinOperator : public Operator {
    static_assert(LeftTuple::num_cols + RightTuple::num_cols == OutputTuple::num_cols+1);
public:
    InnerEqJoinOperator(LeftUpstream b_up, 
                        RightUpstream p_up, 
                        int vec_size, 
                        std::string algo_name,
                        std::string profile_output = "")
    : Operator("InnerEqJoinOperator")
    , l_upstream_(b_up)
    , r_upstream_(p_up)
    , vec_size_(vec_size)
    , algo_name_(algo_name)
    , profile_(!profile_output.empty())
    , profile_output_(profile_output) {
        cudaStreamSynchronize(0);
    }

    void open() 
    { CHECK_LAST_CUDA_ERROR(); l_upstream_->open(); r_upstream_->open(); }
    void close() 
    { CHECK_LAST_CUDA_ERROR(); l_upstream_->close(); r_upstream_->close(); }
    virtual float get_op_time() override {
        return next_this - l_upstream_->next_this - r_upstream_->next_this;
    }
    OutputTuple next() {
        OutputTuple ret;
        SETUP_TIMING();
        TIME_FUNC_ACC(ret=next_(), next_this);
        return ret;
    }

private:
    OutputTuple next_() {
        LeftTuple lt = l_upstream_->next();
        RightTuple rt = r_upstream_->next();


        if(lt.is_end() || rt.is_end()) {
            OutputTuple out;
            out.set_stream_end();
            return out;
        }

        lt.consolidate(false);
        rt.consolidate(false);

        release_mem(lt.select_vec);
        release_mem(rt.select_vec);

        auto max_size = std::max(lt.num_items, rt.num_items);
        if(algo_name_ == "SMJ") {
            join_impl_ = new SortMergeJoin<LeftTuple, RightTuple, OutputTuple, true, true>(lt, rt, vec_size_);
        } else if(algo_name_ == "PHJ") {
            join_impl_ = new PartitionHashJoin<LeftTuple, RightTuple, OutputTuple, true>(lt, rt, 9, ((max_size < (1 << 28)) ? 6 : 7), 0, vec_size_);
        } else if(algo_name_ == "SMJI") {
            join_impl_ = new SortMergeJoinByIndex<LeftTuple, RightTuple, OutputTuple, true>(lt, rt, vec_size_);
        } else if(algo_name_ == "SHJ") {
            join_impl_ = new SortHashJoin<LeftTuple, RightTuple, OutputTuple, true>(lt, rt, 0, ((max_size < (1 << 28)) ? 15 : 16), vec_size_);
        } else {
            std::cout << "Unknown join algorithm: " << algo_name_ << std::endl;
            exit(1);
        }

        auto out = join_impl_->join();
        if(profile_) {
            out.get_info();
            join_impl_->print_stats();
            
            std::ofstream fout;
            fout.open(profile_output_, ios::app);
            fout << get_utc_time() << ","
                << lt.num_items << "," << rt.num_items << ","
                << algo_name_ << ",";

            auto stats = join_impl_->all_stats();
            for(auto t : stats) {
                fout << t << ",";
            }

            fout << std::endl;
            fout.close();
        }
        delete join_impl_;
        
        lt.free_mem();
        rt.free_mem();
        return out;
    }

private:
    LeftUpstream l_upstream_;
    RightUpstream r_upstream_;
    int vec_size_; // ignored
    std::string algo_name_;
    bool profile_;
    std::string profile_output_;

    JoinBase<OutputTuple>* join_impl_;
};