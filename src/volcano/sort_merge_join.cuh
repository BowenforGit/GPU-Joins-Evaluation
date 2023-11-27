#pragma once
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <tuple>
#include <type_traits>

#include <cuda.h>
#include <cub/cub.cuh> 

#include "tuple.cuh"
#include "utils.cuh"
#include "merge_path.cuh"
#include "join_base.hpp"
#include <thrust/gather.h>
#include <thrust/device_ptr.h>

template<typename TupleR,
         typename TupleS,
         typename TupleOut,
         bool     kSortMaterializeCombined = false,
         bool     kAlwaysLateMaterialization = false>
class SortMergeJoin : public JoinBase<TupleOut> {
    static_assert(TupleR::num_cols >= 2 && 
                  TupleS::num_cols >= 2 && 
                  TupleR::num_cols+TupleS::num_cols == TupleOut::num_cols+1);
                  
public:
    explicit SortMergeJoin(TupleR r_in, TupleS s_in, int circular_buffer_size) 
             : r(r_in)
             , s(s_in)
             , circular_buffer_size(circular_buffer_size) {
        nr = r.num_items;
        ns = s.num_items;
        
        out.allocate(circular_buffer_size);

        allocate_mem(&r_sorted_keys, false, sizeof(key_t)*nr);
        allocate_mem(&s_sorted_keys, false, sizeof(key_t)*ns);

        allocate_mem(&r_sorted_vals, false, r.max_col_size*nr);
        allocate_mem(&s_sorted_vals, false, s.max_col_size*ns);

        if constexpr (!early_materialization) {
            allocate_mem(&r_match_idx, false, sizeof(int)*circular_buffer_size);
            allocate_mem(&s_match_idx, false, sizeof(int)*circular_buffer_size);
        }

        if(s.max_col_size * ns >= r.max_col_size * nr) {
            using s_biggest_col_t = typename TupleS::biggest_col_t;
            sort_pairs(COL(s, 0), s_sorted_keys, COL(s, TupleS::biggest_idx()), (s_biggest_col_t*)s_sorted_vals, ns); // this does not actually sort
        } else {
            using r_biggest_col_t = typename TupleR::biggest_col_t;
            sort_pairs(COL(r, 0), r_sorted_keys, COL(r, TupleR::biggest_idx()), (r_biggest_col_t*)r_sorted_vals, nr); // this does not actually sort
        }
        
        std::cout << "Allocating " << temp_storage_bytes << " bytes of temporary storage for sorting.\n";
        allocate_mem(&d_temp_storage, false, temp_storage_bytes);

        cudaEventCreate(&start); 
        cudaEventCreate(&stop);
    }

    ~SortMergeJoin() {
        release_mem(r_sorted_keys);
        release_mem(s_sorted_keys);
        release_mem(r_sorted_vals);
        release_mem(s_sorted_vals);
        release_mem(d_temp_storage);

        if constexpr (!early_materialization) {
            release_mem(r_match_idx);
            release_mem(s_match_idx);
        }
    }

public:
    TupleOut join() override {
        TIME_FUNC_ACC(sort(), sort_time);
        TIME_FUNC_ACC(merge(), merge_time);
        TIME_FUNC_ACC(materialize(), mat_time);

        out.num_items = n_matches;

        return out;
    }

    void print_stats() override {
        std::cout << "Sort: " << sort_time << " ms\n"
                  << "Merge: " << merge_time << " ms\n"
                  << "Materialize: " << mat_time << " ms\n\n";
    }

    std::vector<float> all_stats() override {
        return {sort_time, merge_time, mat_time};
    }

public:
    float sort_time {0};
    float merge_time {0};
    float mat_time {0};

private:
    template<typename KeyT>
    void sort_keys(KeyT* k_in, KeyT* k_out, 
                    const size_t num_items) {
        cub::DeviceRadixSort::SortKeys(d_temp_storage, 
                                       temp_storage_bytes, 
                                       k_in, k_out, num_items, 0, 32); 
    }

    template<typename KeyT, typename InputValIt, typename OutputValIt>
    void sort_pairs(KeyT* k_in, KeyT* k_out, 
                    InputValIt v_in, OutputValIt v_out, 
                    const size_t num_items) {
        cub::DeviceRadixSort::SortPairs(d_temp_storage, 
                                        temp_storage_bytes, 
                                        k_in, k_out, v_in, v_out, num_items, 0, 32); 
    }

    void sort() {
        if constexpr (early_materialization || kSortMaterializeCombined) {
            using r_val_t = std::tuple_element_t<1, typename TupleR::value_type>;
            sort_pairs(COL(r,0), r_sorted_keys, COL(r,1), (r_val_t*)r_sorted_vals, nr);
            
            using s_val_t = std::tuple_element_t<1, typename TupleS::value_type>;
            sort_pairs(COL(s,0), s_sorted_keys, COL(s,1), (s_val_t*)s_sorted_vals, ns);
        }
        else {
            sort_keys(COL(r,0), r_sorted_keys, nr);
            sort_keys(COL(s,0), s_sorted_keys, ns);
        }
    }

    void merge() {
        if constexpr (early_materialization) {
            using r_val_t = std::tuple_element_t<1, typename TupleR::value_type>;
            using s_val_t = std::tuple_element_t<1, typename TupleS::value_type>;
            merge_path(r_sorted_keys, 
                       s_sorted_keys, 
                       (r_val_t*)r_sorted_vals,
                       (s_val_t*)s_sorted_vals,
                       nr, ns, 
                       COL(out,0), COL(out,1), COL(out,2), 
                       n_matches, circular_buffer_size);
        }
        else {
            cub::CountingInputIterator<int> r_itr(0);
            cub::CountingInputIterator<int> s_itr(0);
            merge_path(r_sorted_keys, 
                       s_sorted_keys, 
                       r_itr,
                       s_itr,
                       nr, ns, 
                       COL(out,0), r_match_idx, s_match_idx, 
                       n_matches, circular_buffer_size);
        }
    }

    void materialize() {
        if constexpr (!early_materialization) {
            for_<r_cols-1>([&](auto i) {
                using val_t = std::tuple_element_t<i.value+1, typename TupleR::value_type>;
                if(!kSortMaterializeCombined || i.value > 0) sort_pairs(COL(r, 0), r_sorted_keys, COL(r, i.value+1), (val_t*)r_sorted_vals, nr);
                thrust::device_ptr<val_t> dev_data_ptr((val_t*)r_sorted_vals);
                thrust::device_ptr<int> dev_idx_ptr(r_match_idx);
                thrust::device_ptr<val_t> dev_out_ptr(COL(out, i.value+1));
                thrust::gather(dev_idx_ptr, dev_idx_ptr+std::min(circular_buffer_size, n_matches), dev_data_ptr, dev_out_ptr);
            });
            for_<s_cols-1>([&](auto i) {
                constexpr auto k = i.value+r_cols;
                using val_t = std::tuple_element_t<i.value+1, typename TupleS::value_type>;
                if(!kSortMaterializeCombined || i.value > 0) sort_pairs(COL(s, 0), s_sorted_keys, COL(s, i.value+1), (val_t*)s_sorted_vals, ns);
                thrust::device_ptr<val_t> dev_data_ptr((val_t*)s_sorted_vals);
                thrust::device_ptr<int> dev_idx_ptr(s_match_idx);
                thrust::device_ptr<val_t> dev_out_ptr(COL(out, k));
                thrust::gather(dev_idx_ptr, dev_idx_ptr+std::min(circular_buffer_size, n_matches), dev_data_ptr, dev_out_ptr);
            });
        }
    }

private:
    TupleR r;
    TupleS s;
    TupleOut out;
    
    int nr;
    int ns;
    int n_matches;
    int circular_buffer_size;

    using key_t = std::tuple_element_t<0, typename TupleR::value_type>;

    key_t* r_sorted_keys {nullptr};
    key_t* s_sorted_keys {nullptr};
    int*   idx_seq       {nullptr};
    int*   r_shuffle_idx {nullptr};
    int*   s_shuffle_idx {nullptr};
    void*  r_sorted_vals {nullptr};
    void*  s_sorted_vals {nullptr};
    int*   r_match_idx   {nullptr};
    int*   s_match_idx   {nullptr};

    static constexpr auto  r_cols = TupleR::num_cols;
    static constexpr auto  s_cols = TupleS::num_cols;
    static constexpr bool r_materialize_early = (r_cols == 2 && !kAlwaysLateMaterialization);
    static constexpr bool s_materialize_early = (s_cols == 2 && !kAlwaysLateMaterialization);
    static constexpr bool early_materialization = (r_materialize_early && s_materialize_early);

    void* d_temp_storage {nullptr};
    size_t temp_storage_bytes {0};

    cudaEvent_t start;
    cudaEvent_t stop;
};

template<typename TupleR,
         typename TupleS,
         typename TupleOut,
         bool     kAlwaysLateMaterialization = false>
class SortMergeJoinByIndex : public JoinBase<TupleOut> {
    static_assert(TupleR::num_cols >= 2 && 
                  TupleS::num_cols >= 2 && 
                  TupleR::num_cols+TupleS::num_cols == TupleOut::num_cols+1);
public:
    explicit SortMergeJoinByIndex(TupleR r_in, TupleS s_in, int circular_buffer_size) 
             : r(r_in)
             , s(s_in)
             , circular_buffer_size(circular_buffer_size) {
        nr = r.num_items;
        ns = s.num_items;

        allocate_mem(&r_idx, false, sizeof(int)*nr);
        allocate_mem(&s_idx, false, sizeof(int)*ns);
        fill_sequence<<<num_tb(nr), 1024>>>((int*)(r_idx), 0, nr);
        fill_sequence<<<num_tb(ns), 1024>>>((int*)(s_idx), 0, ns);

        struct Chunk<key_t, int> temp_r;
        temp_r.add_column(COL(r,0));
        temp_r.add_column(r_idx);
        temp_r.num_items = nr;
        
        struct Chunk<key_t, int> temp_s;
        temp_s.add_column(COL(s,0));
        temp_s.add_column(s_idx);
        temp_s.num_items = ns;
        
        joiner = new SortMergeJoin<decltype(temp_r), decltype(temp_s), struct Chunk<key_t, int, int>, true, false>(temp_r, temp_s, circular_buffer_size);

        cudaEventCreate(&start); 
        cudaEventCreate(&stop);
    }

    ~SortMergeJoinByIndex() {
        release_mem(r_idx);
        release_mem(s_idx);
        delete joiner;
    }

public:
    TupleOut join() override {
        auto temp_out = joiner->join();
        sort_time = joiner->all_stats()[0];
        merge_time = joiner->all_stats()[1];
        
        n_matches = temp_out.num_items;
        r_match_idx = temp_out.template get_typed_ptr<1>();
        s_match_idx = temp_out.template get_typed_ptr<2>();
        out.allocate(circular_buffer_size);
        cudaMemcpy(COL(out, 0), COL(temp_out, 0), sizeof(key_t)*circular_buffer_size, cudaMemcpyDeviceToDevice);
        out.num_items = n_matches;
        
        TIME_FUNC_ACC(materialize(), mat_time);
        temp_out.free_mem();

        return out;
    }
    void print_stats() override {
        std::cout << "Sort: " << sort_time << " ms\n"
                  << "Merge: " << merge_time << " ms\n"
                  << "Materialize: " << mat_time << " ms\n\n";
    }

    std::vector<float> all_stats() override {
        return {sort_time, merge_time, mat_time};
    }

public:
    float sort_time {0};
    float merge_time {0};
    float mat_time {0};

private:
    void materialize() {
        for_<r_cols-1>([&](auto i) {
            using val_t = std::tuple_element_t<i.value+1, typename TupleR::value_type>;
            thrust::device_ptr<val_t> dev_data_ptr(COL(r, i.value+1));
            thrust::device_ptr<int> dev_idx_ptr(r_match_idx);
            thrust::device_ptr<val_t> dev_out_ptr(COL(out, i.value+1));
            thrust::gather(dev_idx_ptr, dev_idx_ptr+std::min(circular_buffer_size, n_matches), dev_data_ptr, dev_out_ptr);
        });
        for_<s_cols-1>([&](auto i) {
            constexpr auto k = i.value+r_cols;
            using val_t = std::tuple_element_t<i.value+1, typename TupleS::value_type>;
            thrust::device_ptr<val_t> dev_data_ptr(COL(s, i.value+1));
            thrust::device_ptr<int> dev_idx_ptr(s_match_idx);
            thrust::device_ptr<val_t> dev_out_ptr(COL(out, k));
            thrust::gather(dev_idx_ptr, dev_idx_ptr+std::min(circular_buffer_size, n_matches), dev_data_ptr, dev_out_ptr);
        });
    }

private:
    TupleR r;
    TupleS s;
    TupleOut out;
    
    int nr;
    int ns;
    int n_matches;
    int circular_buffer_size;

    using key_t = std::tuple_element_t<0, typename TupleR::value_type>;

    int*   r_idx         {nullptr};
    int*   s_idx         {nullptr};
    int*   r_match_idx   {nullptr};
    int*   s_match_idx   {nullptr};

    static constexpr auto  r_cols = TupleR::num_cols;
    static constexpr auto  s_cols = TupleS::num_cols;

    JoinBase<struct Chunk<key_t, int, int>>* joiner {nullptr};

    cudaEvent_t start;
    cudaEvent_t stop;
};