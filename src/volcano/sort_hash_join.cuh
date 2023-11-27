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
#include "partition_util.cuh"


template<typename TupleR,
         typename TupleS,
         typename TupleOut,
         bool     kAlwaysLateMaterialization = false>
class SortHashJoin : public JoinBase<TupleOut> {
    static_assert(TupleR::num_cols >= 2 && 
                  TupleS::num_cols >= 2 && 
                  TupleR::num_cols+TupleS::num_cols == TupleOut::num_cols+1);

public:
    explicit SortHashJoin(TupleR r_in, TupleS s_in, int first_bit, int radix_bits, int circular_buffer_size) 
             : r(r_in)
             , s(s_in)
             , first_bit(first_bit)
             , circular_buffer_size(circular_buffer_size)
             , radix_bits(radix_bits) {
        nr = r.num_items;
        ns = s.num_items;
        n_partitions = (1 << radix_bits);

        out.allocate(circular_buffer_size);

        allocate_mem(&d_n_matches);

        using s_biggest_col_t = typename TupleS::biggest_col_t;
        using r_biggest_col_t = typename TupleR::biggest_col_t;
        allocate_mem(&r_offsets, false, sizeof(int)*n_partitions);
        allocate_mem(&s_offsets, false, sizeof(int)*n_partitions);
        allocate_mem(&r_work,    false, sizeof(uint64_t)*n_partitions*2);
        allocate_mem(&s_work,    false, sizeof(uint64_t)*n_partitions*2);
        allocate_mem(&rkeys_partitions, false, sizeof(key_t)*(nr+2048));
        allocate_mem(&skeys_partitions, false, sizeof(key_t)*(ns+2048));
        allocate_mem(&rvals_partitions, false, sizeof(r_biggest_col_t)*(nr+2048));
        allocate_mem(&svals_partitions, false, sizeof(s_biggest_col_t)*(ns+2048));
        allocate_mem(&total_work); // initialized to zero

        if constexpr (!early_materialization) {
            allocate_mem(&r_match_idx, false, sizeof(int)*circular_buffer_size);
            allocate_mem(&s_match_idx, false, sizeof(int)*circular_buffer_size);
        }

        cudaEventCreate(&start); 
        cudaEventCreate(&stop);
    }

    ~SortHashJoin() {
        release_mem(d_n_matches);
        release_mem(r_offsets);
        release_mem(s_offsets);
        release_mem(rkeys_partitions);
        release_mem(skeys_partitions);
        release_mem(rvals_partitions);
        release_mem(svals_partitions);
        release_mem(r_work);
        release_mem(s_work);
        release_mem(total_work);
        if constexpr (!early_materialization) {
            release_mem(r_match_idx);
            release_mem(s_match_idx);
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    TupleOut join() override {
        TIME_FUNC_ACC(partition(), partition_time);
        TIME_FUNC_ACC(join_copartitions(), join_time);
        TIME_FUNC_ACC(materialize_by_gather(), mat_time);

        out.num_items = n_matches;

        return out;
    }

    void print_stats() override {
        std::cout << "Partition: " << partition_time << " ms\n"
                  << "Join: " << join_time << " ms\n"
                  << "Materialize: " << mat_time << " ms\n\n";
    }

    std::vector<float> all_stats() override {
        return {partition_time, join_time, mat_time};
    }

public:
    float partition_time {0};
    float join_time {0};
    float mat_time {0};

private:
    template<typename KeyT, typename ValueT>
    void partition_pairs(KeyT*    keys, 
                        ValueT*   values, 
                        KeyT*     keys_out, 
                        ValueT*   values_out, 
                        int*      offsets, 
                        const int num_items) {
        SinglePassPartition<KeyT, ValueT, int> ssp(keys, values, keys_out, values_out, offsets, num_items, first_bit, radix_bits);
        ssp.process();
    }

    void partition() {
        using r_val_t = std::tuple_element_t<1, typename TupleR::value_type>;
        using s_val_t = std::tuple_element_t<1, typename TupleS::value_type>;

        auto rvals = COL(r,1);
        auto svals = COL(s,1);

        partition_pairs(COL(r,0), rvals, 
                        rkeys_partitions, (r_val_t*)rvals_partitions,
                        r_offsets, nr);
        
        partition_pairs(COL(s,0), svals, 
                        skeys_partitions, (s_val_t*)svals_partitions,
                        s_offsets, ns);
        generate_work_units<<<num_tb(n_partitions,512),512>>>(r_offsets, s_offsets, r_work, s_work, total_work, n_partitions, threshold);
    }

    void join_copartitions() {
        CHECK_LAST_CUDA_ERROR();
        constexpr int NT = 512;
        constexpr int VT = 4;
        if constexpr (early_materialization) {
            size_t sm_bytes = (bucket_size + 512) * (sizeof(key_t) + sizeof(r_val_t) + sizeof(int16_t)) + // elem, payload and next resp.
                            (1 << LOCAL_BUCKETS_BITS) * sizeof(int32_t) + // hash table head
                            + SHUFFLE_SIZE * (NT/32) * (sizeof(key_t) + sizeof(r_val_t) + sizeof(s_val_t));
            std::cout << "sm_bytes: " << sm_bytes << std::endl;
            auto join_fn = join_copartitions_arr<NT, VT, LOCAL_BUCKETS_BITS, SHUFFLE_SIZE, key_t, r_val_t, r_val_t*>;
            cudaFuncSetAttribute(join_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_bytes);
            join_fn<<<n_partitions, NT, sm_bytes>>>(
                                                    rkeys_partitions, (r_val_t*)rvals_partitions, 
                                                    skeys_partitions, (s_val_t*)svals_partitions, 
                                                    r_work, s_work, 
                                                    total_work,
                                                    radix_bits, 4096, 
                                                    d_n_matches,
                                                    COL(out,0), COL(out,1), COL(out,2), 
                                                    circular_buffer_size);
        }
        else {
            size_t sm_bytes = (bucket_size + 512) * (sizeof(key_t) + sizeof(int) + sizeof(int16_t)) + // elem, payload and next resp.
                            (1 << LOCAL_BUCKETS_BITS) * sizeof(int32_t) + // hash table head
                            + SHUFFLE_SIZE * (NT/32) * (sizeof(key_t) + sizeof(int)*2);
            std::cout << "sm_bytes: " << sm_bytes << std::endl;
            cub::CountingInputIterator<int> r_itr(0);
            cub::CountingInputIterator<int> s_itr(0);
            
            auto join_fn = join_copartitions_arr<NT, VT, LOCAL_BUCKETS_BITS, SHUFFLE_SIZE, key_t, int>;
            cudaFuncSetAttribute(join_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_bytes);
            join_fn<<<n_partitions, NT, sm_bytes>>>(
                                                    rkeys_partitions, r_itr, 
                                                    skeys_partitions, s_itr, 
                                                    r_work, s_work, 
                                                    total_work,
                                                    radix_bits, 4096, 
                                                    d_n_matches,
                                                    COL(out,0), r_match_idx, s_match_idx, 
                                                    circular_buffer_size);
        }
        CHECK_LAST_CUDA_ERROR();
        cudaMemcpy(&n_matches, d_n_matches, sizeof(n_matches), cudaMemcpyDeviceToHost);
    }

    void materialize_by_gather() {
        if constexpr (!early_materialization) {
            // partition each payload columns and then gather
            for_<r_cols-1>([&](auto i) {
                using val_t = std::tuple_element_t<i.value+1, typename TupleR::value_type>;
                if(i.value > 0) partition_pairs(COL(r, 0), COL(r, i.value+1), rkeys_partitions, (val_t*)rvals_partitions, nullptr, nr);
                thrust::device_ptr<val_t> dev_data_ptr((val_t*)rvals_partitions);
                thrust::device_ptr<int> dev_idx_ptr(r_match_idx);
                thrust::device_ptr<val_t> dev_out_ptr(COL(out, i.value+1));
                thrust::gather(dev_idx_ptr, dev_idx_ptr+std::min(circular_buffer_size, n_matches), dev_data_ptr, dev_out_ptr);
            });
            for_<s_cols-1>([&](auto i) {
                constexpr auto k = i.value+r_cols;
                using val_t = std::tuple_element_t<i.value+1, typename TupleS::value_type>;
                if(i.value > 0) partition_pairs(COL(s, 0), COL(s, i.value+1), skeys_partitions, (val_t*)svals_partitions, nullptr, ns);
                thrust::device_ptr<val_t> dev_data_ptr((val_t*)svals_partitions);
                thrust::device_ptr<int> dev_idx_ptr(s_match_idx);
                thrust::device_ptr<val_t> dev_out_ptr(COL(out, k));
                thrust::gather(dev_idx_ptr, dev_idx_ptr+std::min(circular_buffer_size, n_matches), dev_data_ptr, dev_out_ptr);
            });
        }
    }

private:
    static constexpr auto  r_cols = TupleR::num_cols;
    static constexpr auto  s_cols = TupleS::num_cols;
    static constexpr bool r_materialize_early = (r_cols == 2 && !kAlwaysLateMaterialization);
    static constexpr bool s_materialize_early = (s_cols == 2 && !kAlwaysLateMaterialization);
    static constexpr bool early_materialization = (r_materialize_early && s_materialize_early);
    static constexpr uint32_t log2_bucket_size = 12;
    static constexpr uint32_t bucket_size = (1 << log2_bucket_size);
    static constexpr int LOCAL_BUCKETS_BITS = 11;
    static constexpr int SHUFFLE_SIZE = (early_materialization ? ((TupleR::row_bytes + TupleS::row_bytes <= 24) ? 32 : 24) : (sizeof(std::tuple_element_t<0, typename TupleR::value_type>) == 4 ? 32 : 16));
    static constexpr int threshold = 2*bucket_size;
    
    using key_t = std::tuple_element_t<0, typename TupleR::value_type>;
    using r_val_t = std::tuple_element_t<1, typename TupleR::value_type>;
    using s_val_t = std::tuple_element_t<1, typename TupleS::value_type>;
    
    TupleR r;
    TupleS s;
    TupleOut out;

    int nr;
    int ns;
    int n_matches;
    int circular_buffer_size;
    int first_bit;
    int n_partitions;
    int radix_bits;

    int*   d_n_matches     {nullptr};
    int*   r_offsets       {nullptr};
    int*   s_offsets       {nullptr};
    uint64_t* r_work       {nullptr};
    uint64_t* s_work       {nullptr};
    int*   total_work      {nullptr};
    key_t* rkeys_partitions{nullptr};
    key_t* skeys_partitions{nullptr};
    void*  rvals_partitions{nullptr};
    void*  svals_partitions{nullptr};
    int*   r_match_idx     {nullptr};
    int*   s_match_idx     {nullptr};

    cudaEvent_t start;
    cudaEvent_t stop;
};