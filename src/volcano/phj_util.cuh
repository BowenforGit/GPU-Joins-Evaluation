/**
 * Code are adapted from the EPFL work. https://github.com/psiul/ICDE2019-GPU-Join
*/

#pragma once
#include <cuda.h>
#include "vec_types.cuh"

__global__ void init_metadata_double ( 
                uint64_t  * __restrict__ heads1,
                uint32_t  * __restrict__ buckets_used1,
                uint32_t  * __restrict__ chains1,
                uint32_t  * __restrict__ out_cnts1,
                uint32_t parts1,
                uint32_t buckets_num1,
                uint64_t  * __restrict__ heads2,
                uint32_t  * __restrict__ buckets_used2,
                uint32_t  * __restrict__ chains2,
                uint32_t  * __restrict__ out_cnts2,
                uint32_t parts2,
                uint32_t buckets_num2,
                const uint32_t bucket_size
                ) {
    auto bucket_size_mask = bucket_size - 1;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < buckets_num1; i += blockDim.x*gridDim.x)
        chains1[i] = 0;

    for (int i = tid; i < parts1; i += blockDim.x*gridDim.x)
        out_cnts1[i] = 0;

    for (int i = tid; i < parts1; i += blockDim.x*gridDim.x)
        heads1[i] = (1 << 18) + (((uint64_t) bucket_size_mask) << 32);

    if (tid == 0) {
        *buckets_used1 = parts1;
    }

    for (int i = tid; i < buckets_num2; i += blockDim.x*gridDim.x)
        chains2[i] = 0;

    for (int i = tid; i < parts2; i += blockDim.x*gridDim.x)
        out_cnts2[i] = 0;

    for (int i = tid; i < parts2; i += blockDim.x*gridDim.x)
        heads2[i] = (1 << 18) + (((uint64_t) bucket_size_mask) << 32);

    if (tid == 0) {
        *buckets_used2 = parts2;
    }
}

template<int NT = 1024, 
         int VT = 4, 
         typename KeyT, 
         typename ValT>
__global__ void partition_pass_one (
                                    const KeyT      * __restrict__ S,
                                    const ValT      *__restrict__ P,
                                          uint64_t  * __restrict__ heads,
                                          uint32_t  * __restrict__ buckets_used,
                                          uint32_t  * __restrict__ chains,
                                          uint32_t  * __restrict__ out_cnts,
                                          KeyT      * __restrict__ output_S,
                                          ValT      * __restrict__ output_P,
                                          size_t                   cnt,
                                          uint32_t                 log_parts,
                                          uint32_t                 first_bit,
                                          uint32_t                 log2_bucket_size) {
    constexpr auto NV = NT * VT;

    extern __shared__ int int_shared[];

    const uint32_t bucket_size      = 1 << log2_bucket_size;
    const uint32_t bucket_size_mask = bucket_size - 1;
    
    const uint32_t parts     = 1 << log_parts;
    const KeyT parts_mask = (KeyT)parts - 1;

    union shuffle_space {
        KeyT key_elem[NV];
        ValT val_elem[NV];
    };
    
    uint32_t * router = (uint32_t *) int_shared;
    union shuffle_space * shuffle = (union shuffle_space *)int_shared;
    uint32_t * shuffle_offsets = (uint32_t *)&shuffle[1];
    uint32_t * histogram = (uint32_t *)&shuffle_offsets[parts];
    uint32_t * bucket_id = (uint32_t *)&histogram[parts];
    uint32_t * next_chain = (uint32_t *)&bucket_id[parts];

    auto key_shuffle = (KeyT*)shuffle;
    auto val_shuffle = (ValT*)shuffle;

    /*partition element counter starts at 0*/
    for (size_t j = threadIdx.x ; j < parts ; j += blockDim.x ) {
        // router[1024*4 + parts + j] = 0;
        histogram[j] = 0;
    }
    
    if (threadIdx.x == 0) 
        router[0] = 0;

    __syncthreads();

    /*iterate over the segments*/
    const size_t segment_start = 0;
    const size_t segment_limit = cnt; 
    const size_t segment_end   = segment_start + ((segment_limit - segment_start + NV - 1)/NV)*NV;

    for (size_t i = VT *(threadIdx.x + blockIdx.x * blockDim.x) + segment_start; i < segment_end ; i += VT * blockDim.x * gridDim.x) {
        vec4<KeyT> thread_keys_vec = *(reinterpret_cast<const vec4<KeyT> *>(S + i));

        uint32_t thread_keys[4];

        /*compute local histogram for a chunk of VT*blockDim.x elements*/
        #pragma unroll
        for (int k = 0 ; k < VT ; ++k){
            if (i + k < segment_limit){
                uint32_t partition = (thread_keys_vec.i[k] >> first_bit) & parts_mask;
                atomicAdd(&histogram[partition], 1);
                thread_keys[k] = partition;
            } else {
                thread_keys[k] = 0;
            }
        }

        __syncthreads();

        for (size_t j = threadIdx.x; j < parts ; j += blockDim.x ) {
            uint32_t cnt = histogram[j];

            if (cnt > 0){
                atomicAdd(out_cnts + j, cnt);
            
                uint32_t pcnt     ;
                uint32_t bucket   ;
                uint32_t next_buck;

                bool repeat = true;

                while (__any_sync(__activemask(), repeat)){
                    if (repeat){
                        /*check if any of the output bucket is filling up*/
                        uint64_t old_heads = atomicAdd((unsigned long long int*)(heads + j), ((unsigned long long int) cnt) << 32);

                        atomicMin((unsigned long long int*)(heads + j), ((unsigned long long int) (2*bucket_size)) << 32);

                        pcnt       = ((uint32_t) (old_heads >> 32));
                        bucket     =  (uint32_t)  old_heads        ;

                        /*now there are two cases:
                        // 2) old_heads.cnt >  bucket_size ( => locked => retry)
                        // if (pcnt       >= bucket_size) continue;*/

                        if (pcnt < bucket_size){
                            /* 1) old_heads.cnt <= bucket_size*/

                            /*check if the bucket was filled*/
                            if (pcnt + cnt >= bucket_size){
                                if (bucket < (1 << 18)) {
                                    next_buck = atomicAdd(buckets_used, 1);                                
                                    chains[bucket]     = next_buck;
                                } else {
                                    next_buck = j;
                                }
                                uint64_t tmp =  next_buck + (((unsigned long long int) (pcnt + cnt - bucket_size)) << 32);

                                atomicExch((unsigned long long int*)(heads + j), tmp);
                            } else {
                                next_buck = bucket;
                            }

                            repeat = false;
                        }
                    }
                }

                shuffle_offsets[j] = atomicAdd(router, cnt);
                histogram[j] = 0;//cnt;//pcnt     ;
                bucket_id[j] = (bucket    << log2_bucket_size) + pcnt;
                next_chain[j] =  next_buck << log2_bucket_size        ;
            }
        }

        __syncthreads();


        uint32_t total_cnt = router[0];

        __syncthreads();

        /*calculate write positions for block-wise shuffle => atomicAdd on start of partition*/
        #pragma unroll
        for (int k = 0 ; k < VT ; ++k){
            if (i + k < segment_limit)
                thread_keys[k] = atomicAdd(&shuffle_offsets[thread_keys[k]], 1);
        }

        /*write the keys in shared memory*/
        #pragma unroll
        for (int k = 0 ; k < VT ; ++k) 
            if (i + k < segment_limit)
                key_shuffle[thread_keys[k]] = thread_keys_vec.i[k];

        __syncthreads();

        int32_t thread_parts[VT];

        /*read shuffled keys and write them to output partitions "somewhat" coalesced*/
        #pragma unroll
        for (int k = 0 ; k < VT ; ++k){
            if (threadIdx.x + NT * k < total_cnt) {
                KeyT  val       = key_shuffle[threadIdx.x + NT * k];
                uint32_t partition = (val >> first_bit) & parts_mask;
                uint32_t cnt       = shuffle_offsets[partition] - (threadIdx.x + NT * k);
                uint32_t bucket    = bucket_id[partition];

                if (((bucket + cnt) ^ bucket) & ~bucket_size_mask){
                    uint32_t next_buck = next_chain[partition];
                    cnt    = ((bucket + cnt) & bucket_size_mask);
                    bucket = next_buck;
                }
                
                bucket += cnt;
        
                output_S[bucket] = val;

                thread_parts[k] = partition;
            }
        }

        __syncthreads();

        /*read payloads of original data*/
        vec4<ValT> thread_vals_vec = *(reinterpret_cast<const vec4<ValT> *>(P + i));

        /*shuffle payloads in shared memory, in the same offsets that we used for their corresponding keys*/
        #pragma unroll
        for (int k = 0 ; k < VT ; ++k) 
            if (i + k < segment_limit) {
                // router[thread_keys[k]] = thread_vals.i[k];
                val_shuffle[thread_keys[k]] = thread_vals_vec.i[k];
            }

        __syncthreads();

        /*write payloads to partition buckets in "somewhat coalesced manner"*/
        #pragma unroll
        for (int k = 0 ; k < VT ; ++k){
            if (threadIdx.x + NT * k < total_cnt) {
                ValT  val       = val_shuffle[threadIdx.x + NT * k];
                uint32_t partition = thread_parts[k];
                uint32_t cnt       = shuffle_offsets[partition] - (threadIdx.x + NT * k);
                uint32_t bucket    = bucket_id[partition];

                if (((bucket + cnt) ^ bucket) & ~bucket_size_mask){
                    uint32_t next_buck = next_chain[partition];
                    cnt    = ((bucket + cnt) & bucket_size_mask);
                    bucket = next_buck;
                }
                bucket += cnt;
        
                output_P[bucket] = val;
            }
        }

        if (threadIdx.x == 0) router[0] = 0;
    }
}

template<int NT = 1024, 
         int VT = 4, 
         typename KeyT, 
         typename ValT>
__global__ void partition_pass_two (
                                    const KeyT      * __restrict__ S,
                                    const ValT      * __restrict__ P,
                                    const uint32_t  * __restrict__ bucket_info,
                                          uint32_t  * __restrict__ buckets_used,
                                          uint64_t  *              heads,
                                          uint32_t  * __restrict__ chains,
                                          uint32_t  * __restrict__ out_cnts,
                                          KeyT      * __restrict__ output_S,
                                          ValT      * __restrict__ output_P,
                                          uint32_t                 log_parts,
                                          uint32_t                 first_bit,
                                          uint32_t  *              bucket_num_ptr,
                                          uint32_t                 log2_bucket_size) {
    constexpr auto NV = NT * VT;
    
    extern __shared__ int int_shared[];

    const uint32_t bucket_size      = 1 << log2_bucket_size;
    const uint32_t bucket_size_mask = bucket_size - 1;
    
    const uint32_t parts     = 1 << log_parts;
    const KeyT parts_mask = (KeyT)parts - 1;

    uint32_t buckets_num = *bucket_num_ptr;

    union shuffle_space {
        KeyT key_elem[NV];
        ValT val_elem[NV];
    };

    uint32_t * router = (uint32_t *) int_shared; //[1024*4 + parts];
    union shuffle_space     * shuffle = (union shuffle_space *)int_shared;
    uint32_t * shuffle_offsets = (uint32_t *)&shuffle[1];
    uint32_t * histogram = (uint32_t *)&shuffle_offsets[parts];
    uint32_t * bucket_id = (uint32_t *)&histogram[parts];
    uint32_t * next_chain = (uint32_t *)&bucket_id[parts];

    auto key_shuffle = (KeyT*)shuffle;
    auto val_shuffle = (ValT*)shuffle;

    for (size_t j = threadIdx.x ; j < parts ; j += blockDim.x ) {
        histogram[j] = 0;
    }
    
    if (threadIdx.x == 0) 
        router[0] = 0;

    __syncthreads();
    
    /*each CUDA block processes a bucket at a time*/
    for (size_t i = blockIdx.x; i < buckets_num; i += gridDim.x) {
        uint32_t info = bucket_info[i];
        /*number of elements per bucket*/
        uint32_t cnt = info & ((1 << 13) - 1);
        /*id of original partition*/
        uint32_t pid = info >> 13;

        for(size_t start = 0; start < cnt; start += NV) {
            auto x = start + VT*threadIdx.x;

            vec4<KeyT> thread_keys_vec = *(reinterpret_cast<const vec4<KeyT> *>(S + bucket_size * i + x));

            uint32_t thread_keys[VT];

            /*compute local histogram for the bucket*/
            #pragma unroll
            for (int k = 0 ; k < VT ; ++k){
                if (x + k < cnt){
                    uint32_t partition = (thread_keys_vec.i[k] >> first_bit) & parts_mask;
                    atomicAdd(&histogram[partition], 1);
                    thread_keys[k] = partition;
                } else {
                    thread_keys[k] = 0;
                }
            }

            __syncthreads();
            for (auto j = threadIdx.x; j < parts ; j += blockDim.x ) {
                uint32_t cnt = histogram[j];

                if (cnt > 0){
                    atomicAdd(out_cnts + (pid << log_parts) + j, cnt);
                    
                    uint32_t pcnt     ;
                    uint32_t bucket   ;
                    uint32_t next_buck;

                    bool repeat = true;

                    while (__any_sync(__activemask(), repeat)){
                        if (repeat){
                            uint64_t old_heads = atomicAdd((unsigned long long int*)(heads + (pid << log_parts) + j), ((unsigned long long int) cnt) << 32);
        
                            atomicMin((unsigned long long int*)(heads + (pid << log_parts) + j), ((unsigned long long int) (2*bucket_size)) << 32);

                            pcnt       = ((uint32_t) (old_heads >> 32));
                            bucket     =  (uint32_t)  old_heads        ;


                            if (pcnt < bucket_size){
                                if (pcnt + cnt >= bucket_size){
                                    if (bucket < (1 << 18)) {
                                        next_buck = atomicAdd(buckets_used, 1);                           
                                        chains[bucket]     = next_buck;
                                    } else {
                                        next_buck = (pid << log_parts) + j;
                                    }

                                    unsigned long long int tmp =  next_buck + (((uint64_t) (pcnt + cnt - bucket_size)) << 32);

                                    atomicExch((unsigned long long int*)(heads + (pid << log_parts) + j), tmp);
                                } else {
                                    next_buck = bucket;
                                }
        
                                repeat = false;
                            }
                        }
                    }

                    shuffle_offsets[j] = atomicAdd(router, cnt);
                    histogram[j] = 0;//cnt;//pcnt     ;
                    bucket_id[j] = (bucket    << log2_bucket_size) + pcnt;
                    next_chain[j] =  next_buck << log2_bucket_size        ;
                }
            }

            __syncthreads();
        
        
            uint32_t total_cnt = router[0];
        
            __syncthreads();

            /*calculate write positions for block-wise shuffle => atomicAdd on start of partition*/
            #pragma unroll
            for (int k = 0 ; k < VT ; ++k){
                if (x + k < cnt) {
                    thread_keys[k] = atomicAdd(&shuffle_offsets[thread_keys[k]], 1);
                }
            }
        
            /*write the keys in shared memory*/
            #pragma unroll
            for (int k = 0 ; k < VT ; ++k) 
                if (x + k < cnt) {
                    key_shuffle[thread_keys[k]] = thread_keys_vec.i[k];
                }
        
            __syncthreads();
        
            int32_t thread_parts[VT];

            /*read shuffled keys and write them to output partitions "somewhat" coalesced*/
            #pragma unroll
            for (int k = 0 ; k < VT ; ++k){
                if (threadIdx.x + NT * k < total_cnt) {
                    KeyT  val       = key_shuffle[threadIdx.x + NT * k];
                    uint32_t partition = (val >> first_bit) & parts_mask;
                    uint32_t cnt       = shuffle_offsets[partition] - (threadIdx.x + NT * k);
                    uint32_t bucket    = bucket_id[partition];


                    if (((bucket + cnt) ^ bucket) & ~bucket_size_mask){
                        uint32_t next_buck = next_chain[partition];
                        cnt    = ((bucket + cnt) & bucket_size_mask);
                        bucket = next_buck;
                    }
                        
                    bucket += cnt;
                
                    output_S[bucket] = val;

                    thread_parts[k] = partition;
                }
            }

            __syncthreads();

            /*read payloads of original data*/
            vec4<ValT> thread_vals_vec = *(reinterpret_cast<const vec4<ValT> *>(P + bucket_size * i + x));

            /*shuffle payloads in shared memory, in the same offsets that we used for their corresponding keys*/
            #pragma unroll
            for (int k = 0 ; k < VT ; ++k) 
                if (x + k < cnt) {
                    val_shuffle[thread_keys[k]] = thread_vals_vec.i[k];
                }

            __syncthreads();

            /*write payloads to partition buckets in "somewhat coalesced manner"*/
            #pragma unroll
            for (int k = 0 ; k < VT ; ++k){
                if (threadIdx.x + NT * k < total_cnt) {
                    ValT  val       = val_shuffle[threadIdx.x + NT * k];
                    uint32_t partition = thread_parts[k];
                    uint32_t cnt       = shuffle_offsets[partition] - (threadIdx.x + NT * k);
                    uint32_t bucket    = bucket_id[partition];

                    if (((bucket + cnt) ^ bucket) & ~bucket_size_mask){
                        // uint32_t next_buck = router[1024 * 4 + 3 * parts + partition];
                        uint32_t next_buck = next_chain[partition];
                        cnt    = ((bucket + cnt) & bucket_size_mask);
                        bucket = next_buck;
                    }
                    bucket += cnt;
            
                    output_P[bucket] = val;
                }
            }

            if (threadIdx.x == 0) router[0] = 0;

            if(start + NV >= cnt) break;
        }
    }
}

__global__ void compute_bucket_info (uint32_t* chains, uint32_t* out_cnts, uint32_t log_parts) {
    uint32_t parts = 1 << log_parts;

    for (int p = threadIdx.x + blockIdx.x*blockDim.x; p < parts; p += gridDim.x*blockDim.x) {
        uint32_t cur = p;
        int32_t cnt = out_cnts[p];

        while (cnt > 0) {
            uint32_t local_cnt = (cnt >= 4096)? 4096 : cnt;
            uint32_t val = (p << 13) + local_cnt;
            
            uint32_t next = chains[cur];
            chains[cur] = val;

            cur = next;
            cnt -= 4096;
        }
    }
}

__global__ void decompose_chains (uint32_t* bucket_info, uint32_t* chains, uint32_t* out_cnts, uint32_t log_parts, int threshold, int bucket_size) {
    uint32_t parts = 1 << log_parts;

    for (int p = threadIdx.x + blockIdx.x*blockDim.x; p < parts; p += gridDim.x*blockDim.x) {
        uint32_t cur = p;
        int32_t  cnt = out_cnts[p];
        uint32_t first_cnt = (cnt >= threshold)? threshold : cnt;
        int32_t  cutoff = 0; 

        while (cnt > 0) {
            cutoff += bucket_size;
            cnt -= bucket_size;

            uint32_t next = chains[cur];
            
            if (cutoff >= threshold && cnt > 0) {
                uint32_t local_cnt = (cnt >= threshold)? threshold : cnt;

                bucket_info[next] = (p << 15) + local_cnt;
                chains[cur] = 0;
                cutoff = 0;
            } else if (next != 0) {
                bucket_info[next] = 0;
            }


            cur = next;
        }

        bucket_info[p] = (p << 15) + first_cnt;
    }
}

// Modified from the EPFL work. The differences are
// 1. We materialize into columns not rows.
// 2. Increase the shuffle memory because we have a bigger L1 cache
template<int NT = 512, 
         int VT = 4, 
         int LOCAL_BUCKETS_BITS = 11, 
         int SHUFFLE_SIZE = 32,
         typename KeyT,
         typename ValT>
__global__ void join_copartitions (
                                    const KeyT*                  R,
                                    const ValT*                  Pr,
                                    const uint32_t*              R_chain,
                                    const uint32_t*              bucket_info,
                                    const KeyT*                  S,
                                    const ValT*                  Ps,
                                    const uint32_t*              S_cnts,
                                    const uint32_t*              S_chain,
                                    int32_t                      log_parts,
                                    uint32_t*                    buckets_num,
                                    uint32_t                     bucket_size,
                                    int32_t*                     results,
                                    KeyT*                        keys_out,
                                    ValT*                        r_output,
                                    ValT*                        s_output,
                                    int32_t                      circular_buffer_size) {
    constexpr int LOCAL_BUCKETS = (1 << LOCAL_BUCKETS_BITS);

    extern __shared__ int16_t temp[];

    struct shuffle_space {
        ValT val_S_elem[SHUFFLE_SIZE];
        ValT val_R_elem[SHUFFLE_SIZE];
        KeyT key_elem[SHUFFLE_SIZE];
    };

    KeyT* elem = (KeyT*)temp;
    ValT* payload = (ValT*)&elem[bucket_size+512];
    int16_t* next = (int16_t*)&payload[bucket_size+512];
    int32_t* head = (int32_t*)&next[bucket_size+512];
    struct shuffle_space * shuffle = (struct shuffle_space *)&head[LOCAL_BUCKETS];

    int tid = threadIdx.x;
    int block = blockIdx.x;
    int width = blockDim.x;
    int pwidth = gridDim.x;
    int parts = 1 << log_parts;

    int lid = tid % 32;
    int gid = tid / 32;
    int gnum = blockDim.x/32;

    int count = 0;

    int ptr;

    int threadmask = (lid < 31)? ~((1 << (lid+1)) - 1) : 0;

    int shuffle_ptr = 0;

    auto warp_shuffle = shuffle + gid;

    int buckets_cnt = *buckets_num;

    using key_vec_t = vec4<KeyT>;
    using value_vec_t = vec4<ValT>;

    for (uint32_t bucket_r = block; bucket_r < buckets_cnt; bucket_r += pwidth) {
        int info = bucket_info[bucket_r];

        if (info != 0) { 
            int p = info >> 15;
            int len_R = info & ((1 << 15) - 1);
            int len_S = S_cnts[p];

            if(len_S == 0) continue;

            if (len_S > 4096+512) {
                int bucket_r_loop = bucket_r;

                for (int offset_r = 0; offset_r < len_R; offset_r += bucket_size) {
                    for (int i = tid; i < LOCAL_BUCKETS; i += blockDim.x)
                        head[i] = -1;
                    __syncthreads();

                    for (int base_r = 0; base_r < bucket_size; base_r += VT*blockDim.x) {
                        key_vec_t data_R = *(reinterpret_cast<const key_vec_t *>(R + bucket_size * bucket_r_loop + base_r + VT*threadIdx.x));
                        value_vec_t data_Pr = *(reinterpret_cast<const value_vec_t *>(Pr + bucket_size * bucket_r_loop + base_r + VT*threadIdx.x));
                        int l_cnt_R = len_R - offset_r - base_r - VT * threadIdx.x;

                        int cnt = 0;                    

                        #pragma unroll
                        for (int k = 0; k < VT; k++) {
                            if (k < l_cnt_R) {
                                auto val = data_R.i[k];
                                elem[base_r + k*blockDim.x + tid] = val;
                                payload[base_r + k*blockDim.x + tid] = data_Pr.i[k];
                                int hval = (val >> log_parts) & (LOCAL_BUCKETS - 1);

                                int32_t last = atomicExch(&head[hval], base_r + k*blockDim.x + tid);
                                next[base_r + k*blockDim.x + tid] = last;
                            }
                        }
                    }

                    bucket_r_loop = R_chain[bucket_r_loop];

                    __syncthreads();

                    int bucket_s_loop = p;
                    int base_s = 0;

                    for (int offset_s = 0; offset_s < len_S; offset_s += VT*blockDim.x) {
                        key_vec_t data_S = *(reinterpret_cast<const key_vec_t *>(S + bucket_size * bucket_s_loop + base_s + VT*threadIdx.x));
                        value_vec_t data_Ps = *(reinterpret_cast<const value_vec_t *>(Ps + bucket_size * bucket_s_loop + base_s + VT*threadIdx.x));
                        int l_cnt_S = len_S - offset_s - VT * threadIdx.x;

                        #pragma unroll
                        for (int k = 0; k < VT; k++) {
                            auto val = data_S.i[k];
                            auto pval = data_Ps.i[k];
                            int32_t hval =  (val >> log_parts) & (LOCAL_BUCKETS - 1);
                            ValT pay;

                            int32_t pos = (k < l_cnt_S)? head[hval] : -1;

                            /*check at warp level whether someone is still following chain => this way we can shuffle without risk*/
                            int pred = (pos >= 0);

                            while (__any_sync(__activemask(), pred)) {
                                int wr_intention = 0;

                                /*we have a match, fetch the data to be written*/
                                if (pred) {
                                    // if (elem[pos] == tval) {
                                    if(elem[pos] == val) {
                                        pay = payload[pos];
                                        wr_intention = 1;
                                        count++;
                                    }

                                    pos = next[pos];
                                    pred = (pos >= 0);
                                }

                                /*find out who had a match in this execution step*/
                                int mask = __ballot_sync(__activemask(), wr_intention);

                                /*our software managed buffer will overflow, flush it*/
                                int wr_offset = shuffle_ptr +  __popc(mask & threadmask);
                                shuffle_ptr = shuffle_ptr + __popc(mask);
                                
                                /*while it overflows, flush
                                we flush 16 keys and then the 16 corresponding payloads consecutively, of course other formats might be friendlier*/
                                while (shuffle_ptr >= SHUFFLE_SIZE) {
                                    if (wr_intention && (wr_offset < SHUFFLE_SIZE)) {
                                        warp_shuffle->val_R_elem[wr_offset] = pay;
                                        warp_shuffle->val_S_elem[wr_offset] = pval;
                                        warp_shuffle->key_elem[wr_offset] = val;
                                        wr_intention = 0;
                                    }

                                   if (lid == 0) {
                                        ptr = atomicAdd(results, SHUFFLE_SIZE);
                                        // ptr = ptr % circular_buffer_size;
                                   }

                                    ptr = __shfl_sync(__activemask(), ptr, 0);

                                    auto w_pos = (ptr + lid) % circular_buffer_size;

                                    if(lid < SHUFFLE_SIZE) {
                                        r_output[w_pos] = warp_shuffle->val_R_elem[lid];
                                        s_output[w_pos] = warp_shuffle->val_S_elem[lid];
                                        keys_out[w_pos] = warp_shuffle->key_elem[lid];
                                    }

                                    wr_offset -= SHUFFLE_SIZE;
                                    shuffle_ptr -= SHUFFLE_SIZE;
                                }

                                /*now the fit, write them in buffer*/
                                if (wr_intention && (wr_offset >= 0)) {
                                    warp_shuffle->val_R_elem[wr_offset] = pay; // R
                                    warp_shuffle->val_S_elem[wr_offset] = pval; // S
                                    warp_shuffle->key_elem[wr_offset] = val; // key
                                    wr_intention = 0;
                                }
                            }                   
                        }

                        base_s += VT*blockDim.x;
                        if (base_s >= bucket_size) {
                            bucket_s_loop = S_chain[bucket_s_loop];
                            base_s = 0;
                        }
                    }

                    __syncthreads();
                }
            } else {
                for (int i = tid; i < LOCAL_BUCKETS; i += blockDim.x)
                    head[i] = -1;

                int rem_s = len_S % 4096;
                rem_s = (rem_s + VT - 1)/VT;

                __syncthreads();

                int off;
                int it;
                int base = 0;

                it = p;
                off = 0;


                for (off = 0; off < len_S;) {
                    key_vec_t data_S = *(reinterpret_cast<const key_vec_t *>(S + bucket_size * it + base + VT*threadIdx.x));
                    value_vec_t data_Ps = *(reinterpret_cast<const value_vec_t *>(Ps + bucket_size * it + base +VT*threadIdx.x));
                    int l_cnt_S = len_S - off - VT * threadIdx.x;

                    #pragma unroll
                    for (int k = 0; k < VT; k++) {
                        if (k < l_cnt_S) {
                            auto val = data_S.i[k];
                            elem[off + tid] = val;
                            payload[off + tid] = data_Ps.i[k];
                            int hval = (val >> log_parts) & (LOCAL_BUCKETS - 1);

                            int32_t last = atomicExch(&head[hval], off + tid);
                            next[off + tid] = last;
                        }   

                        off += (off < bucket_size)? blockDim.x : rem_s;
                        base += blockDim.x;
                    }

                    if (base >= bucket_size) {
                        it = S_chain[it];  
                        base = 0;
                    }
                }

                __syncthreads();

                it = bucket_r;
                off = 0;

                for (; 0 < len_R; len_R -= VT*blockDim.x) {
                    int l_cnt_R = len_R - VT * threadIdx.x;
                    key_vec_t data_R;
                    value_vec_t data_Pr;

                    data_R = *(reinterpret_cast<const key_vec_t *>(R + bucket_size * it + off + VT*threadIdx.x));
                    data_Pr = *(reinterpret_cast<const value_vec_t *>(Pr + bucket_size * it + off + VT*threadIdx.x));

                    #pragma unroll
                    for (int k = 0; k < VT; k++) {
                        auto val = data_R.i[k];
                        auto pval = data_Pr.i[k];
                        int32_t hval =  (val >> log_parts) & (LOCAL_BUCKETS - 1);
                        ValT pay;

                        int32_t pos = (k < l_cnt_R)? head[hval] : -1;

                        /*same as previous code block*/
                        int pred = (pos >= 0);

                        while (__any_sync(__activemask(), pred)) {
                            int wr_intention = 0;

                            if (pred) {
                                if(elem[pos] == val) {
                                    pay = payload[pos];
                                    wr_intention = 1;
                                    count++;
                                }

                                pos = next[pos];
                                pred = (pos >= 0);
                            }

                            int mask = __ballot_sync(__activemask(), wr_intention);

                            int wr_offset = shuffle_ptr +  __popc(mask & threadmask);
                            shuffle_ptr = shuffle_ptr + __popc(mask);
                                
                            while (shuffle_ptr >= SHUFFLE_SIZE) {
                                if (wr_intention && (wr_offset < SHUFFLE_SIZE)) {
                                    warp_shuffle->val_R_elem[wr_offset] = pval; // R
                                    warp_shuffle->val_S_elem[wr_offset] = pay; // S
                                    warp_shuffle->key_elem[wr_offset] = val; // keys
                                    wr_intention = 0;
                                }

                                if (lid == 0) {
                                    ptr = atomicAdd(results, SHUFFLE_SIZE);
                                }

                                ptr = __shfl_sync(__activemask(), ptr, 0);

                                auto w_pos = (ptr + lid) % circular_buffer_size;

                                if(lid < SHUFFLE_SIZE) {
                                    r_output[w_pos] = warp_shuffle->val_R_elem[lid];
                                    s_output[w_pos] = warp_shuffle->val_S_elem[lid];
                                    keys_out[w_pos] = warp_shuffle->key_elem[lid];
                                }

                                wr_offset -= SHUFFLE_SIZE;
                                shuffle_ptr -= SHUFFLE_SIZE;
                            }

                            if (wr_intention && (wr_offset >= 0)) {
                                warp_shuffle->val_R_elem[wr_offset] = pval;
                                warp_shuffle->val_S_elem[wr_offset] = pay;
                                warp_shuffle->key_elem[wr_offset] = val;
                                wr_intention = 0;
                            }
                        }                   
                    }
                    
                    off += VT*blockDim.x;
                    if (off >= bucket_size) {
                        it = R_chain[it];
                        off = 0;
                    }
                }
                
                if(bucket_r + pwidth >= buckets_cnt) break;
                __syncthreads();
            }
        }
    }

    if (lid == 0) {
        ptr = atomicAdd(results, shuffle_ptr);
    }

    ptr = __shfl_sync(__activemask(), ptr, 0);

    if (lid < shuffle_ptr) {
        auto w_pos = (ptr + lid) % circular_buffer_size;
        if(lid < SHUFFLE_SIZE) {
            r_output[w_pos] = warp_shuffle->val_R_elem[lid];
            s_output[w_pos] = warp_shuffle->val_S_elem[lid];
            keys_out[w_pos] = warp_shuffle->key_elem[lid];
        }
    }
}