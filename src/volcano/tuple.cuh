#pragma once

#include <iostream>
#include <cuda.h>
#include <tuple>
#include <cassert>
#include "utils.cuh"

template<int N, typename T>
__host__ __device__ auto get_typed_ptr(T t) {
    using full_chunk_t = typename T::full_chunk_t;
    return full_chunk_t::template get_typed_ptr<N>(t.data);
}

template<typename... Args>
struct Chunk {
    static constexpr int num_cols = sizeof...(Args);
    static constexpr size_t row_bytes = (sizeof(Args) + ... + 0);
    static constexpr size_t max_col_size = std::max({sizeof(Args)...});
    using row_type = std::tuple<Args*...>;
    using value_type = std::tuple<Args...>;

    // https://stackoverflow.com/a/68879765
    static constexpr std::size_t biggest_idx() {
        std::size_t const sizes[] { sizeof(Args) ... };
        return std::max_element(std::begin(sizes), std::end(sizes)) - std::begin(sizes);
    }
    
    using biggest_col_t = std::tuple_element_t<biggest_idx(), std::tuple<Args...>>;
    
    void* data[num_cols];
    int* select_vec {nullptr}; // len = num_items
    int num_items {0};

    __host__ Chunk() : select_vec(nullptr), 
                       num_items(0), 
                       col_idx_init_(0),
                       data_stream_offset_(0),
                       on_cpu(false),
                       gpu_data_(nullptr),
                       end_of_stream_(false),
                       consolidated(true) { 
    }
    
    __host__ Chunk(int N, size_t off, Args*... args)
                    : num_items(N),
                    capacity(N),
                    data_stream_offset_(off),
                    col_idx_init_(0),
                    on_cpu(false),
                    gpu_data_(nullptr),
                    end_of_stream_(false),
                    consolidated(true) {
        CHECK_LAST_CUDA_ERROR();
        allocate_mem(&select_vec, false, N*sizeof(int));
        assert(select_vec);
        CHECK_CUDA_ERROR(cudaMemset(select_vec, 0xff, N*sizeof(int))); // non-zero means selected
        populate(args...);
        CHECK_LAST_CUDA_ERROR();
    }

    __host__ Chunk(cudaStream_t* st, int N, size_t off, Args*... args)
                    : st_(*st),
                    num_items(N),
                    capacity(N),
                    data_stream_offset_(off),
                    col_idx_init_(0),
                    on_cpu(false),
                    gpu_data_(nullptr),
                    end_of_stream_(false),
                    consolidated(true) {
        CHECK_LAST_CUDA_ERROR();
        allocate_mem(&select_vec, false, N*sizeof(int), *st);
        CHECK_CUDA_ERROR(cudaMemsetAsync(select_vec, 0xff, N*sizeof(int), *st)); // non-zero means selected
        populate_async(st, args...);
        CHECK_LAST_CUDA_ERROR();
    }

    __host__ __device__ bool empty() { return num_items == 0; }
    __host__ bool is_end() { return end_of_stream_; }
    __host__ void set_stream_end() { end_of_stream_ = true; }
    __host__ void set_stream_end(bool end) { end_of_stream_ = end; }
    
    template<class T> 
    __host__ void populate(T* data_src) {
        CHECK_LAST_CUDA_ERROR();
        assert(col_idx_init_ == num_cols - 1);
        size_t sz = sizeof(T)*num_items;
        allocate_mem(&data[col_idx_init_], false, sz);
        CHECK_CUDA_ERROR(cudaMemcpy(data[col_idx_init_], (void*)(data_src+data_stream_offset_), sz, cudaMemcpyHostToDevice));
    }

    template<class T, typename... Rest>
    __host__ void populate(T* data_src, Rest*... rest) {
        CHECK_LAST_CUDA_ERROR();
        size_t sz = sizeof(T)*num_items;
        allocate_mem(&data[col_idx_init_], false, sz);
        CHECK_CUDA_ERROR(cudaMemcpy(data[col_idx_init_], (void*)(data_src+data_stream_offset_), sz, cudaMemcpyHostToDevice));
        col_idx_init_++;
        populate(rest...);
    }

    template<class T> 
    __host__ void populate_async(cudaStream_t* st, T* data_src) {
        CHECK_LAST_CUDA_ERROR();
        assert(col_idx_init_ == num_cols - 1);
        size_t sz = sizeof(T)*num_items;
        allocate_mem(&data[col_idx_init_], false, sz, *st);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(data[col_idx_init_], (void*)(data_src+data_stream_offset_), sz, cudaMemcpyHostToDevice, *st));
    }

    template<class T, typename... Rest>
    __host__ void populate_async(cudaStream_t* st, T* data_src, Rest*... rest) {
        CHECK_LAST_CUDA_ERROR();
        size_t sz = sizeof(T)*num_items;
        allocate_mem(&data[col_idx_init_], false, sz, *st);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(data[col_idx_init_], (void*)(data_src+data_stream_offset_), sz, cudaMemcpyHostToDevice, *st));
        col_idx_init_++;
        populate_async(st, rest...);
    }

    template<int N> 
    __host__ __device__
    auto
    get_typed_ptr() {
        static_assert(N < num_cols);
        return reinterpret_cast<NthTypeOf<N, Args...>*>(data[N]);
    }

    template<int N> 
    __host__ __device__
    static auto
    get_typed_ptr(void** d) {
        static_assert(N < num_cols);
        return reinterpret_cast<NthTypeOf<N, Args...>*>(d[N]);
    }

    __host__
    void free_mem(bool free_select_vec = true, std::vector<int>* skip = nullptr) {
        if(empty()) return;
        if(free_select_vec) {
            if(select_vec) release_mem(select_vec, st_);
            select_vec = nullptr;
        }
        for(int i = 0; i < num_cols; i++) {
            if(!data[i] || (skip && (*skip)[i])) continue;
            if(on_cpu) free(data[i]);
            else if(on_um) CHECK_CUDA_ERROR(cudaFree(data[i]));
            else release_mem(data[i], st_);
            data[i] = nullptr;
        }
        if(gpu_data_) {
            release_mem(gpu_data_, st_);
            gpu_data_ = nullptr;
        };

        capacity = 0;
    }

    __host__
    auto consolidate(bool to_cpu = true) {
        assert(!on_cpu);
        CHECK_LAST_CUDA_ERROR();
        if(!select_vec || is_consolidated()) {
            if(to_cpu)
                return send_data_to_cpu();
            else
                return *this;
        }

        // Cub implementation
        int* d_num_selected_out;
        allocate_mem(&d_num_selected_out, false);
        void* to_free[num_cols];
        for_<num_cols>([&] (auto i) {
            size_t sz = sizeof(NthTypeOf<i.value, Args...>)*num_items;
            auto d_in = get_typed_ptr<i.value>();
            auto d_flags = select_vec;
            decltype(d_in) d_out = nullptr;
            allocate_mem(&d_out, false, sz);
            assert(d_flags && d_in && d_out);
            // Determine temporary device storage requirements
            void     *d_temp_storage = NULL;
            size_t   temp_storage_bytes = 0;
            CHECK_CUDA_ERROR(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items));
            allocate_mem(&d_temp_storage, false, temp_storage_bytes);
            CHECK_CUDA_ERROR(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items));
            to_free[i.value] = data[i.value];
            if(d_temp_storage) release_mem(d_temp_storage);

            data[i.value] = (void*)d_out;
        });

        for(int i = 0; i < num_cols; i++) {
            if(to_free[i]) {
                if(on_um) CHECK_CUDA_ERROR(cudaFree(to_free[i]));
                else release_mem(to_free[i]);
            }
        }

        CHECK_CUDA_ERROR(cudaMemcpy(&num_items, d_num_selected_out, sizeof(num_items), cudaMemcpyDeviceToHost));
        release_mem(d_num_selected_out);
        
        consolidated = true;
        on_um = false;

        if(to_cpu)
            return send_data_to_cpu();
        else
            return *this;
    }

    __host__
    auto add_column(void* col_ptr) {
        assert(col_idx_init_ < num_cols && gpu_data_ == nullptr); // make sure the gpu_data_ is not set, otherwise it may lead to data inconsistency
        data[col_idx_init_++] = col_ptr;
    }

    __host__
    void** get_gpu_data() {
        if(gpu_data_ != nullptr) return gpu_data_;
        void* data_out_g = nullptr;
        allocate_mem(&data_out_g, false, sizeof(void*)*num_cols);
        CHECK_CUDA_ERROR(cudaMemcpy(data_out_g, reinterpret_cast<void*>(data), sizeof(void*)*num_cols, cudaMemcpyHostToDevice));
        gpu_data_ = reinterpret_cast<void**>(data_out_g);
        return gpu_data_;
    }

    __host__
    void set_consolidated(bool c) { consolidated = c; }

    __host__
    bool is_consolidated() { return consolidated; }
    
    // for debugging
    __host__
    int get_selected_items_num() {
        if(!select_vec) {
            std::cout << "Select vector is null!\n";
            return -1;
        }
        int* h_sel = new int[num_items];
        CHECK_CUDA_ERROR(cudaMemcpy(h_sel, select_vec, sizeof(int)*num_items, cudaMemcpyDeviceToHost));
        int cnt = 0;
        for(int i = 0; i < num_items; i++) {
            if(h_sel[i]) cnt++;
        } 

        delete [] h_sel;

        return cnt;
    }

    // for debugging
    __host__
    void get_info() {
        std::cout << "=== Tuple addr = " << this << std::endl;
        std::cout << "num_cols = " << num_cols << std::endl;
        std::cout << "num_items = " << num_items << std::endl;
        std::cout << "select_vec addr = " << select_vec << std::endl;
        std::cout << "addr of data = " << data << std::endl;
        std::cout << "end-of-stream? " << (is_end() ? "yes\n" : "no\n");
        std::cout << "On CPU? " << (on_cpu ? "yes\n" : "no\n");
        for(int i = 0; i < num_cols; i++) {
            std::cout << "Col-" << i << " addr = " << data[i] << std::endl;
        }
        std::cout << "===\n";
    }

    __host__
    void allocate(int N) {
        assert(empty());
        for_<num_cols>([&] (auto i) {
            data[i.value] = nullptr;
            size_t sz = N * sizeof(NthTypeOf<i.value, Args...>);
            allocate_mem(&(data[i.value]), true, sz);
            assert(data[i.value]);
        });
    }

    __host__
    void allocate_um(int N) {
        assert(empty());
        for_<num_cols>([&] (auto i) {
            data[i.value] = nullptr;
            size_t sz = N * sizeof(NthTypeOf<i.value, Args...>);
            CHECK_CUDA_ERROR(cudaMallocManaged(&(data[i.value]), sz));
            assert(data[i.value]);
        });
        on_um = true;
    }

    __host__
    void allocate_cpu(int N) {
        assert(empty());
        for_<num_cols>([&] (auto i) {
            data[i.value] = nullptr;
            size_t sz = N * sizeof(NthTypeOf<i.value, Args...>);
            if(sz != 0) {
                CHECK_CUDA_ERROR(cudaMallocHost(&(data[i.value]), sz));
                assert(data[i.value]);
            }
        });
        on_cpu = true;
    }

    __host__
    void allocate_select_vec() {
        assert(select_vec == nullptr);
        allocate_mem(&select_vec, false, num_items*sizeof(int), st_);
        CHECK_CUDA_ERROR(cudaMemsetAsync(select_vec, 0xff, num_items*sizeof(int), st_)); // non-zero means selected
    }

    __host__
    void peek(int n = 1) {
        if(num_items == 0) {
            std::cout << "Nothing to peek\n";
            return;
        }

        for_<num_cols>([&] (auto i) {
            std::cout << "Column-" << i.value << ": ";
            n = (n > 0) ? std::min(n, num_items) : num_items;
            for(int k = 0; k < n; k++) {
                auto s = get_typed_ptr<i.value>();
                if(!on_cpu && !on_um) {
                    std::remove_pointer_t<decltype(s)> v;
                    CHECK_CUDA_ERROR(cudaMemcpy(&v, s+k, sizeof(v), cudaMemcpyDeviceToHost));
                    std::cout << v << " ";
                } else {
                    std::cout << s[k] << " ";
                }
            }
            std::cout << std::endl;
        });
        std::cout << std::endl;
    }

    __host__
    auto copy() {
        if(empty()) return *this;
        assert(!on_cpu);
        struct Chunk ret = *this; // copy all the states

        // invalidate some states to avoid data aliasing
        ret.gpu_data_ = nullptr;
        ret.set_num_items(0);
        ret.select_vec = nullptr;
        ret.on_um = false;

        // copy all the data fields
        ret.allocate(num_items);
        for_<num_cols>([&](auto i) {
            size_t sz = num_items * sizeof(NthTypeOf<i.value, Args...>);
            CHECK_CUDA_ERROR(cudaMemcpy((ret.data)[i.value], data[i.value], sz, cudaMemcpyDefault));
        });
        
        ret.set_num_items(num_items);
        if(select_vec) {
            ret.allocate_select_vec();
            CHECK_CUDA_ERROR(cudaMemcpy(ret.select_vec, select_vec, num_items*sizeof(*select_vec), cudaMemcpyDefault));
        }
        
        return ret;
    }

    __host__
    void set_num_items(int n) {
        num_items = n;
        capacity = max(capacity, n);
    }

    __host__
    auto& send_data_to_gpu_async(cudaStream_t st) {
        assert(on_cpu && !on_um);
        if(num_items == 0) { return *this; }
        
        for_<num_cols>([&] (auto j) {
            size_t sz = num_items * sizeof(NthTypeOf<j.value, Args...>);
            void* buf {nullptr};
            allocate_mem(&buf, false, sz);
            assert(buf);
            CHECK_CUDA_ERROR(cudaMemcpyAsync(buf, data[j.value], sz, cudaMemcpyHostToDevice, st));
            data[j.value] = buf;
        });
        on_cpu = false;
        on_um = false;

        return *this;
    }   

private:
    template<int N, typename... Ts> using NthTypeOf = typename std::tuple_element<N, std::tuple<Ts...>>::type;
    int col_idx_init_{0};
    size_t data_stream_offset_;
    bool on_cpu {false};
    bool on_um {false};
    void** gpu_data_ {nullptr};
    bool end_of_stream_ {false};
    bool consolidated {true};
    cudaStream_t st_ {0}; // by default, the default stream
    int capacity {0}; // in number of elements

    __host__
    auto& send_data_to_cpu() {
        assert(!on_cpu);
        if(on_um || num_items == 0) { return *this; }
        
        for_<num_cols>([&] (auto j) {
            size_t sz = num_items * sizeof(NthTypeOf<j.value, Args...>);
            void* buf {nullptr};
            buf = malloc(sz);
            assert(buf);
            CHECK_CUDA_ERROR(cudaMemcpy(buf, data[j.value], sz, cudaMemcpyDeviceToHost));
            release_mem(data[j.value]);
            data[j.value] = buf;
        });
        on_cpu = true;
        on_um = false;
        return *this;
    }
};
