#pragma once

#include <cuda.h>
#include <cub/cub.cuh> 
#include <iostream>

#include "mem_manager.hpp"

#define COL(t,i) (t).template get_typed_ptr<(i)>()

// Timing functions are from Anil Shanbhag (https://github.com/anilshanbhag/crystal)

#define SETUP_TIMING() cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

#define TIME_FUNC(f,t) { \
    cudaEventRecord(start, 0); \
    f; \
    cudaEventRecord(stop, 0); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&t, start,stop); \
}

#define TIME_FUNC_ACC(f,t) { \
    cudaEventRecord(start, 0); \
    f; \
    cudaEventRecord(stop, 0); \
    cudaEventSynchronize(stop); \
    float temp; \
    cudaEventElapsedTime(&temp, start,stop); \
    t += temp; \
}

// https://stackoverflow.com/a/47563100/9054082
template<std::size_t N>
struct num { static const constexpr auto value = N; };
template <class F, std::size_t... Is>
__device__ __host__
void for_(F func, std::index_sequence<Is...>)
{
  (func(num<Is>{}), ...);
}
template <std::size_t N, typename F>
__device__ __host__
void for_(F func)
{
  for_(func, std::make_index_sequence<N>());
}

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{/*
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        cudaGetLastError(); // pop out the current error otherwise it'll be carried over
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
*/}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{/*
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
*/}

inline void alloc_by_cuda(void** ptr, bool clear, size_t sz, cudaStream_t stream) {
    CHECK_CUDA_ERROR(cudaMallocAsync(ptr, sz, stream));
    if(clear) CHECK_CUDA_ERROR(cudaMemsetAsync(*ptr, 0, sz, stream));
}

inline size_t get_cuda_free_mem() {
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    std::cout << free << std::endl;

    return free;
}

// Choose your own memory pool size
constexpr size_t mem_pool_size = 25769803776*0.9;
static UserSpaceMM<mem_pool_size>* mm;

inline void alloc_by_rmm_mempool(void** ptr, bool clear, size_t sz, cudaStream_t stream) {
    if(mm == nullptr) mm = new UserSpaceMM<mem_pool_size>();
    mm->allocate(ptr, clear, sz, stream);
}

inline void free_rmm_mempool(void* ptr, cudaStream_t stream) {
    mm->release(ptr, stream);
}

// #define USE_CUDA_MEMALLOC
template<typename T>
inline void allocate_mem(T** ptr, bool clear = true, size_t sz = sizeof(T), cudaStream_t stream = 0) {
    assert(ptr != nullptr);

#ifdef USE_CUDA_MEMALLOC
    alloc_by_cuda((void**)ptr, clear, sz, stream);
#else
    alloc_by_rmm_mempool((void**)ptr, clear, sz, stream);
#endif
}

template<typename T>
inline void release_mem(T* ptr, cudaStream_t stream = 0) {
#ifdef USE_CUDA_MEMALLOC
    CHECK_CUDA_ERROR(cudaFreeAsync(ptr, stream));
#else
    free_rmm_mempool((void*)ptr, stream);
#endif
}

__host__ __device__
inline int num_tb(const int N, const int threads_per_block=1024, const int items_per_thread=1) {
    return (N + threads_per_block*items_per_thread - 1) / (threads_per_block*items_per_thread);
}

void debug_cuda_mem_usage() {
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    std::cout << "Free bytes : " << free << " Total bytes : " << total << std::endl;
}

// The following two utility functions only work for 1D thread layout
__device__ __forceinline__ int get_cuda_tid() {
    return threadIdx.x + blockDim.x * blockIdx.x;
}

__device__ __forceinline__ int nthreads() {
    return blockDim.x * gridDim.x;
}

template<typename T>
void print_gpu_arr(const T* arr, size_t n, size_t offset=0) {
    T* temp = new T[n];
    cudaMemcpy(temp, arr+offset, sizeof(T)*n, cudaMemcpyDeviceToHost);
    printf("%p: ", arr);
    for(auto i = 0; i < n; i++) {
        std::cout << temp[i] << " ";
    }
    std::cout << std::endl;
    delete [] temp;
}

template<typename T>
__global__ void fill_sequence(T* arr, const T start, const size_t N) {
    for(int t = get_cuda_tid(); t < N; t += nthreads()) {
        arr[t] = start+t;
    }
}