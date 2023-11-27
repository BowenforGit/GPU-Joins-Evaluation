/*
Code adapted from  multicore-hashjoins-0.2@https://www.systems.ethz.ch/node/334
All credit to the original author: Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
*/
#pragma once

#include <cmath>
#include <limits>
#include <random>
#include <type_traits>
#include <omp.h>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <atomic>

template<typename T> using zipf_distribution_t = std::discrete_distribution<T>;
// #define RAND_RANGE(N) ((double)rand() / ((double)RAND_MAX + 1) * (N))
#define RAND_RANGE48(N,STATE) ((double)nrand48(STATE)/((double)RAND_MAX+1)*(N))
#define LOCK(L) while ((L).test_and_set()) {}

template<typename T>
void knuth_shuffle(T * keys, const size_t num_tuples, const int user_seed=42)
{
    auto locks = new std::atomic_flag[num_tuples];
    for(size_t i = 0; i < num_tuples; i++) {
        locks[i].clear();
    }

    #pragma omp parallel 
    {
        union {
            unsigned short state[3];
            unsigned int seed_;
        } seed;
        seed.seed_ = omp_get_thread_num() + user_seed;
        seed.state[2] = 0;
        
        #pragma omp for
        for (int i = num_tuples - 1; i > 0; i--) {
            int64_t j = RAND_RANGE48(i, seed.state);
            LOCK(locks[i]);
            LOCK(locks[j]);
            std::swap(keys[i], keys[j]);
            locks[j].clear();
            locks[i].clear();
        }
    }
}

template<typename T, typename DistT>
void create_integral_relation_nonunique(T** relation, const size_t num_tuples,
                                   bool pinned,
                                   const T vmin = std::numeric_limits<T>::min(), 
                                   const T vmax = std::numeric_limits<T>::max(),
                                   int seed = 42) {
    static_assert(std::is_same_v<T, typename DistT::result_type>);
    static_assert(std::is_integral<T>::value);
    
    assert(relation != nullptr);
    if(pinned) {
        cudaMallocHost((void**)relation, sizeof(T)*num_tuples);
    }
    else {
        *relation = new T[num_tuples];
    }

    assert(*relation);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::default_random_engine e(seed+tid);
        DistT dist(vmin, vmax);
        
        #pragma omp for
        for(size_t i = 0; i < num_tuples; i++) {
            (*relation)[i] = dist(e);
        }
    }
}

template<typename T>
void create_integral_relation_unique(T** relation, const size_t num_tuples,
                                   bool pinned,
                                   const T start = 0,
                                   const bool shuffle = true,
                                   int seed = 42, const bool sparse = false) {
    static_assert(std::is_integral<T>::value);
    
    assert(relation != nullptr);
    if(pinned) {
        cudaMallocHost((void**)relation, sizeof(T)*num_tuples);
    }
    else {
        *relation = new T[num_tuples];
    }

    assert(*relation);

    auto ptr = *relation;
    
    if(sparse) {
        // FIXME: this could lead to duplicates
        std::uniform_int_distribution<T> dist;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::default_random_engine e(seed+tid);
            
            #pragma omp for
            for(size_t i = 0; i < num_tuples; i++) {
                ptr[i] = dist(e);
            }
        }
    } 
    else {
        #pragma omp parallel for
        for(size_t i = 0; i < num_tuples; i++) {
            ptr[i] = i+start;
        }
        if(shuffle) {
            // std::default_random_engine e(seed);
            // std::shuffle(ptr, ptr+num_tuples, e);
            knuth_shuffle(ptr, num_tuples);
        }
    }
}

/**
 * Generate a lookup table with the cumulated density function
 *
 * (This is derived from code originally written by Rene Mueller.)
 */
inline std::vector<double> gen_zipf_lut(double zipf_factor, unsigned int alphabet_size) {
    std::vector<double> lut;

	/*
	 * Compute scaling factor such that
	 *
	 *   sum (lut[i], i=1..alphabet_size) = 1.0
	 *
	 */
	// scaling_factor = 0.0;
	// for (unsigned int i = 1; i <= alphabet_size; i++)
	// 	scaling_factor += 1.0 / pow(i, zipf_factor);

	/*
	 * Generate the lookup table
	 */
	for (unsigned int i = 1; i <= alphabet_size; i++) {
		// sum += 1.0 / pow(i, zipf_factor);
        // lut.push_back(sum / scaling_factor);
        lut.push_back(1.0 / pow(i, zipf_factor));
	}

    // cout << "Generate lookup table\n";

	return lut;
}

// overload to generate zipf distribution
template<typename T>
void create_integral_relation_nonunique(T** relation, const size_t num_tuples,
                                        bool pinned,
                                        double zipf_factor, 
                                        unsigned int alphabet_size,
                                        int seed = 42) {
    static_assert(std::is_integral<T>::value);
    
    assert(relation != nullptr);
    if(pinned) {
        cudaMallocHost((void**)relation, sizeof(T)*num_tuples);
    }
    else {
        *relation = new T[num_tuples];
    }

    assert(*relation);

    auto lut = gen_zipf_lut(zipf_factor, alphabet_size);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::default_random_engine e(seed+tid);
        zipf_distribution_t<T> dist(lut.begin(), lut.end());
        
        #pragma omp for
        for(size_t i = 0; i < num_tuples; i++) {
            (*relation)[i] = dist(e);
        }
    }
}

template<typename T>
void create_fk_from_pk_uniform(T** fk, const size_t n_fk, const T* pk, const size_t n_pk, const int seed = 42, const bool pinned = false) {
    if(pinned) {
        cudaMallocHost((void**)fk, sizeof(T)*n_fk);
    }
    else {
        *fk = new T[n_fk];
    }

    auto n_fold = n_fk / n_pk;
    auto remainder = n_fk % n_pk;

    #pragma omp parallel for
    for(size_t i = 0; i < n_fold; i++) {
        memcpy((*fk) + i*n_pk, pk, sizeof(T)*n_pk);
    }

    if(remainder != 0) {
        memcpy((*fk) + n_fold*n_pk, pk, sizeof(T)*remainder);
    }

    // std::default_random_engine e(seed);
    // std::shuffle((*fk), (*fk)+n_fk, e);
    knuth_shuffle((*fk), n_fk);
}

template<typename T>
void create_fk_from_pk_zipf(T** fk, const size_t n_fk, const T* pk, const size_t n_pk, const double zipf_factor, const int seed = 42, const bool pinned = false) {
    auto lut = gen_zipf_lut(zipf_factor, n_pk);

    if(pinned) {
        cudaMallocHost((void**)fk, sizeof(T)*n_fk);
    }
    else {
        *fk = new T[n_fk];
    }
    
    zipf_distribution_t<T> dist(lut.begin(), lut.end());
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::default_random_engine e(seed+tid);
        
        #pragma omp for
        for(size_t i = 0; i < n_fk; i++) {
            (*fk)[i] = pk[dist(e)];
        }
    }
}

template<typename T>
void create_relation_with_identical_elem(T** relation, const size_t n, T elem, const bool pinned = false) {
    if(pinned) {
        cudaMallocHost((void**)relation, sizeof(T)*n);
    }
    else {
        *relation = new T[n];
    }

    #pragma omp parallel for
    for(size_t i = 0; i < n; i++) {
        (*relation)[i] = elem;
    }
}

template<typename T>
void adjust_selectivity(T* keys, const size_t n, const double selectivity, const T start, const int seed = 42) {
    const size_t n_change = n * (1.0 - selectivity);
    const T max = std::numeric_limits<T>::max();

    T k = start;
    for(int i = 0; i < n_change; i++) {
        keys[i] = start;
        k = (k < max) ? k+1 : start;
    }

    std::default_random_engine e(seed);
    std::shuffle(keys, keys+n, e);
}