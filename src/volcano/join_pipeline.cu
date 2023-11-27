#define CUB_STDERR

#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <unistd.h>
#include <fstream>

#include <cuda.h>

#include "tuple.cuh"
#include "utils.cuh"
#include "operators.cuh"
#include "../data_gen/generator.cuh"
#include "sort_merge_join.cuh"
#include "partitioned_hash_join.cuh"
#include "sort_hash_join.cuh"
#include "experiment_util.cuh"
#include "join_base.hpp"

using namespace std;

DECL_TUP_1_TO_8(int, int)

template<typename R, typename S, typename T>
auto get_joiner(std::string algo, R dim, S fact) {
    JoinBase<T>* impl;
    
    if(algo == "SMJ" || (algo == "SMJI" && dim.num_cols == 2 && fact.num_cols == 2)) {
        impl = new SortMergeJoin<R, S, T, true>(dim, fact, fact.num_items);
    } else if(algo == "PHJ") {
        impl = new PartitionHashJoin<R, S, T>(dim, fact, 9, 6, 0, fact.num_items);
    } else if(algo == "SHJ") {
        impl = new SortHashJoin<R, S, T>(dim, fact, 0, 15, fact.num_items);
    } else if(algo == "SMJI") {
        impl = new SortMergeJoinByIndex<R, S, T, false>(dim, fact, fact.num_items);
    } 
    else {
        std::cout << "Unsupported join algorithm\n";
        std::exit(-1);
    }

    return impl;
}

#define JOIN_PIPELINE(jid, ljid, fdim, odim) \
    float gather_time ## jid = 0; \
    TU ## fdim fact_j ## jid; \
    fact_j ## jid.num_items = fact.num_items; \
    int* fk ## jid = nullptr; \
    TIME_FUNC((fk ## jid = gather(COL(fact, (jid)), COL(r ## ljid, r ## ljid.num_cols-1), fact.num_items)), gather_time ## jid); \
    fact_j ## jid.add_column(fk ## jid); \
    for_<r ## ljid.num_cols-1>([&] (auto i) { \
        fact_j ## jid.add_column(COL(r ## ljid, i.value+1)); \
    }); \
    auto j ## jid = get_joiner<TupleDim, TU ## fdim, TU ## odim>(algo, dim, fact_j ## jid); \
    auto r ## jid = j ## jid->join(); \
    write_stats(j ## jid, gather_time ## jid, algo, (jid), output); \
    release_mem(COL(r ## jid, 0)); \
    fact_j ## jid.free_mem(); \
    delete j ## jid; \

int* gather(int* in, int* map, const int n) {
    int* out = nullptr;
    allocate_mem(&out, false, sizeof(int)*n);
    thrust::device_ptr<int> dev_data_ptr(in);
    thrust::device_ptr<int> dev_idx_ptr(map);
    thrust::device_ptr<int> dev_out_ptr(out);
    thrust::gather(dev_idx_ptr, dev_idx_ptr+n, dev_data_ptr, dev_out_ptr);
    release_mem(in);
    return out;
}

template<typename Joiner>
void write_stats(Joiner* j, const float gather_time, const std::string algo, const int jid, const std::string output) {
    std::ofstream fout(output, std::ios_base::app);
    fout << get_utc_time() << "," << jid << "," << algo << ",";
    auto stats = j->all_stats();
    for(auto t : stats) {
        fout << t << ",";
    }
    fout << gather_time << ",";
    fout << std::endl;
    fout.close();
}

template<typename TupleFact, typename TupleDim>
void exec_join_pipeline(TupleFact fact, TupleDim dim, std::string algo, std::string output) {
    SETUP_TIMING();
    
    int* tid = nullptr;
    allocate_mem(&tid, false, sizeof(int)*fact.num_items);
    fill_sequence<<<num_tb(fact.num_items), 1024>>>(tid, 0, fact.num_items);
    
    // Join 1
    TU2 fact_j0;
    fact_j0.num_items = fact.num_items;
    fact_j0.add_column(COL(fact,0));
    fact_j0.add_column(tid);
    auto j0 = get_joiner<TupleDim, TU2, TU3>(algo, dim, fact_j0);
    auto r0 = j0->join();
    // r0.peek(8);
    write_stats(j0, 0, algo, 0, output);
    release_mem(COL(r0,0));
    release_mem(COL(fact, 0));
    release_mem(tid);
    delete j0;

    JOIN_PIPELINE(1, 0, 3, 4)
    JOIN_PIPELINE(2, 1, 4, 5)
    JOIN_PIPELINE(3, 2, 5, 6)
    JOIN_PIPELINE(4, 3, 6, 7)
    JOIN_PIPELINE(5, 4, 7, 8)
    JOIN_PIPELINE(6, 5, 8, 9)
    JOIN_PIPELINE(7, 6, 9, 10)

    std::cout << r7.num_items << "\n";
    cudaDeviceSynchronize();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template<typename TupleFact, typename TupleDim>
void prepare_data(TupleFact& fact, const int N_fact, TupleDim& dim, const int N_dim, std::string data_path_prefix) {
    constexpr int DIM_NUM_COLS = TupleDim::num_cols, FACT_NUM_COLS = TupleFact::num_cols;

    std::array<int*, DIM_NUM_COLS> dim_cols;
    std::array<int*, FACT_NUM_COLS> fact_cols;

    std::string dim_path =  data_path_prefix + "int/" + "r_" + std::to_string(N_dim)  + ".bin";
    std::string fact_path = data_path_prefix + "int/" + "s_" + std::to_string(N_dim) + "_" +std::to_string(N_fact) + "_uniform.bin";
    
    if(input_exists(dim_path)) {
        cout << "Dim table read from disk\n";
        alloc_load_column(dim_path, dim_cols[0], N_dim);
    } else {
        create_integral_relation_unique(&dim_cols[0], N_dim, false, static_cast<int>(0), true, 42);
        write_to_disk(dim_cols[0], N_dim, dim_path);
    }

    if(input_exists(fact_path)) {
        cout << "Fact table read from disk\n";
        alloc_load_column(fact_path, fact_cols[0], N_fact);
    } else {
        create_fk_from_pk_uniform(&fact_cols[0], N_fact, dim_cols[0], N_dim);
        write_to_disk(fact_cols[0], N_fact, fact_path);
    }

    for(int i = 1; i < FACT_NUM_COLS; i++) {
        fact_cols[i] = new int[N_fact];
        memcpy(fact_cols[i], fact_cols[0], sizeof(int)*N_fact);
        knuth_shuffle(fact_cols[i], N_fact, i+42);
    }

    for(int i = 1; i < DIM_NUM_COLS; i++) {
        dim_cols[i] = new int[N_dim];
        memcpy(dim_cols[i], dim_cols[0], sizeof(int)*N_dim);
    }

    cout << "Data preparation is done\n";

    auto b_cols = std::tuple_cat(dim_cols);
    auto p_cols = std::tuple_cat(fact_cols);

    ScanOperator<TupleDim> op1(std::move(b_cols), N_dim, N_dim);
    ScanOperator<TupleFact> op2(std::move(p_cols), N_fact, N_fact);

    op1.open(); op2.open();
    dim = op1.next();
    fact = op2.next();
    op1.close(); op2.close();

    release_mem(dim.select_vec);
    release_mem(fact.select_vec);
    dim.select_vec = nullptr;
    fact.select_vec = nullptr;

    for(int i = 0; i < DIM_NUM_COLS; i++) {
        delete [] dim_cols[i];
    }

    for(int i = 0; i < FACT_NUM_COLS; i++) {
        delete [] fact_cols[i];
    }
}

void print_usage() {
    std::cout << "Application: Run a sequence of eight joins\n";
    std::cout << "Usage: <binary> <data_path_prefix> <N_dim> <N_fact> <algo>\n";
    std::cout << "data_path_prefix: path to the generated data directory if any; otherwise provide a location where you want the generated data to be stored\n";
    std::cout << "N_dim: number of tuples in the dimension table (in log-scale)\n";
    std::cout << "N_fact: number of tuples in the fact table (in log-scale)\n";
    std::cout << "algo: SMJ, PHJ, SHJ, SMJI (case sensitive)\n";
    std::cout << "Example: ./bin/volcano/join_pipeline /home/username/data/ 25 27 SMJ\n";
    std::cout << "Use the sort merge join to join a fact table with 2^27 tuples with 8 dimension table with 2^25 tuples\n";
    std::exit(-1);
}

int main(int argc, char** argv) {
    if(argc != 5) {
        print_usage();
    }
    
    std::string data_path_prefix = argv[1];
    if(data_path_prefix.back() != '/') data_path_prefix += "/";
    int N_dim = atoi(argv[2]);
    int N_fact = atoi(argv[3]);
    std::string algo = argv[4];

    TU2 dim;
    TU8 fact;

    std::string output = "join_pipeline_" + std::to_string(N_dim) + "_" + std::to_string(N_fact) + "_8.csv";
    prepare_data(fact, (1 << N_fact), dim, (1 << N_dim), data_path_prefix);
    exec_join_pipeline(fact, dim, algo, output);

    return 0;
}