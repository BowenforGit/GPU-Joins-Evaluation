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
#include "tpc_utils.hpp"
#include "../data_gen/generator.cuh"
#include "sort_merge_join.cuh"
#include "partitioned_hash_join.cuh"
#include "sort_hash_join.cuh"
#include "experiment_util.cuh"
#include "join_base.hpp"

using namespace std;
#define KEY_COL_T long
#define PAYLOAD_COL_T long


#define LOAD_Q95_COL(dir, p, N, from, to, seed) \
    to* p; \
    GET_DATA_TYPE(p) \
    { \
        std::string filename = dir + "/" + #p + ".bin"; \
        read_col<from,to>(filename, p, (N), true, (seed)); \
    }

#define LOAD_Q95_COL_ALL(dir, Nws) \
    LOAD_Q95_COL((dir), ws_order_number, (Nws), int, KEY_COL_T, 42) \
    LOAD_Q95_COL((dir), ws_warehouse_sk, (Nws), long, PAYLOAD_COL_T, 42)

void run_q95(std::string algo, std::string profile_output_) {
    std::string dir(TPC_DATA_PREFIX"tpcds_sf100/q95");
    const int Nws = 72001237;

    LOAD_Q95_COL_ALL(dir, Nws)

    using ws_t = struct Chunk<ws_order_number_t, ws_warehouse_sk_t>;
    using join_t = struct Chunk<ws_order_number_t, ws_warehouse_sk_t, ws_warehouse_sk_t>;

    auto ws_cols = std::make_tuple(ws_order_number, ws_warehouse_sk);

    ScanOperator<ws_t> ws1_scan(std::move(ws_cols), Nws, Nws);
    ScanOperator<ws_t> ws2_scan(std::move(ws_cols), Nws, Nws);

    ws1_scan.open(); ws2_scan.open();
    auto relation_ws1 = ws1_scan.next();
    auto relation_ws2 = ws2_scan.next();
    ws1_scan.close(); ws2_scan.close();

    // const int circular_buffer_size = 904010989;
    const int circular_buffer_size = Nws;
    const int first_bit = 0;
    const int log_part1 = 9;
    const int log_part2 = 6;
    
    JoinBase<join_t> *impl;
    if (algo == "SMJ" || algo == "SMJI") {
        impl = new SortMergeJoin<ws_t, ws_t, join_t, true>(relation_ws1, relation_ws2, circular_buffer_size);
    } else if (algo == "PHJ") {
        impl = new PartitionHashJoin<ws_t, ws_t, join_t>(relation_ws1, relation_ws2, log_part1, log_part2, first_bit, circular_buffer_size);
    } else if (algo == "SHJ") {
        impl = new SortHashJoin<ws_t, ws_t, join_t>(relation_ws1, relation_ws2, first_bit, log_part1+log_part2, circular_buffer_size);
    }
    else {
        std::cout << "Invalid algorithm name: " << algo << std::endl;
        exit(1);
    }

    auto out = impl->join();
    cudaDeviceSynchronize();

    cout << "\nOutput Cardinality = " << out.num_items << endl;
    cout << "Results (first 10 items): \n";
    out.peek(min(10, out.num_items));

    impl->print_stats();

    std::ofstream fout;
    fout.open(profile_output_, ios::app);
    fout << get_utc_time() << ","
        << relation_ws1.num_items << "," << relation_ws2.num_items << ","
        << algo << ",";

    auto stats = impl->all_stats();
    for(auto t : stats) {
        fout << t << ",";
    }

    fout << std::endl;
    fout.close();

    relation_ws1.free_mem();
    relation_ws2.free_mem();
    out.free_mem();
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: ./bin/volcano/q95 <algo> <profile_output>" << std::endl;
        exit(1);
    }
    std::string algo = argv[1];
    std::string profile_output = argv[2];
    run_q95(algo, profile_output);
    return 0;
}