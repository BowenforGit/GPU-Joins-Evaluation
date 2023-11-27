
#include <iostream>
#include <stdio.h>
#include <curand.h>
#include <vector>
#include <tuple>
#include <chrono>
#include <unistd.h>
#include <fstream>
#include <cuda.h>
#include <cub/cub.cuh>

#include "tuple.cuh"
#include "utils.cuh"
#include "operators.cuh"
#include "tpc_utils.hpp"

#define KEY_COL_T long

enum customer {
    C_CUSTKEY,
    C_NAME,
};

enum orders {
    O_CUSTKEY,
    O_ORDERKEY,
    O_ORDERDATE,
    O_TOTALPRICE,
};

enum lineitem {
    L_ORDERKEY,
    L_QUANTITY,
};

void execute_q18_part_tpch_all_long(int vec_size, std::string file, std::string algo) {
    constexpr int N_customer = 1500000;
    constexpr int N_orders = 15000000;
    constexpr int N_lineitem = 59986052;
    
    READ_COL_SHUFFLE(c_custkey, N_customer, long, KEY_COL_T, 42)
    READ_COL_SHUFFLE(c_name, N_customer, long, long, 42)
    READ_COL_SHUFFLE(o_custkey, N_orders, long, KEY_COL_T, 42)
    READ_COL_SHUFFLE(o_orderkey, N_orders, long, KEY_COL_T, 42)
    READ_COL_SHUFFLE(o_orderdate, N_orders, long, long, 42)
    READ_COL_SHUFFLE(o_totalprice, N_orders, long, long, 42)
    READ_COL_SHUFFLE(l_orderkey, N_lineitem, long, KEY_COL_T, 42)
    READ_COL_SHUFFLE(l_quantity, N_lineitem, long, long, 42)
    
    using CUSTOMER_T = struct Chunk<c_custkey_t,c_name_t>;
    using ORDERS_T = struct Chunk<o_custkey_t,o_orderkey_t,o_orderdate_t,o_totalprice_t>;
    using LINEITEM_T = struct Chunk<l_orderkey_t,l_quantity_t>;
    using CUSTOMER_ORDERS_JOIN_0 = struct Chunk<c_custkey_t,c_name_t,o_orderkey_t,o_orderdate_t,o_totalprice_t>;
    using CUSTOMER_ORDERS_JOIN_0_PROJ_0 = struct Chunk<o_orderkey_t,c_name_t,o_orderdate_t,o_totalprice_t>;
    using CUSTOMER_LINEITEM_JOIN_1 = struct Chunk<o_orderkey_t,c_name_t,o_orderdate_t,o_totalprice_t,l_quantity_t>;

    ScanOperator<CUSTOMER_T> op_scan_customer(std::make_tuple(c_custkey,c_name), N_customer, vec_size);
    ScanOperator<ORDERS_T> op_scan_orders(std::make_tuple(o_custkey,o_orderkey,o_orderdate,o_totalprice), N_orders, vec_size);
    ScanOperator<LINEITEM_T> op_scan_lineitem(std::make_tuple(l_orderkey,l_quantity), N_lineitem, vec_size);
    InnerEqJoinOperator<CUSTOMER_T, ORDERS_T, CUSTOMER_ORDERS_JOIN_0, decltype(&op_scan_customer), decltype(&op_scan_orders)> op_join_0(&op_scan_customer, &op_scan_orders, vec_size, "SMJ");
    ProjectOperator<CUSTOMER_ORDERS_JOIN_0, CUSTOMER_ORDERS_JOIN_0_PROJ_0, decltype(&op_join_0), ProjMoveFrom<2>,ProjMoveFrom<1>,ProjMoveFrom<3>,ProjMoveFrom<4>> op_proj_0(&op_join_0);
    InnerEqJoinOperator<CUSTOMER_ORDERS_JOIN_0_PROJ_0, LINEITEM_T, CUSTOMER_LINEITEM_JOIN_1, decltype(&op_proj_0), decltype(&op_scan_lineitem)> op_join_1(&op_proj_0, &op_scan_lineitem, vec_size, algo, file);
    MaterializeOperator<CUSTOMER_LINEITEM_JOIN_1, decltype(&op_join_1)> op_root(&op_join_1);
    
    std::vector<Operator*> op_vec {&op_scan_customer, &op_scan_orders, &op_scan_lineitem, &op_join_0, &op_proj_0, &op_join_1, &op_root};

    op_root.open();
    auto t = op_root.next();
    while(!t.is_end()) {
        cout << "(" << t.num_items << " rows)\n";
        t.free_mem();
        t = op_root.next();
    }
    op_root.close();

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


int main(int argc, char** argv) {
    if(argc < 3) {
        cout << "Usage: ./bin/volcano/tpch_q18 <algo> <profile_output_file>" << endl;
        return 0;
    }
    std::string algo(argv[1]);
    std::string profile_output = argv[2];
    execute_q18_part_tpch_all_long(59986052, profile_output, algo);
    return 0;
}