
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

enum lineitem {
    L_ORDERKEY,
    L_SUPPKEY,
    L_SHIPDATE,
    L_EXTENDEDPRICE,
    L_DISCOUNT,
};

enum orders {
    O_CUSTKEY,
    O_ORDERKEY,
};

enum customer {
    C_NATIONKEY,
    C_CUSTKEY,
};

enum supplier {
    S_NATIONKEY,
    S_SUPPKEY,
};

enum nation {
    N_NATIONKEY,
    N_NAME,
};

template<typename Tuple>
struct Pred_0
{
    enum { NATION1_NAME = 4, NATION2_NAME = 5};
    __device__ __forceinline__ Pred_0() {}
    __device__ __forceinline__ bool operator()(Tuple& t, int i) const {
        auto n1 = t.template get_typed_ptr<NATION1_NAME>();
        auto n2 = t.template get_typed_ptr<NATION2_NAME>();
        return (n1[i] == 7 && n2[i] == 8) || (n1[i] == 8 && n2[i] == 7);
    }
};

template<typename Tuple>
struct Pred_1
{
    __device__ __forceinline__ Pred_1() {}
    __device__ __forceinline__ bool operator()(Tuple& t, int i) const {
        auto date = t.template get_typed_ptr<L_SHIPDATE>();
        return 788918400 <= date[i] && date[i] <= 851990400;
    }
};

void execute_q7_tpch_all_long(int vec_size, std::string file, std::string algo) {
    constexpr int N_lineitem = 59986052;
    constexpr int N_orders = 15000000;
    constexpr int N_customer = 1500000;
    constexpr int N_supplier = 100000;
    constexpr int N_nation = 25;

    READ_COL_SHUFFLE(l_orderkey, N_lineitem, long, KEY_COL_T, 42)
    READ_COL_SHUFFLE(l_suppkey, N_lineitem, long, KEY_COL_T, 42)
    READ_COL_SHUFFLE(l_shipdate, N_lineitem, long, long, 42)
    READ_COL_SHUFFLE(l_extendedprice, N_lineitem, long, long, 42)
    READ_COL_SHUFFLE(l_discount, N_lineitem, long, long, 42)
    READ_COL_SHUFFLE(o_custkey, N_orders, long, KEY_COL_T, 42)
    READ_COL_SHUFFLE(o_orderkey, N_orders, long, KEY_COL_T, 42)
    READ_COL_SHUFFLE(c_nationkey, N_customer, long, KEY_COL_T, 42)
    READ_COL_SHUFFLE(c_custkey, N_customer, long, KEY_COL_T, 42)
    READ_COL_SHUFFLE(s_nationkey, N_supplier, long, KEY_COL_T, 42)
    READ_COL_SHUFFLE(s_suppkey, N_supplier, long, KEY_COL_T, 42)
    READ_COL_SHUFFLE(n_nationkey, N_nation, long, KEY_COL_T, 42)
    READ_COL_SHUFFLE(n_name, N_nation, long, long, 42)
    
    DEF_DATA_TYPE(volume, long)
    DEF_DATA_TYPE(l_shipyear, long)
    DEF_DATA_TYPE(combined_group_key, long)
    
    using LINEITEM_T = struct Chunk<l_orderkey_t,l_suppkey_t,l_shipdate_t,l_extendedprice_t,l_discount_t>;
    using ORDERS_T = struct Chunk<o_custkey_t,o_orderkey_t>;
    using CUSTOMER_T = struct Chunk<c_nationkey_t,c_custkey_t>;
    using SUPPLIER_T = struct Chunk<s_nationkey_t,s_suppkey_t>;
    using NATION_T = struct Chunk<n_nationkey_t,n_name_t>;
    using NATION_T = struct Chunk<n_nationkey_t,n_name_t>;
    using SUPPLIER_NATION_JOIN_0 = struct Chunk<s_nationkey_t,s_suppkey_t,n_name_t>;
    using SUPPLIER_NATION_JOIN_0_PROJ_0 = struct Chunk<s_suppkey_t,n_name_t>;
    using CUSTOMER_NATION_JOIN_1 = struct Chunk<c_nationkey_t,c_custkey_t,n_name_t>;
    using CUSTOMER_NATION_JOIN_1_PROJ_1 = struct Chunk<c_custkey_t,n_name_t>;
    using ORDERS_CUSTOMER_JOIN_2 = struct Chunk<o_custkey_t,o_orderkey_t,n_name_t>;
    using ORDERS_CUSTOMER_JOIN_2_PROJ_2 = struct Chunk<o_orderkey_t,n_name_t>;
    using LINEITEM_ORDERS_JOIN_3 = struct Chunk<l_orderkey_t,l_suppkey_t,l_shipdate_t,l_extendedprice_t,l_discount_t,n_name_t>;

    ScanOperator<LINEITEM_T> op_scan_lineitem(std::make_tuple(l_orderkey,l_suppkey,l_shipdate,l_extendedprice,l_discount), N_lineitem, vec_size);
    SelectTupleOperator<LINEITEM_T, Pred_1<LINEITEM_T>, decltype(&op_scan_lineitem)> op_select_1(&op_scan_lineitem);
    ScanOperator<ORDERS_T> op_scan_orders(std::make_tuple(o_custkey,o_orderkey), N_orders, vec_size);
    ScanOperator<CUSTOMER_T> op_scan_customer(std::make_tuple(c_nationkey,c_custkey), N_customer, vec_size);
    ScanOperator<SUPPLIER_T> op_scan_supplier(std::make_tuple(s_nationkey,s_suppkey), N_supplier, vec_size);
    ScanOperator<NATION_T> op_scan_nation(std::make_tuple(n_nationkey,n_name), N_nation, vec_size);
    ScanOperator<NATION_T> op_scan_nation_1(std::make_tuple(n_nationkey,n_name), N_nation, vec_size);
    
    InnerEqJoinOperator<SUPPLIER_T, NATION_T, SUPPLIER_NATION_JOIN_0, decltype(&op_scan_supplier), decltype(&op_scan_nation)> op_join_0(&op_scan_supplier, &op_scan_nation, vec_size, "SMJ");
    ProjectOperator<SUPPLIER_NATION_JOIN_0, SUPPLIER_NATION_JOIN_0_PROJ_0, decltype(&op_join_0), ProjMoveFrom<1>,ProjMoveFrom<2>> op_proj_0(&op_join_0);
    
    InnerEqJoinOperator<CUSTOMER_T, NATION_T, CUSTOMER_NATION_JOIN_1, decltype(&op_scan_customer), decltype(&op_scan_nation_1)> op_join_1(&op_scan_customer, &op_scan_nation_1, vec_size, "SMJ");
    ProjectOperator<CUSTOMER_NATION_JOIN_1, CUSTOMER_NATION_JOIN_1_PROJ_1, decltype(&op_join_1), ProjMoveFrom<1>,ProjMoveFrom<2>> op_proj_1(&op_join_1);
    
    InnerEqJoinOperator<ORDERS_T, CUSTOMER_NATION_JOIN_1_PROJ_1, ORDERS_CUSTOMER_JOIN_2, decltype(&op_scan_orders), decltype(&op_proj_1)> op_join_2(&op_scan_orders, &op_proj_1, vec_size, "SMJ");
    ProjectOperator<ORDERS_CUSTOMER_JOIN_2, ORDERS_CUSTOMER_JOIN_2_PROJ_2, decltype(&op_join_2), ProjMoveFrom<1>,ProjMoveFrom<2>> op_proj_2(&op_join_2);
    
    InnerEqJoinOperator<LINEITEM_T, ORDERS_CUSTOMER_JOIN_2_PROJ_2, LINEITEM_ORDERS_JOIN_3, decltype(&op_select_1), decltype(&op_proj_2)> op_join_3(&op_select_1, &op_proj_2, vec_size, algo, file);
    
    MaterializeOperator<LINEITEM_ORDERS_JOIN_3, decltype(&op_join_3)> op_root(&op_join_3);
    
    std::vector<Operator*> op_vec {&op_scan_lineitem, &op_select_1, &op_scan_orders, &op_scan_customer, &op_scan_supplier, &op_scan_nation, &op_scan_nation_1, &op_join_0, &op_proj_0, &op_join_1, &op_proj_1, &op_join_2, &op_proj_2, &op_join_3, &op_root};
    
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
        cout << "Usage: ./bin/volcano/tpch_q7 <algo> <profile_output_file>" << endl;
        return 0;
    }
    std::string algo(argv[1]);
    std::string profile_output = argv[2];
    execute_q7_tpch_all_long(59986052, profile_output, algo);
    return 0;
}