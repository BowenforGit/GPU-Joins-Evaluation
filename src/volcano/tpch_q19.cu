
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
    L_PARTKEY,
    L_QUANTITY,
    L_SHIPMODE,
    L_EXTENDEDPRICE,
    L_DISCOUNT,
    L_SHIPINSTRUCT,
};

enum part {
    P_PARTKEY,
    P_BRAND, 
    P_CONTAINER, 
    P_SIZE, 
};

template<typename Tuple>
struct Pred_0
{
    __device__ __forceinline__ Pred_0() {}
    __device__ __forceinline__ bool operator()(Tuple& t, int i) const {
        auto si = t.template get_typed_ptr<L_SHIPINSTRUCT>();
        auto sm = t.template get_typed_ptr<L_SHIPMODE>();
        return (sm[i] == 0 || sm[i] == 4) && si[i] == 1;
    }
};

void execute_q19_tpch_all_long(int vec_size, std::string file, std::string algo) {
    constexpr int N_lineitem = 59986052;
    constexpr int N_part = 2000000;

    READ_COL_SHUFFLE(l_partkey, N_lineitem, long, KEY_COL_T, 42)
    READ_COL_SHUFFLE(l_quantity, N_lineitem, long, long, 42)
    READ_COL_SHUFFLE(l_shipmode, N_lineitem, long, long, 42)
    READ_COL_SHUFFLE(l_extendedprice, N_lineitem, long, long, 42)
    READ_COL_SHUFFLE(l_discount, N_lineitem, long, long, 42)
    READ_COL_SHUFFLE(l_shipinstruct, N_lineitem, long, long, 42)
    READ_COL_SHUFFLE(p_partkey, N_part, long, KEY_COL_T, 42)
    READ_COL_SHUFFLE(p_brand, N_part, long, long, 42)
    READ_COL_SHUFFLE(p_container, N_part, long, long, 42)
    READ_COL_SHUFFLE(p_size, N_part, long, long, 42)

    std::cout << "Data is loaded\n";
    
    using LINEITEM_T = struct Chunk<l_partkey_t,l_quantity_t,l_shipmode_t,l_extendedprice_t,l_discount_t,l_shipinstruct_t>;
    using PART_T = struct Chunk<p_partkey_t,p_brand_t,p_container_t,p_size_t>;
    using LINEITEM_T_PROJ_0 = struct Chunk<l_partkey_t,l_quantity_t,l_extendedprice_t,l_discount_t>;
    using PART_LINEITEM_JOIN_0 = struct Chunk<p_partkey_t, p_brand_t, p_container_t, p_size_t, l_quantity_t, l_extendedprice_t,l_discount_t>;

    ScanOperator<LINEITEM_T> op_scan_lineitem(std::make_tuple(l_partkey,l_quantity,l_shipmode,l_extendedprice,l_discount,l_shipinstruct), N_lineitem, vec_size);
    SelectTupleOperator<LINEITEM_T, Pred_0<LINEITEM_T>, decltype(&op_scan_lineitem)> op_select_0(&op_scan_lineitem);
    ProjectOperator<LINEITEM_T, LINEITEM_T_PROJ_0, decltype(&op_select_0), ProjMoveFrom<0>,ProjMoveFrom<1>,ProjMoveFrom<3>,ProjMoveFrom<4>> op_proj_0(&op_select_0);
    
    ScanOperator<PART_T> op_scan_part(std::make_tuple(p_partkey,p_brand,p_container,p_size), N_part, vec_size);
    
    InnerEqJoinOperator<PART_T, LINEITEM_T_PROJ_0, PART_LINEITEM_JOIN_0, decltype(&op_scan_part), decltype(&op_proj_0)> op_join_0(&op_scan_part, &op_proj_0, vec_size, algo, file);
    
    MaterializeOperator<PART_LINEITEM_JOIN_0, decltype(&op_join_0)> op_root(&op_join_0);

    std::vector<Operator*> op_vec {&op_scan_lineitem, &op_scan_part, &op_select_0, &op_proj_0, &op_join_0, &op_root};
    
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
        std::cout << "Usage: ./bin/volcano/tpch_q19 <algo> <profile_output>\n";
        exit(1);
    }
    std::string algo(argv[1]);
    std::string profile_output = argv[2];
    execute_q19_tpch_all_long(59986052, profile_output, algo);
    return 0;
}