#define CUB_STDERR
// #define CHECK_CORRECTNESS
// #define SORTED_REL
#define MR_FILTER_FK

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

//  #define KEY_T_8B
//  #define COL_T_8B

using namespace std;

enum join_algo {
    PHJ,
    SMJ,
    SHJ,
    SMJI,
    UnsupportedJoinAlgo
};

enum Input {
    RelR,
    RelS,
    UniqueKeys
};

char join_algo_name[4][32] = {{"PHJ"}, {"SMJ"}, {"SHJ"}, {"SMJI"}};

#define RUN_CASE(c1, c2, c3) { \
    if(args.pr+1 == c1 && args.ps+1 == c2) { \
        if(args.type == PK_FK || args.type == FK_FK) { \
            run_test_multicols<join_key_t, col_t, TU ## c1, TU ## c2, TU ## c3>(args); \
        } \
    } \
}

struct join_args {
    int nr {4096};
    int ns {4096};
    int pr {1};
    int ps {1};
    int vec_size {8192};
    int unique_keys {4096}; 
    enum join_type type {PK_FK};
    enum dist_type dist {UNIFORM};
    double zipf_factor {1.5};
    int selectivity {1};
    bool agg_only {false};
    std::string output {"join_exp.csv"};
    bool late_materialization {false};
    enum join_algo algo {SMJ};
    int phj_log_part1 {9};
    int phj_log_part2 {6};
    std::string data_path_prefix {"/scratch/wubo/joinmb/"};
#ifndef KEY_T_8B
    int key_bytes {4};
#else
    int key_bytes {8};
#endif
#ifndef COL_T_8B
    int val_bytes {4};
#else
    int val_bytes {8};
#endif

    void print() {
        cout << "||R|| = " << nr << " "
             << "||S|| = " << ns << "\n"
             << "R payload columns = " << pr << " "
             << "S payload columns = " << ps << "\n"
             << "Join algorithm: " << join_algo_name[algo] << "\n"
             << "Join type: " << (type == PK_FK ? "Primary-foreign" : "Unique") << "\n"
             << "(if PK-FK) Distribution type: " << (dist == UNIFORM ? "Uniform" : "Zipf") << "\n"
             << "(if zipf) factor = " << zipf_factor << "\n"
             << "(if PK-FK) Selectivity = " << selectivity << "\n"
             << "(if PHJ) log_part1 = " << phj_log_part1 << " log_part2 = " << phj_log_part2 << "\n"
             << "key_bytes = " << key_bytes << " val_bytes = " << val_bytes << "\n"
             << "Late Materialization only? " << (late_materialization ? "Yes" : "No") << "\n"
             << "Output file: " << output << "\n"
             << "Data path prefix: " << data_path_prefix << "\n"
             << "Aggregation only? " << (agg_only ? "Yes" : "No") << "\n\n";
    }

    void check() {
        assert(pr >= 0 && ps >= 0);
        assert(!output.empty());
        assert(algo < UnsupportedJoinAlgo);
        if(type == FK_FK) assert(unique_keys <= nr && unique_keys <= ns);
    }
};

std::string get_path_name(enum Input table, const struct join_args& args) {
    auto nr = args.nr;
    auto ns = args.ns;
    auto uk = args.unique_keys;

#ifndef KEY_T_8B
    std::string subfolder = "int/";
#else
    std::string subfolder = "long/";
#endif
    
    if(table == UniqueKeys) {
        return args.data_path_prefix+subfolder+"r_" + std::to_string(uk) + ".bin";
    }
    
    if(args.type == PK_FK) {
        return table == RelR ? args.data_path_prefix+subfolder+"r_" + std::to_string(nr) + ".bin"
                             : args.data_path_prefix+subfolder+"s_" + std::to_string(nr) + "_" +std::to_string(ns) + "_" + (args.dist == UNIFORM ? "uniform" : "zipf_") + (args.dist == UNIFORM ? "" : std::to_string(args.zipf_factor))+".bin";
    }
    else {
        return table == RelR ? args.data_path_prefix+subfolder+"s_" + std::to_string(args.unique_keys) + "_" +std::to_string(nr) + "_uniform.bin"
                             : args.data_path_prefix+subfolder+"s_" + std::to_string(args.unique_keys) + "_" +std::to_string(ns) + "_uniform.bin";
    }
}

template<typename T>
void sort_on_gpu(T* keys, int num_items) {
    T* d_keys;
    T* d_sorted_keys;
    cudaMalloc(&d_keys, sizeof(T)*num_items);
    cudaMemcpy(d_keys, keys, sizeof(T)*num_items, cudaMemcpyDefault);
    cudaMalloc(&d_sorted_keys, sizeof(T)*num_items);

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes, d_keys, d_sorted_keys, num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, d_sorted_keys, num_items);

    cudaMemcpy(keys, d_sorted_keys, sizeof(T)*num_items, cudaMemcpyDefault);
    cudaFree(d_keys);
    cudaFree(d_sorted_keys);
    cudaDeviceSynchronize();
}

template<typename join_key_t, typename col_t, typename TupleR, typename TupleS>
void prepare_workload(const struct join_args& args, TupleR& relation_r, TupleS& relation_s) {
    constexpr int R_NUM_COLS = TupleR::num_cols, S_NUM_COLS = TupleS::num_cols;

    auto nr = args.nr;
    auto ns = args.ns;

    join_key_t *rkeys = nullptr, *skeys = nullptr;
    std::array<col_t*, R_NUM_COLS-1> r;
    std::array<col_t*, S_NUM_COLS-1> s;

    std::string rpath = get_path_name(RelR, args);
    std::string spath = get_path_name(RelS, args);
    
    if(args.type == PK_FK) {
        // create relation R
        if(input_exists(rpath)) {
            cout << "R read from disk\n";
            alloc_load_column(rpath, rkeys, nr);
        } else {
            create_integral_relation_unique(&rkeys, nr, false, static_cast<join_key_t>(0), true, 42);
            write_to_disk(rkeys, nr, rpath);
        }

        // create relation S
        if(input_exists(spath)) {
            cout << "S read from disk\n";
            alloc_load_column(spath, skeys, ns);
        } else {
            if(args.dist == UNIFORM) {
                create_fk_from_pk_uniform(&skeys, ns, rkeys, nr);
            }
            else {
                create_fk_from_pk_zipf(&skeys, ns, rkeys, nr, args.zipf_factor);
            }

            write_to_disk(skeys, ns, spath);
        }
    }
    else if(args.type == FK_FK) {
        if(args.dist == ZIPF) {
            std::cout << "FKFK join with zipf distribution is not supported for now\n";
            std::exit(-1);
        }

        join_key_t* uk = nullptr;
        auto nuk = args.unique_keys;
        if(!input_exists(rpath) || !input_exists(spath)) {
            std::string upath = get_path_name(UniqueKeys, args);
            if(input_exists(upath)) {
                cout << "Unique keys read from disk\n";
                alloc_load_column(upath, uk, nuk);
            } else {
                create_integral_relation_unique(&uk, nuk, false, static_cast<join_key_t>(0), true, 42);
                write_to_disk(uk, nuk, upath);
            }
        }

        if(input_exists(rpath)) {
            cout << "R read from disk\n";
            alloc_load_column(rpath, rkeys, nr);
        } else {
            create_fk_from_pk_uniform(&rkeys, nr, uk, nuk);
            write_to_disk(rkeys, nr, rpath);
        }

        // create relation S
        if(input_exists(spath)) {
            cout << "S read from disk\n";
            alloc_load_column(spath, skeys, ns);
        } else {    
            create_fk_from_pk_uniform(&skeys, ns, uk, nuk);
            write_to_disk(skeys, ns, spath);
        }
    }
    else {
        std::cout << "Unsupported join type\n";
        std::exit(-1);
    }

#ifdef MR_FILTER_FK
    if(args.selectivity > 1) {
        if(args.selectivity >= args.nr) assert(false);
        #pragma omp parallel for
        for(int i = 0; i < nr; i++) {
            if(i % args.selectivity == 0) continue;
            rkeys[i] += (1<<30);
        }
    }
    std::cout << "Filtered FK to reduce the match ratio\n";
#endif

#ifdef SORTED_REL
    sort_on_gpu(rkeys, nr);
    sort_on_gpu(skeys, ns);
#endif

    if(sizeof(col_t) == sizeof(join_key_t)) {
        for(int i = 0; i < S_NUM_COLS-1; i++) {
            s[i] = new col_t[ns];
            memcpy(s[i], skeys, sizeof(col_t)*ns);
        }

        for(int i = 0; i < R_NUM_COLS-1; i++) {
            r[i] = new col_t[nr];
            memcpy(r[i], rkeys, sizeof(col_t)*nr);
        }
    } else {
        for(int i = 0; i < S_NUM_COLS-1; i++) {
            s[i] = new col_t[ns];
        }

        for(int i = 0; i < R_NUM_COLS-1; i++) {
            r[i] = new col_t[nr];
        }

        #pragma unroll
        for(int i = 0; i < ns; i++) {
            s[0][i] = static_cast<col_t>(skeys[i]);
        }

        #pragma unroll
        for(int i = 0; i < nr; i++) {
            r[0][i] = static_cast<col_t>(rkeys[i]);
        }

        for(int i = 1; i < S_NUM_COLS-1; i++) {
            memcpy(s[i], s[0], sizeof(col_t)*ns);
        }

        for(int i = 1; i < R_NUM_COLS-1; i++) {
            memcpy(r[i], r[0], sizeof(col_t)*nr);
        }
    }

    cout << "Data preparation is done\n";

    auto b_cols = std::tuple_cat(std::make_tuple(rkeys), std::tuple_cat(r));
    auto p_cols = std::tuple_cat(std::make_tuple(skeys), std::tuple_cat(s));

    ScanOperator<TupleR> op1(std::move(b_cols), nr, nr);
    ScanOperator<TupleS> op2(std::move(p_cols), ns, ns);

    op1.open(); op2.open();
    relation_r = op1.next();
    relation_s = op2.next();
    op1.close(); op2.close();

    // adjust the match ratio
    // if the match ratio is 1 out of M, 
    // then we randomly remove floor(|R|/M) elements from relation R (assuming M < |R|)
    // this is simulating the filtering before join
#ifndef MR_FILTER_FK
    if(args.selectivity > 1) {
        if(args.selectivity >= args.nr) assert(false);
        relation_r.num_items /= args.selectivity;
    }
    cout << "The effective |R| after adjusting the selectivity is " << relation_r.num_items << endl; 
#endif

    release_mem(relation_r.select_vec);
    release_mem(relation_s.select_vec);
    relation_r.select_vec = nullptr;
    relation_s.select_vec = nullptr;

    delete[] rkeys;
    delete[] skeys;
    for(int i = 0; i < R_NUM_COLS-1; i++) {
        delete [] r[i];
    }

    for(int i = 0; i < S_NUM_COLS-1; i++) {
        delete [] s[i];
    }
}

template<typename join_key_t, typename col_t, typename TupleR, typename TupleS, typename Tout>
void check_correctness(const struct join_args& args, TupleR& r, TupleS& s, Tout& t) {
#ifdef CHECK_CORRECTNESS
    cout << "Verifying...\n";

    long long checksum = 0;

    if(args.selectivity > 1) {
        auto ks = new join_key_t[r.num_items];
        cudaMemcpy(ks, COL(r,0), sizeof(join_key_t)*r.num_items, cudaMemcpyDefault);
        for(int i = 0; i < r.num_items; i++) {
            checksum += ks[i];
        }
        checksum *= args.ns/args.nr;
        delete [] ks;
    }
    else {
        auto ks = new join_key_t[s.num_items];
        cudaMemcpy(ks, COL(s,0), sizeof(join_key_t)*s.num_items, cudaMemcpyDefault);
        for(int i = 0; i < s.num_items; i++) {
            checksum += ks[i];
        }
        delete [] ks;

        // extra logic for FKFK join
        if(args.type == FK_FK) {
            // find for each element in S, how many elements in R have the same key
            if(args.dist == ZIPF) {
                std::cout << "FKFK join with zipf distribution is not supported for correctness check for now\n";
                return;
            }

            auto match_partner_per_key = args.nr / args.unique_keys;
            checksum *= match_partner_per_key;
        }
    }
    
    auto keys = new join_key_t[t.num_items];
    cudaMemcpy(keys, COL(t,0), sizeof(join_key_t)*t.num_items, cudaMemcpyDefault);

    long long sum = 0;
    for(int i = 0; i < t.num_items; i++) {
        sum += keys[i];
    }

    if(sum != checksum) {
        cout << "[INCORRECT] Checksum is incorrect, sum = " << sum << " and the difference is " << checksum - sum <<"\n";
        // std::exit(-1);
    }

    auto vals = new col_t[t.num_items];
    for_<t.num_cols-1>([&](auto c) {
        cudaMemcpy(vals, COL(t,c.value+1), sizeof(col_t)*t.num_items, cudaMemcpyDefault);
        for(int i = 0; i < t.num_items; i++) {
            if(static_cast<col_t>(keys[i]) != vals[i]) {
                cout << "[INCORRECT] Unmatched key and value\n";
                cout << "The " << i << "-th value of column " << c.value+1 << " is " << vals[i];
                cout << " but the key is " << keys[i] << endl; 
                std::exit(-1);
            }
        }
    });
    
    cout << "[CORRECT]\n";
    
    delete [] keys;
    delete [] vals;
#endif
}

template<typename TupleR, typename TupleS, typename ResultTuple>
ResultTuple exec_join(TupleR& relation_r, TupleS& relation_s, const struct join_args& args, JoinBase<ResultTuple>*& impl) {
    int circular_buffer_size;
    int first_bit = 0;

#ifdef CHECK_CORRECTNESS
    if(args.type == PK_FK)
        circular_buffer_size = relation_s.num_items;
    else if(args.type == FK_FK)
        circular_buffer_size = relation_s.num_items * (args.nr / args.unique_keys);
#else
    circular_buffer_size = std::max(relation_r.num_items, relation_s.num_items);
#endif

    std::cout << "Circular buffer size = " << circular_buffer_size << "\n";
    if(args.algo == SMJ || (args.algo == SMJI && args.pr == 1 && args.ps == 1)) {
        impl = new SortMergeJoin<TupleR, TupleS, ResultTuple, true>(relation_r, relation_s, circular_buffer_size);
    } else if(args.algo == PHJ) {
        impl = new PartitionHashJoin<TupleR, TupleS, ResultTuple>(relation_r, relation_s, args.phj_log_part1, args.phj_log_part2, first_bit, circular_buffer_size);
    } else if(args.algo == SHJ) {
        impl = new SortHashJoin<TupleR, TupleS, ResultTuple>(relation_r, relation_s, first_bit, args.phj_log_part1+args.phj_log_part2, circular_buffer_size);
    } else if(args.algo == SMJI) {
        impl = new SortMergeJoinByIndex<TupleR, TupleS, ResultTuple, false>(relation_r, relation_s, circular_buffer_size);
    } 
    else {
        std::cout << "Unsupported join algorithm\n";
        std::exit(-1);
    }

    return impl->join();
}

template<typename JoinImpl>
void exp_stats(JoinImpl* impl, const struct join_args& args) {
    cout << "\n==== Statistics ==== \n";
    impl->print_stats();
    cout << "Peak memory used: " << mm->get_peak_mem_used() << " bytes\n";

    if(!args.output.empty()) {
        ofstream fout;
        fout.open(args.output, ios::app);
        fout << get_utc_time() << ","
             << args.nr << "," << args.ns << ","
             << args.pr << "," << args.ps << ","
             << join_algo_name[args.algo] << ","
             << (args.type == PK_FK ? "pk_fk," : "fk_fk,")
             << args.unique_keys << ","
             << (args.dist == UNIFORM ? "uniform," : "zipf,")
             << args.zipf_factor << ","
             << args.selectivity << ","
             << (args.agg_only ? "aggregation," : "materialization,")
             << args.phj_log_part1 << "," << args.phj_log_part2 << ","
             << args.key_bytes << "," << args.val_bytes << ",";

        auto stats = impl->all_stats();
        for(auto t : stats) {
            fout << t << ",";
        }

        fout << endl;
        fout.close();
    }
}

template<typename join_key_t, typename col_t, typename TupleR, typename TupleS, typename ResultTuple>
void run_test_multicols(const struct join_args& args) {
    TupleR relation_r;
    TupleS relation_s;

    prepare_workload<join_key_t, col_t>(args, relation_r, relation_s);

    JoinBase<ResultTuple>* impl;
    auto out = exec_join(relation_r, relation_s, args, impl);
    
    cudaDeviceSynchronize();

    cout << "\nOutput Cardinality = " << out.num_items << endl;
    cout << "Results (first 10 items): \n";
    out.peek(args.agg_only ? 1 : min(10, out.num_items));

    check_correctness<join_key_t, col_t>(args, relation_r, relation_s, out);

    exp_stats(impl, args);

    relation_r.free_mem();
    relation_s.free_mem();
    out.free_mem();
}

void print_usage() {
    cout << "Join Microbenchmarks\n";
    cout << "Usage: <binary> [-l|-h] -r <log_2(|R|)> -s <log_2(|S|)> -m <R payload cols> -n <S payload cols> -t <join type> -d <distribution> -z <zipf factor> -o <output file> -f <data path prefix> -e <selectivity> -u <unique keys> -i <join algorithm> -p <phj log part1> -q <phj log part2>\n";
    cout << "Options:\n";
    cout << "-l: use log scale for |R|, |S|, and unique keys. Default: no.\n";
    cout << "-h: print this message\n";
    cout << "-r: log_2(|R|) if using -l flag otherwise the actual size\n";
    cout << "-s: log_2(|S|) if using -l flag otherwise the actual size\n";
    cout << "-m: number of payload columns in R\n";
    cout << "-n: number of payload columns in S\n";
    cout << "-t: join type, pkfk or fkfk. Default: pkfk\n";
    cout << "-d: distribution type, uniform or zipf. Default: uniform\n";
    cout << "-z: zipf factor, only valid when -d zipf is used\n";
    cout << "-o: output file name\n";
    cout << "-f: path to the generated data directory if any; otherwise provide a location where you want the generated data to be stored\n";
    cout << "-e: selectivity, only valid when -t pkfk is used. Default: 1.\n";
    cout << "-u: number of unique keys, only valid when -t fkfk is used\n";
    cout << "-i: join algorithm, phj, shj, smj, smji (case sensitive)\n";
    cout << "-p: log_2(partitions in 1st pass) for PHJ. Default: 9.\n";
    cout << "-q: log_2(partitions in 2nd pass) for PHJ. Default: 6.\n";
    cout << "(Note: -p and -q are only valid when -i phj or -i shj is used)\n";
    cout << "Example: ./bin/volcano/join_exp -l -r 12 -s 12 -m 1 -n 1 -t pkfk -d uniform -o join_exp.csv -f /home/data/ -e 1 -i phj -p 9 -q 6\n";
}

void parse_args(int argc, char** argv, struct join_args& args) {
    bool use_log_scale = false;
    for(;;)
    {
      switch(getopt(argc, argv, "r:s:v:m:n:t:d:z:o:e:u:i:p:q:f:alh"))
      {
        case 'r':
            args.nr = atoi(optarg);
            continue;
        case 's':
            args.ns = atoi(optarg);
            continue;
        case 'v':
            args.vec_size = atoi(optarg);
            continue;
        case 'm':
            args.pr = atoi(optarg);
            continue;
        case 'n':
            args.ps = atoi(optarg);
            continue;
        case 't':
            if(strcasecmp(optarg, "fkfk") == 0) {
                args.type = FK_FK;
            }
            continue;
        case 'd':
            if(strcasecmp(optarg, "zipf") == 0) {
                args.dist = ZIPF;
            }
            else {
                args.zipf_factor = 0.0f;
            }
            continue;
        case 'z':
            args.zipf_factor = atof(optarg);
            continue;
        case 'o':
            args.output = std::string(optarg);
            continue;
        case 'f':
            args.data_path_prefix = std::string(optarg);
            if(args.data_path_prefix.back() != '/') args.data_path_prefix += "/";
            continue;
        case 'e':
            args.selectivity = atoi(optarg);
            continue;
        case 'u':
            args.unique_keys = atoi(optarg);
            continue;
        case 'a':
            args.agg_only = true;
            continue;
        case 'l':
            use_log_scale = true;
            continue;
        case 'i':
            if(std::string(optarg) == "phj") args.algo = PHJ;
            else if(std::string(optarg) == "shj") args.algo = SHJ;
            else if(std::string(optarg) == "smji") args.algo = SMJI;
            else args.algo = SMJ;
            continue;
        case 'p':
            args.phj_log_part1 = atoi(optarg);
            continue;
        case 'q': 
            args.phj_log_part2 = atoi(optarg);
            continue;
        case 'h':
            print_usage();
            exit(0);

        default :
          printf("[Invalid Input]\n Use -h for help\n");
          break;

        case -1:
          break;
      }

      break;
    }

    if(use_log_scale) {
        args.nr = (1 << args.nr);
        args.ns = (1 << args.ns);
        args.unique_keys = (1 << args.unique_keys);
    }

    args.check();
    args.print();
}

int main(int argc, char** argv) {
#ifndef COL_T_8B
    using col_t = int;
#else
    using col_t = long;
#endif

#ifndef KEY_T_8B
    using join_key_t = int;
#else
    using join_key_t = long;
#endif

    DECL_TUP_1_TO_8(join_key_t, col_t)

    struct join_args args;
    parse_args(argc, argv, args);

    // increasing number of payloads in FK table
    RUN_CASE(2, 3, 4);
    RUN_CASE(2, 4, 5);
    RUN_CASE(2, 5, 6);
    RUN_CASE(2, 6, 7);
    RUN_CASE(2, 7, 8);

    // increasing number of payloads in FK table
    RUN_CASE(3, 2, 4);
    RUN_CASE(4, 2, 5);
    RUN_CASE(5, 2, 6);
    RUN_CASE(6, 2, 7);
    RUN_CASE(7, 2, 8);

    // both sides have payload columns to materialize
    RUN_CASE(2, 2, 3);
    RUN_CASE(3, 3, 5);
    RUN_CASE(4, 4, 7);
    RUN_CASE(5, 5, 9);
    RUN_CASE(6, 6, 11);
    RUN_CASE(7, 7, 13);

    
    return 0;
}
