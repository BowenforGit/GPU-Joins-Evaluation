#pragma once
#define CUB_STDERR

#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <unistd.h>
#include <fstream>
#include <ctime>

#include <cuda.h>

#include "tuple.cuh"
#include "utils.cuh"
#include "../data_gen/generator.cuh"

#define DECL_TUP(c, ...) using TU ## c = struct Chunk<__VA_ARGS__>;

enum join_type {
    PK_FK,
    FK_FK,
    NOT_SUPPORTED_JOIN_TYPE
};

enum dist_type {
    UNIFORM,
    ZIPF
};

#define repeat_2(x) x,x
#define repeat_4(x) repeat_2(x),repeat_2(x)
#define repeat_8(x) repeat_4(x),repeat_4(x)

#define DECL_TUP_1_TO_8(join_key_t, join_val_t)  \
    DECL_TUP(1, join_key_t) \
    DECL_TUP(2, join_key_t, join_val_t) \
    DECL_TUP(3, join_key_t, join_val_t, join_val_t) \
    DECL_TUP(4, join_key_t, join_val_t, join_val_t, join_val_t) \
    DECL_TUP(5, join_key_t, join_val_t, join_val_t, join_val_t, join_val_t) \
    DECL_TUP(6, join_key_t, join_val_t, join_val_t, join_val_t, join_val_t, join_val_t) \
    DECL_TUP(7, join_key_t, join_val_t, join_val_t, join_val_t, join_val_t, join_val_t, join_val_t) \
    DECL_TUP(8, join_key_t, join_val_t, join_val_t, join_val_t, join_val_t, join_val_t, join_val_t, join_val_t) \
    DECL_TUP(9, join_key_t,  repeat_8(join_val_t)) \
    DECL_TUP(10, join_key_t, repeat_8(join_val_t), join_val_t) \
    DECL_TUP(11, join_key_t, repeat_8(join_val_t), repeat_2(join_val_t)) \
    DECL_TUP(12, join_key_t, repeat_8(join_val_t), repeat_2(join_val_t), join_val_t) \
    DECL_TUP(13, join_key_t, repeat_8(join_val_t), repeat_4(join_val_t)) \

template<class T>
void load_column(const std::string& file_name, T* dst, const int N) {
    std::ifstream file(file_name);
    file.read(reinterpret_cast<char*>(dst), N*sizeof(T));
    file.close();
}

template<class T>
void alloc_load_column(const std::string& file_name, T*& dst, const int N) {
    dst = new T[N];
    load_column(file_name, dst, N);
}

template<typename T>
void write_binary_file(const T* arr, const size_t N, const std::string& path) {
    std::ofstream f(path, std::ofstream::binary);
    f.write(reinterpret_cast<const char*>(arr), sizeof(T) * N);
}

template<typename T, size_t kThreshold = 134217728>
void write_to_disk(const T* arr, const size_t N, const std::string& path) {
    if(sizeof(T)*N >= kThreshold) {
        write_binary_file(arr, N, path);
    }
}

inline bool input_exists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

inline std::string get_utc_time()
{
    // Example of the very popular RFC 3339 format UTC time
    std::time_t time = std::time({});
    char timeString[std::size("yyyy-mm-ddThh:mm:ssZ")];
    std::strftime(std::data(timeString), std::size(timeString),
                  "%FT%TZ", std::localtime(&time));
    return std::string(timeString);
}