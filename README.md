# Performance Study of GPU-based Joins


This reposistory contains implementations of four GPU-based join implementations and code to evaluate them.
The implementations contain
* SMJI - State-of-the-art sort merge join using radix sort and Merge Path. "SMJ-UM" in the paper.
* SMJ - Improved SMJI using the GFTR technique. "SMJ-OM" in the paper.
* PHJ - State-of-the-art partitioned hash join implementation from [Sioulas et al.](https://ieeexplore.ieee.org/document/8731612). "PHJ-UM" in the paper.
* SHJ - Improved PHJ using the GFTR pattern with a redesigned partitioning strategy. "PHJ-OM" in the paper.

## Dependencies

Our code is developed and tested under the following environment.

* CUDA 12.2 or 12.3 (including nvcc, cub, thrust and so on)
* GCC 10.2.1 or 11.4.0 (10.2.1 for the A100 machine, 11.4.0 for the RTX 3090 machine)

## Usage

### Configure the project

There are two places in the code base that need to be customized to your machine.

1. In the `Makefile`, specify the compute capability of your GPU. We have tested the code on RTX 3090 (8.6) and A100 (8.0).
2. In `src/volcano/utils.cuh`, change the `mem_pool_size` variable to your own GPU's memory capacity.
3. In `src/volcano/tpc_utils.hpp`, change the `TPC_DATA_PREFIX` (absolute path) to the directory where the TPC-H and TPC-DS data are stored. See more in "Run the TPC-H/DS benchmarks".

Then in the project home directory, run

```
sh configure.sh
```

This will compile all available executables to `bin/volcano/`, including

* `bin/volcano/join_exp_4b4b`: 4-byte keys + 4-byte non-keys for Section 5.2.1 - 5.2.6.
* `bin/volcano/join_exp_4b8b`: 4-byte keys + 8-byte non-keys for Section 5.2.5.
* `bin/volcano/join_exp_8b8b`: 8-byte keys + 8-byte non-keys for Section 5.2.5.
* `bin/volcano/join_pipeline`: sequence of joins for Section 5.2.7.
* `bin/volcano/tpch_[7,18,19]`: Joins extracted from TPC-H Q7, 18, 19 for Section 5.3.
* `bin/volcano/tpcds_[64,95]`: Joins extracted from TPC-DS Q64, 95 for Section 5.3.

Because the microbenchmark will generate input data and the data generation is slow, 
it is more efficient to cache the generated data on disk. 
Suppose you want to store the input data in `<path_to_input_data>`, then run

```
mkdir -p <path_to_input_data>/int/ <path_to_input_data>/long/
```

### Run the microbenchmarks
You can directly invoke the four executables and pass in the configurations. You can check the instructions by passing the `-h` flag to each executable.

Alternatively, we have prepared a script to run all the microbenchmarks in `run.sh`. Simply run the following command in the project home directory.

```
sh run.sh <repeat times> <path_to_input_data>
```

The results will be written to the `exp_results/gpu_join/` directory, and each microbenchmark is stored as a separate CSV file.

### Run the TPC-H/DS benchmarks
Running TPC-H/DS requires input data to be generated from the data generators, i.e., dbgen and dsdgen. For ease of use, we have uploaded all relevant data to the [polybox](https://polybox.ethz.ch/index.php/s/TveX7M7k0LYomkA). Download the `vldb24_tpch_tpcds.tar.xz` to your machine and then decompress and extract the data.

```
xz -d -v vldb24_tpch_tpcds.tar.xz
tar -xvf file.tar -C .
```

In the end, make sure your `TPC_DATA_PREFIX` directory have the following structure.

* `TPC_DATA_PREFIX`
    * `tpch_sf10`
    * `tpcds_sf100`
        * `q64`
        * `q95`

## Notes
1. It is recommended to turn on the persistence mode to reduce the program launch time. See [this guide](https://docs.nvidia.com/deploy/driver-persistence/index.html). You can check if it is turned out via `nvidia-smi`.
