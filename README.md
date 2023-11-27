# Performance Study of GPU-based Joins


This reposistory contains implementations of four GPU-based join implementations and code to evaluate them.
The implementations contain
* SMJI - State-of-the-art sort merge join using radix sort and Merge Path. "SMJ-UM" in the paper.
* SMJ - Improved SMJI using the GFTR technique. "SMJ-OM" in the paper.
* PHJ - State-of-the-art partitioned hash join implementation from [Sioulas et al.](https://ieeexplore.ieee.org/document/8731612). "PHJ-UM" in the paper.
* SHJ - Improved PHJ using the GFTR pattern with a redesigned partitioning strategy. "PHJ-OM" in the paper.

## Dependencies
```
python
cuda == 12.2 or 12.3 # These are the CUDA versions the code is developed under
```

## Usage

<!-- ### Install python packages
In the project home directory,
```
python -m pip install -r requirements.txt
``` -->

### Configure the project

In the `Makefile`, specify the compute capability of your GPU. We have tested the code on RTX 3090 (8.6) and A100 (8.0).

Then in the project home directory, run

```
sh configure.sh
```

This will compile all available executables to `bin/volcano/`.

There are four executables.

* `bin/volcano/join_exp_4b4b`: 4-byte keys and 4-byte non-key attributes for microbenchmarks in Section 5.2.1 - 5.2.7.
* `bin/volcano/join_exp_4b8b`: 4-byte keys and 8-byte non-key attributes for microbenchmarks in Section 5.2.1 - 5.2.7.
* `bin/volcano/join_exp_8b8b`: 8-byte keys and 8-byte non-key attributes for microbenchmarks in Section 5.2.1 - 5.2.7.
* `bin/volcano/join_pipeline`: he sequence of joins discussed in Section 5.2.8.

Because the microbenchmark will generate input data and the generation is slow, 
it is more efficient to cache the generated data on disk. 
Suppose you want to store the input data in `<path_to_input_data>` then

```
mkdir -p <path_to_input_data>/int/
mkdir -p <path_to_input_data>/long/
```

<!-- ### Compile the code

There are two executable whose source files are `src/volcano/join_exp.cu` and `src/volcano/join_pipeline.cu`. The former one is the microbenchmarks covering Section 5.2.1 - 5.2.7; the latter one covers the sequence of joins discussed in Section 5.2.8.

To compile, run in the project home directory

```
make bin/volcano/join_exp
make bin/volcano/join_pipeline
``` -->

### Run the microbenchmarks
You can directly invoke the two executables and pass in the configurations. You can check the instructions by only passing the `-h` flag to the executable.

Alternatively, we have prepared a script to run all the microbenchmarks in `run.sh`. Simply run in the project home directory
```
sh run.sh <repeat times> <path_to_input_data>
```

## Caveat
