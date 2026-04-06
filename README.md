# Flex

Sparse-dense matrix multiplication (SpMM) on GPUs.

This repository contains **Flex** (our SpMM kernel) along with **standalone benchmarks** for three state-of-the-art TC-based SpMM baselines, each compiled as a single self-contained CUDA file with cuSPARSE as the accuracy/performance reference.

## Baselines

| Baseline | Source File | Paper / Venue |
|----------|------------|---------------|
| **DTC-SpMM** | `dtc_spmm.cu` | [DTC-SpMM (ASPLOS '24)](https://github.com/HPMLL/DTC-SpMM_ASPLOS24) |
| **FlashSparse** | `flashsparse_spmm.cu` | [FlashSparse (PPoPP '25)](https://github.com/ParCIS/FlashSparse) |
| **Voltrix-SpMM** | `voltrix_spmm.cu` | [Voltrix-SpMM (ATC '25)](https://github.com/YaqiXia/Voltrix-SpMM) |
| **Flex** | `flex.cu` / `flex_spmm.cu` | Ours (CUDA cores) |

Each baseline `.cu` file is fully self-contained (all kernel code inlined, no external dependencies beyond CUDA and cuSPARSE). All benchmarks use **actual edge weights** from the input CSR and compare against cuSPARSE for correctness (RMSE).

## Requirements

- NVIDIA GPU with Tensor Core support
  - DTC-SpMM and FlashSparse: Ampere or later (`-arch native`)
  - Voltrix-SpMM: **Hopper only** (`-arch=sm_90`, requires TMA / mbarrier)
- CUDA Toolkit (≥ 11.8 recommended)
- cuSPARSE (included with CUDA)
- C++17 compiler

## Datasets

CSR-format CSV files in `data/`. Each file contains three lines: `row_ptr`, `col_idx`, `vals`.

| Dataset | Nodes | Edges |
|---------|------:|------:|
| pubmed | 19,717 | 108,365 |
| flickr | 89,250 | 989,006 |
| ppi | 14,755 | 458,973 |
| reddit | 232,965 | 23,446,803 |
| amazon | 1,569,960 | 264,339,468 |
| yelp | 716,847 | 13,954,819 |
| wiki-Vote | 8,297 | 103,689 |
| web-NotreDame | 325,729 | 1,497,134 |
| soc-sign-epinions | 131,828 | 841,372 |

## Compile

```bash
# Flex (main project)
./compile.sh

# Baselines (each produces a standalone binary)
./compile_dtc.sh            # -> ./dtc_spmm
./compile_flashsparse.sh    # -> ./flashsparse_spmm
./compile_voltrix.sh        # -> ./voltrix_spmm  (Hopper only)
```

## Run

```bash
# Flex
./run.sh

# Baselines  (default K=64, override via argument)
./run_dtc.sh            # DTC-SpMM on pubmed + flickr
./run_flashsparse.sh    # FlashSparse on pubmed + flickr
./run_voltrix.sh        # Voltrix-SpMM on pubmed + flickr
./run_voltrix.sh 128    # override K=128
```

Each baseline can also be invoked directly with multiple datasets:

```bash
./dtc_spmm ./data/pubmed.csv 64 ./data/flickr.csv ./data/reddit.csv
./flashsparse_spmm ./data/pubmed.csv 64 ./data/flickr.csv
./voltrix_spmm ./data/pubmed.csv 64 ./data/flickr.csv
```

### Output Format

All baselines print a unified table:

```
Dataset          Nodes      Edges    K |   cuSP/ms   cuSP/GF |   Kern/ms   Kern/GF  Kern/RMSE |   Prep/ms
-------------- -------- ---------- ---- | --------- --------- | --------- --------- ---------- | ---------
pubmed            19717     108365   64 |    0.0394    351.97 |    0.0280    495.90   6.26e-05 |      1.23
flickr            89250     989006   64 |    0.1341    944.10 |    0.1185   1067.94   4.59e-05 |      3.45
```

- **cuSP/ms**, **cuSP/GF**: cuSPARSE time and throughput (GFlop/s)
- **Kern/ms**, **Kern/GF**: Baseline kernel time and throughput
- **Kern/RMSE**: Root mean square error vs cuSPARSE output
- **Prep/ms**: Preprocessing time (where applicable)

## Vertex Ordering

The code for DEG, RCM and Gorder was taken from [here](https://github.com/lecfab/rescience-gorder).

## ASpT Results

See `aspt/` for the [ASpT](http://gitlab.hpcrl.cse.ohio-state.edu/chong/ppopp19_ae) integration.

| ASpT(k=128) |    3090    |        |          |    4090    |         |          |    H100    |         |          |
|:-----------:|:----------:|:------:|:--------:|:----------:|:-------:|:--------:|:----------:|:-------:|:--------:|
|             | tPre/tElap | GFlops | Errs (%) | tPre/tElap |  GFlops | Errs (%) | tPre/tElap |  GFlops | Errs (%) |
|    Pubmed   |    92.5    | 311.39 |   0.005  |    18.7    |  639.8  |   0.005  |    90.13   |  275.83 |   0.005  |
|    Flickr   |    2.84    |  499.5 |   0.001  |     4.5    |  1308.2 |  0.0011  |    6.18    | 1037.78 |  0.0011  |
|    Reddit   |    1.149   | 259.35 |  97.071  |    3.97    |  1100.8 |  97.077  |    2.58    | 1237.25 |  99.028  |
|     PPI     |    7.77    | 671.01 |  0.0067  |    8.17    |  1182.9 |  0.0067  |    9.53    |  811.62 |  0.0067  |
|    Amazon   |   21.415   | 284.49 |   90.25  |   329.07   | 1150.06 |   90.25  |    522.4   | 1314.59 |   98.34  |
|     Yelp    |  0.212705  | 470.03 |  0.00014 |     0.3    | 1135.66 |  0.0001  |    0.44    | 1451.86 |  0.0001  |


| ASpT(k=32) |    3090    |         |          |    4090    |         |          |    H100    |         |          |
|:----------:|:----------:|:-------:|:--------:|:----------:|:-------:|:--------:|:----------:|:-------:|:--------:|
|            | tPre/tElap |  GFlops | Errs (%) | tPre/tElap |  GFlops | Errs (%) | tPre/tElap |  GFlops | Errs (%) |
|   Pubmed   |   109.75   |  105.67 |   0.005  |   177.32   |  167.88 |   0.005  |    84.9    |   70.8  |   0.005  |
|   Flickr   |    7.83    |  391.22 |   0.001  |    12.58   |  977.28 |  0.0011  |    11.15   | 485.998 |  0.0011  |
|   Reddit   |    0.514   |  557.83 |   0.001  |    1.28    | 1982.98 |  0.00098 |    1.28    |  1819.7 |  0.00098 |
|     PPI    |    12.67   |  311.59 |  0.0067  |   13.625   |  504.92 |  0.0067  |    9.63    |  324.82 |  0.0067  |
|   Amazon   |   472.63   | 316.668 |   98.47  |      *     |    *    |     *    |   1427.16  |  1056.2 |   98.34  |
|    Yelp    |    0.762   |  479.74 |  0.0001  |    1.318   | 1256.62 |  0.00014 |    1.232   |  1044.4 |  0.0001  |


