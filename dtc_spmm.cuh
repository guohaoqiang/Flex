///////////////////////////////////////////////////////////////////////////////
// DTC-SpMM — Standalone CUDA baseline (no PyTorch, no Flex dependency)
// Extracted from: https://github.com/HPMLL/DTC-SpMM_ASPLOS24
///////////////////////////////////////////////////////////////////////////////

#ifndef DTC_SPMM_CUH
#define DTC_SPMM_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// DTC-SpMM constants
#define BLK_H 16
#define BLK_W 8
#define WARP_SIZE 32
#define TCBLOCK_PER_WARP 64
#define DTC_EXE_TIME 100

#endif // DTC_SPMM_CUH
