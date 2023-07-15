#ifndef COMMON_H
#define COMMON_H 
#include <cuda_runtime.h> // cudaMalloc, cudaMemcpy, etc.
#include <gp/cuda-gpuinfo.h>
#include <gp/ptable.h>
#include <nperf.h>
#include <cublas_v2.h>       // cuSgemm
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <vector>
//#define AXW

struct Metrics {
    float t = 0.0f;
    float spmm_t = 0.0f;
    float gemm_t = 0.0f;
    float flops = 0.0f;
    float spmm_flops = 0.0f;
    float gemm_flops = 0.0f;
    float dataMovement = 0.0f;

    void operator+=(const Metrics& b) {
        t += b.t;
        spmm_t += b.spmm_t;
        gemm_t += b.gemm_t;
        flops = b.flops;
        spmm_flops = b.spmm_flops;
        gemm_flops = b.gemm_flops;
    }

    void operator/=(const float& x) {
        t /= x;
        flops /= x;
    }
};

class Perfs {
public:
    Perfs():cuSpmmSetup(0.0),cuSpmmProcessing(0.0),cuSpmm_time(0.0),cuspmm_throughput(0.0),cuspmm_bandwidth(0.0){}
    float cuSpmmSetup;
    float cuSpmmProcessing;
    float cuSpmm_time;
    float cuspmm_throughput;
    float cuspmm_bandwidth;

    std::vector<float> flex_spmm_time;
    std::vector<float> flex_spmm_throughput;
    std::vector<float> flex_spmm_bandwidth;
    std::vector<int> flex_spmm_errors;
};

#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d) in %s\n",             \
               __LINE__, cudaGetErrorString(status), status, __FILE__);                  \
       /* return EXIT_FAILURE; */                                                  \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        /*return EXIT_FAILURE;  */                                               \
    }                                                                          \
}

template< typename T >
inline void
cuda_freez(T*& ptr_dev)
{
  if ( !ptr_dev ) return;
  CUDA_CHECK( cudaFree( ptr_dev ) );
  ptr_dev = nullptr;
}

template<typename T, typename T2>
inline bool set_max(T& accum, const T2& v)
{
  if ( v <= accum ) return false;
  accum = v;
  return true;
}

template<typename T, typename T2>
inline bool set_min(T& accum, const T2& v)
{
  if ( v >= accum ) return false;
  accum = v;
  return true;
}

#endif /* COMMON_H */
