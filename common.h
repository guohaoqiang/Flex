#ifndef COMMON_H
#define COMMON_H 
//#define TM 4
//#define TN 4
#include <cuda_runtime.h> // cudaMalloc, cudaMemcpy, etc.
#include <cublas_v2.h>       // cuSgemm
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <vector>
struct Metrics {
    float t = 0.0f;
    float spgemm_t = 0.0f;
    float gemm_t = 0.0f;
    float flops = 0.0f;
    float spgemm_flops = 0.0f;
    float gemm_flops = 0.0f;
    float dataMovement = 0.0f;

    void operator+=(const Metrics& b) {
        t += b.t;
        spgemm_t += b.spgemm_t;
        gemm_t += b.gemm_t;
        flops = b.flops;
        spgemm_flops = b.spgemm_flops;
        gemm_flops = b.gemm_flops;
    }

    void operator/=(const float& x) {
        t /= x;
        flops /= x;
    }
};

class Perfs {
public:
    Perfs():cuspgemm_time(0.0),cuspgemm_throughput(0.0),cuspgemm_bandwidth(0.0){}
    float cuspgemm_time;
    float cuspgemm_throughput;
    float cuspgemm_bandwidth;

    std::vector<float> flex_spgemm_time;
    std::vector<float> flex_spgemm_throughput;
    std::vector<float> flex_spgemm_bandwidth;
    std::vector<int> flex_spgemm_errors;
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
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
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
#endif /* COMMON_H */
