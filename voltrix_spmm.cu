/*
 * voltrix_spmm.cu  –  Standalone Voltrix-SpMM benchmark
 * Extracted from https://github.com/YaqiXia/Voltrix-SpMM  (USENIX ATC '25)
 * Requires NVIDIA Hopper GPU (compute capability >= 9.0)
 *
 * Compile:
 *   nvcc -O3 -std=c++17 -arch=sm_90 -Xcompiler=-fopenmp \
 *        -o voltrix_spmm voltrix_spmm.cu -lcusparse
 *
 * Run:
 *   ./voltrix_spmm ./data/pubmed.csv 128
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <mma.h>
#include <stdint.h>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

/* ================================================================
 *  Section 1 : Constants
 * ================================================================ */

#define VX_BLK_H 16
#define VX_BLK_W 8
#define VX_WARP_SIZE 32
#define VX_WPB 4

static constexpr int BLK_H = VX_BLK_H;
static constexpr int BLK_W = VX_BLK_W;
static constexpr int WARP_SIZE = VX_WARP_SIZE;
static constexpr int WPB = VX_WPB;

#define HOST __forceinline__ __host__
#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__

/* ================================================================
 *  Section 2 : PersistenKernelTraits / Config  (from traits.h)
 * ================================================================ */

template <typename T, int32_t NUM_AGENTS, int32_t NUM_BUFFERS,
          int32_t MAX_MMAS_PER_WARP, int32_t CONSUMER_WARPS_PER_BLOCK,
          int32_t PADDING_SIZE = 0>
struct PersistenKernelTraits {
    static_assert(NUM_AGENTS == 2, "NUM_AGENTS should be 2");
    static constexpr int32_t NUM_BARRIERS_PER_BUFFER = 2;
    static constexpr int32_t NUM_BARRIERS =
        NUM_BUFFERS * NUM_BARRIERS_PER_BUFFER * (NUM_AGENTS - 1);
    static constexpr int32_t BYTES_PER_SCALAR = sizeof(T);
    static constexpr int32_t DENSE_X_STRIDE_PER_MMA =
        CONSUMER_WARPS_PER_BLOCK * BLK_H;
    static constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK =
        MAX_MMAS_PER_WARP * CONSUMER_WARPS_PER_BLOCK * BLK_H;
    static constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK_PADDED =
        MAX_MMAS_PER_WARP * CONSUMER_WARPS_PER_BLOCK * BLK_H + PADDING_SIZE;
    static constexpr int32_t NUM_DENSE_X_SHARED_PER_BUFFER =
        MAX_FEATURE_DIM_PER_BLOCK * BLK_W;
    static constexpr int32_t NUM_DENSE_X_SHARED_PER_BUFFER_PADDED =
        MAX_FEATURE_DIM_PER_BLOCK * BLK_W + PADDING_SIZE * BLK_W;
    static_assert(MAX_FEATURE_DIM_PER_BLOCK % BLK_H == 0, "");
    static_assert(MAX_FEATURE_DIM_PER_BLOCK / BLK_H % CONSUMER_WARPS_PER_BLOCK == 0, "");
    static constexpr int32_t MaxMMAsPerWarp =
        MAX_FEATURE_DIM_PER_BLOCK / BLK_H / CONSUMER_WARPS_PER_BLOCK;
    static constexpr int32_t PRODUCER_WARPS_PER_BLOCK = 1;
    static constexpr int32_t WARPS_PER_BLOCK =
        CONSUMER_WARPS_PER_BLOCK + PRODUCER_WARPS_PER_BLOCK;
    static constexpr int32_t THREADS_PER_WARP = 32;
    static constexpr int32_t THREADS_PER_BLOCK =
        WARPS_PER_BLOCK * THREADS_PER_WARP;
    static constexpr int32_t THREADS_PER_PRODUCER =
        PRODUCER_WARPS_PER_BLOCK * THREADS_PER_WARP;
    static constexpr int32_t THREADS_PER_CONSUMER =
        CONSUMER_WARPS_PER_BLOCK * THREADS_PER_WARP;
};

template <typename T, int32_t NUM_AGENTS, int32_t NUM_BUFFERS,
          int32_t MAX_MMAS_PER_WARP, int32_t CONSUMER_WARPS_PER_BLOCK,
          int32_t GLOBAL_NUM_BLOCKS, int32_t PADDING_SIZE = 0>
struct PersistenKernelConfig {
    using Traits = PersistenKernelTraits<T, NUM_AGENTS, NUM_BUFFERS,
                                         MAX_MMAS_PER_WARP,
                                         CONSUMER_WARPS_PER_BLOCK,
                                         PADDING_SIZE>;
    static constexpr dim3 GRID = dim3(GLOBAL_NUM_BLOCKS, 1, 1);
    static constexpr dim3 BLOCK =
        dim3(Traits::THREADS_PER_WARP, Traits::WARPS_PER_BLOCK, 1);
    static constexpr int32_t DENSE_X_SHARED_MEMORY_SIZE =
        Traits::NUM_DENSE_X_SHARED_PER_BUFFER_PADDED * NUM_BUFFERS *
        Traits::BYTES_PER_SCALAR;
};

/* ================================================================
 *  Section 3 : Device helper functions
 * ================================================================ */

template <typename T>
constexpr DEVICE __host__ T vx_div_round_up(T a, T b) {
    return (a + b - 1) / b;
}

DEVICE uint32_t cast_smem_ptr_to_uint(void const *const ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

struct alignas(16) Uint4 {
    uint32_t vs[4];
    __device__ __forceinline__ uint32_t &operator[](int i) { return vs[i]; }
};

/* --- MMA wrappers ------------------------------------------------------- */

template <int32_t M, int32_t N, int32_t K> struct MMA {};

template <> struct MMA<16, 8, 8> {
    template <typename AT, typename BT, typename CT>
    static void DEVICE mma(AT *a_frag, BT *b_frag, CT *acc_frag) {
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+f"(acc_frag[0]), "+f"(acc_frag[1]),
              "+f"(acc_frag[2]), "+f"(acc_frag[3])
            : "r"(a_frag[0]), "r"(a_frag[1]),
              "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1]));
    }
};

/* --- mbarrier helpers (Hopper raw PTX) ---------------------------------- */

DEVICE void vx_memcpy_async(float *dst, const float *src, int32_t size,
                            uint64_t &barrier) {
    uint32_t dst32 = cast_smem_ptr_to_uint(dst);
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile(
        "{\n\t"
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];\n\t"
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%3], %2;\n\t"
        "}" ::"r"(dst32),
        "l"(src), "r"(size), "r"(smem_addr));
}

DEVICE void mbarrier_init(uint64_t &barrier, int32_t count) {
    auto smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile("{\n\t"
                 "mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 "\t}" ::"r"(smem_addr),
                 "r"(count));
}

DEVICE void mbarrier_arrive(uint64_t &barrier) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile("{\n\t"
                 "mbarrier.arrive.shared::cta.b64 _, [%0];\n"
                 "\t}"
                 :
                 : "r"(smem_addr));
}

DEVICE void mbarrier_arrive_and_wait(uint64_t &barrier) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile("{\n\t"
                 ".reg .b64 phase;\n\t"
                 ".reg .pred p;\n\t"
                 "mbarrier.arrive.shared::cta.b64 phase, [%0];\n\t"
                 "LAB_WAIT: \n\t"
                 "mbarrier.try_wait.shared.b64 p, [%0], phase; \n\t"
                 "@p bra.uni DONE; \n\t"
                 "bra.uni     LAB_WAIT; \n\t"
                 "DONE: \n\t"
                 "}"
                 :
                 : "r"(smem_addr));
}

DEVICE void invalidate(uint64_t &barrier) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier);
    asm volatile("{\n\t"
                 "mbarrier.inval.shared.b64 [%0]; \n\t"
                 "}"
                 :
                 : "r"(smem_addr));
}

DEVICE void cp_async_mbarrier_arrive(uint64_t &barrier) {
    uint32_t smem_ptr = cast_smem_ptr_to_uint(&barrier);
    asm volatile("{\n\t"
                 "cp.async.mbarrier.arrive.shared.b64 [%0];\n"
                 "\t}" ::"r"(smem_ptr));
}

/* ================================================================
 *  Section 4 : SpMM kernels  (from spmm_kernels.cuh)
 * ================================================================ */

/* --- spmm_mma161616_spa_swizzle_d  (model 0 & 1, 1D grid) ------------- */
template <int32_t NUM_AGENTS, int32_t NUM_BUFFERS,
          int32_t MAX_MMAS_PER_WARP,
          int32_t CONSUMER_WARPS_PER_BLOCK, int32_t PADDING_SIZE = 0>
__global__ void
spmm_mma161616_spa_swizzle_d(
    const int *__restrict__ blks_offsets,
    const uint32_t *__restrict__ hspa_packed,
    const float *__restrict__ hspa_float,
    const int32_t *__restrict__ hind,
    const int num_nodes, const int num_edges,
    const int embedding_dim,
    const float *__restrict__ input,
    float *__restrict__ output) {

    using Traits = PersistenKernelTraits<float, NUM_AGENTS, NUM_BUFFERS,
                                         MAX_MMAS_PER_WARP,
                                         CONSUMER_WARPS_PER_BLOCK,
                                         PADDING_SIZE>;

    constexpr int32_t NUM_BARRIERS        = Traits::NUM_BARRIERS;
    constexpr int32_t THREADS_PER_WARP    = Traits::THREADS_PER_WARP;
    constexpr int32_t THREADS_PER_BLOCK   = Traits::THREADS_PER_BLOCK;
    constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK =
        Traits::MAX_FEATURE_DIM_PER_BLOCK;
    constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK_PADDED =
        Traits::MAX_FEATURE_DIM_PER_BLOCK_PADDED;
    constexpr int32_t NUM_DENSE_X_SHARED_PER_BUFFER_PADDED =
        2 * Traits::NUM_DENSE_X_SHARED_PER_BUFFER_PADDED;
    constexpr int32_t DENSE_X_STRIDE_PER_MMA =
        Traits::DENSE_X_STRIDE_PER_MMA;

    const int32_t bid    = blockIdx.x;
    const int32_t wid    = threadIdx.y;
    const int32_t laneid = threadIdx.x;
    const int32_t tid    = threadIdx.y * THREADS_PER_WARP + laneid;

    alignas(16) extern __shared__ float dense_X[];
    alignas(16) __shared__ uint32_t sparse_A[NUM_BUFFERS * 4 * 2];

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t bars[NUM_BARRIERS];
    if (tid == 0) {
#pragma unroll
        for (int32_t i = 0; i < NUM_BARRIERS; i++)
            mbarrier_init(bars[i], THREADS_PER_BLOCK);
    }
    __syncthreads();

    int32_t current_buffer = NUM_BUFFERS - 1;
    const int32_t num_TC_blocks = blks_offsets[bid + 1] - blks_offsets[bid];
    const int32_t num_blocks_per_row = embedding_dim / BLK_H;
    constexpr int32_t MAX_BLOCKS_PER_STAGE = MAX_FEATURE_DIM_PER_BLOCK / BLK_H;

    if (bid >= num_nodes / BLK_H) return;

    const bool is_producer_warp = (wid >= CONSUMER_WARPS_PER_BLOCK);
    if (is_producer_warp) {
        auto spa_packed_start = blks_offsets[bid] * BLK_H * BLK_W / 32;
        auto ind_start        = blks_offsets[bid] * BLK_W;

        for (int32_t s = 0; s < num_blocks_per_row; s += MAX_BLOCKS_PER_STAGE) {
            int32_t step_blocks = min(MAX_BLOCKS_PER_STAGE,
                                      num_blocks_per_row - s);
            int32_t tma_load_size = step_blocks * BLK_H * sizeof(float);

            for (unsigned i = 0; i < (unsigned)num_TC_blocks; i += 2) {
                current_buffer = (current_buffer + 1) % NUM_BUFFERS;
                auto &ld_bar  = bars[current_buffer];
                auto &mma_bar = bars[NUM_BUFFERS + current_buffer];
                auto dense_X_ptr =
                    &dense_X[current_buffer * NUM_DENSE_X_SHARED_PER_BUFFER_PADDED];
                {
                    mbarrier_arrive_and_wait(mma_bar);

                    if (laneid < 2 && i + laneid < (unsigned)num_TC_blocks) {
                        auto sparse_A_ptr = cast_smem_ptr_to_uint(
                            &sparse_A[current_buffer * 8 + laneid * 4]);
                        int32_t offset_spa_packed =
                            i * BLK_H * BLK_W / 32 + spa_packed_start;
                        auto packed_ptr =
                            &hspa_packed[offset_spa_packed + laneid * 4];
                        asm volatile(
                            "{\n\t"
                            "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                            "\t}" ::"r"(sparse_A_ptr),
                            "l"(packed_ptr));
                    }

                    int32_t offset_ind = i * BLK_W + ind_start;
                    if (laneid < BLK_W * 2 &&
                        i + laneid / BLK_W < (unsigned)num_TC_blocks) {
                        auto src_ofs = hind[laneid + offset_ind];
                        vx_memcpy_async(
                            dense_X_ptr +
                                laneid * MAX_FEATURE_DIM_PER_BLOCK_PADDED,
                            input + src_ofs * embedding_dim + s * BLK_H,
                            tma_load_size, ld_bar);
                    }
                    cp_async_mbarrier_arrive(ld_bar);
                    mbarrier_arrive(ld_bar);
                }
            }
        }
    } else {
        /* consumer warps */
#pragma unroll
        for (int32_t b = 0; b < NUM_BUFFERS; b++)
            mbarrier_arrive(bars[NUM_BUFFERS + b]);

        uint32_t a_frag[2][4];
        uint32_t b_frag[2][MAX_MMAS_PER_WARP][2][2];
        float    acc_frag[MAX_MMAS_PER_WARP][2][4];
        static_assert(sizeof(float[2][4]) == 32);

        /* precompute per-thread indices for the swizzle MMA layout */
        const int32_t frag_row0 = laneid / 4;          /* 0..7 */
        const int32_t frag_col0 = laneid % 4;          /* 0..3 */
        const auto spa_float_start_d = blks_offsets[bid] * BLK_H * BLK_W;

        for (int32_t s = 0; s < num_blocks_per_row; s += MAX_BLOCKS_PER_STAGE) {
            const int32_t step_blocks =
                min(MAX_BLOCKS_PER_STAGE, num_blocks_per_row - s);
            const int32_t num_blocks = step_blocks;
            const int32_t num_mmas =
                num_blocks / CONSUMER_WARPS_PER_BLOCK +
                (wid < num_blocks % CONSUMER_WARPS_PER_BLOCK);

#pragma unroll
            for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++)
#pragma unroll
                for (int32_t ii = 0; ii < 2; ++ii)
#pragma unroll
                    for (int32_t v = 0; v < 4; ++v)
                        acc_frag[f][ii][v] = 0.0f;

            const auto group_id        = laneid >> 2;
            const auto lane_id_in_group = laneid % 4;

            for (int32_t i = 0; i < num_TC_blocks; i += 2) {
                current_buffer = (current_buffer + 1) % NUM_BUFFERS;
                auto &ld_bar  = bars[current_buffer];
                auto &mma_bar = bars[NUM_BUFFERS + current_buffer];
                auto dense_X_ptr  =
                    &dense_X[current_buffer * NUM_DENSE_X_SHARED_PER_BUFFER_PADDED];
                {
                    mbarrier_arrive_and_wait(ld_bar);

                    /* load A fragment from global float hspa (weighted) */
#pragma unroll
                    for (int32_t k = 0; k < 2; k++) {
                        if (i + k < num_TC_blocks) {
                            int32_t base = (i + k) * BLK_H * BLK_W + spa_float_start_d;
                            a_frag[k][0] = __float_as_uint(hspa_float[base + frag_row0       * BLK_W + frag_col0]);
                            a_frag[k][1] = __float_as_uint(hspa_float[base + (frag_row0 + 8) * BLK_W + frag_col0]);
                            a_frag[k][2] = __float_as_uint(hspa_float[base + frag_row0       * BLK_W + (frag_col0 + 4)]);
                            a_frag[k][3] = __float_as_uint(hspa_float[base + (frag_row0 + 8) * BLK_W + (frag_col0 + 4)]);
                        } else {
                            a_frag[k][0] = a_frag[k][1] = a_frag[k][2] = a_frag[k][3] = 0;
                        }
#pragma unroll
                        for (int32_t t = 0; t < 4; t++)
                            asm volatile("cvt.rna.tf32.f32 %0, %0;\n"
                                         : "+r"(a_frag[k][t]));
                    }

#pragma unroll
                    for (int32_t k = 0; k < 2; k++) {
                        auto dense_base =
                            dense_X_ptr + wid * BLK_H +
                            k * BLK_W * MAX_FEATURE_DIM_PER_BLOCK_PADDED;
#pragma unroll
                        for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) {
                            if (f < num_mmas && i + k < num_TC_blocks) {
                                b_frag[k][f][0][0] = *reinterpret_cast<uint32_t *>(
                                    dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id +
                                    lane_id_in_group * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
                                b_frag[k][f][0][1] = *reinterpret_cast<uint32_t *>(
                                    dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id +
                                    (lane_id_in_group + 4) *
                                        MAX_FEATURE_DIM_PER_BLOCK_PADDED);
                                b_frag[k][f][1][0] = *reinterpret_cast<uint32_t *>(
                                    dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id +
                                    8 +
                                    lane_id_in_group * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
                                b_frag[k][f][1][1] = *reinterpret_cast<uint32_t *>(
                                    dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id +
                                    8 +
                                    (lane_id_in_group + 4) *
                                        MAX_FEATURE_DIM_PER_BLOCK_PADDED);
#pragma unroll
                                for (int32_t t = 0; t < 2; t++)
#pragma unroll
                                    for (int32_t v = 0; v < 2; v++)
                                        asm volatile("cvt.rna.tf32.f32 %0, %0;"
                                                     : "+r"(b_frag[k][f][t][v]));

                                MMA<16, 8, 8>::mma(a_frag[k], b_frag[k][f][0],
                                                   acc_frag[f][0]);
                                MMA<16, 8, 8>::mma(a_frag[k], b_frag[k][f][1],
                                                   acc_frag[f][1]);
                            }
                        }
                    }
                    mbarrier_arrive(mma_bar);
                }
            }

            auto offset_base =
                output + bid * BLK_H * embedding_dim + (s + wid) * BLK_H;

#pragma unroll
            for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) {
                if (f < num_mmas) {
#pragma unroll
                    for (int32_t ii = 0; ii < 2; ii++) {
#pragma unroll
                        for (int32_t v = 0; v < 4; v += 2) {
                            auto row = group_id + (v / 2) * 8;
                            auto col = (lane_id_in_group * 2) + ii * 8;
                            auto ptr = offset_base + f * DENSE_X_STRIDE_PER_MMA +
                                       row * embedding_dim + col;
                            double val = *(double *)&acc_frag[f][ii][v];
                            asm volatile("st.global.wt.f64 [%0], %1;\n" ::"l"(ptr),
                                         "d"(val));
                        }
                    }
                }
            }
        }
    }

    if (tid == 0) {
#pragma unroll
        for (int32_t i = 0; i < NUM_BARRIERS; i++)
            invalidate(bars[i]);
    }
}

/* --- spmm_mma161616_spa_swizzle_dd  (model 2, 2D grid) ---------------- */
template <int32_t NUM_AGENTS, int32_t NUM_BUFFERS,
          int32_t MAX_MMAS_PER_WARP,
          int32_t CONSUMER_WARPS_PER_BLOCK, int32_t PADDING_SIZE = 0>
__global__ void
spmm_mma161616_spa_swizzle_dd(
    const int *__restrict__ blks_offsets,
    const uint32_t *__restrict__ hspa_packed,
    const float *__restrict__ hspa_float,
    const int32_t *__restrict__ hind,
    const int num_nodes, const int num_edges,
    const int embedding_dim,
    const float *__restrict__ input,
    float *__restrict__ output) {

    using Traits = PersistenKernelTraits<float, NUM_AGENTS, NUM_BUFFERS,
                                         MAX_MMAS_PER_WARP,
                                         CONSUMER_WARPS_PER_BLOCK,
                                         PADDING_SIZE>;

    constexpr int32_t NUM_BARRIERS        = Traits::NUM_BARRIERS;
    constexpr int32_t THREADS_PER_WARP    = Traits::THREADS_PER_WARP;
    constexpr int32_t THREADS_PER_BLOCK   = Traits::THREADS_PER_BLOCK;
    constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK =
        Traits::MAX_FEATURE_DIM_PER_BLOCK;
    constexpr int32_t MAX_FEATURE_DIM_PER_BLOCK_PADDED =
        Traits::MAX_FEATURE_DIM_PER_BLOCK_PADDED;
    constexpr int32_t NUM_DENSE_X_SHARED_PER_BUFFER_PADDED =
        2 * Traits::NUM_DENSE_X_SHARED_PER_BUFFER_PADDED;
    constexpr int32_t DENSE_X_STRIDE_PER_MMA =
        Traits::DENSE_X_STRIDE_PER_MMA;

    const int32_t wid    = threadIdx.y;
    const int32_t laneid = threadIdx.x;
    const int32_t tid    = threadIdx.y * THREADS_PER_WARP + laneid;

    alignas(16) extern __shared__ float dense_X[];
    alignas(16) __shared__ uint32_t sparse_A[NUM_BUFFERS * 4 * 2];

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t bars[NUM_BARRIERS];
    if (tid == 0) {
#pragma unroll
        for (int32_t i = 0; i < NUM_BARRIERS; i++)
            mbarrier_init(bars[i], THREADS_PER_BLOCK);
    }
    __syncthreads();

    int32_t current_buffer = NUM_BUFFERS - 1;
    const int32_t num_blocks_per_row = embedding_dim / BLK_H;
    constexpr int32_t MAX_BLOCKS_PER_STAGE = MAX_FEATURE_DIM_PER_BLOCK / BLK_H;

    int32_t row           = blockIdx.x;
    int32_t start_col_blk = blockIdx.y * MAX_BLOCKS_PER_STAGE;
    int32_t step_blocks   = min(MAX_BLOCKS_PER_STAGE,
                                num_blocks_per_row - start_col_blk);

    const int32_t num_TC_blocks = blks_offsets[row + 1] - blks_offsets[row];
    if (row >= num_nodes / BLK_H) return;

    const bool is_producer_warp = (wid >= CONSUMER_WARPS_PER_BLOCK);
    if (is_producer_warp) {
        auto spa_packed_start = blks_offsets[row] * BLK_H * BLK_W / 32;
        auto ind_start        = blks_offsets[row] * BLK_W;
        int32_t tma_load_size = step_blocks * BLK_H * sizeof(float);

        for (unsigned i = 0; i < (unsigned)num_TC_blocks; i += 2) {
            current_buffer = (current_buffer + 1) % NUM_BUFFERS;
            auto &ld_bar   = bars[current_buffer];
            auto &mma_bar  = bars[NUM_BUFFERS + current_buffer];
            auto dense_X_ptr =
                &dense_X[current_buffer * NUM_DENSE_X_SHARED_PER_BUFFER_PADDED];
            {
                mbarrier_arrive_and_wait(mma_bar);

                if (laneid < 2 && i + laneid < (unsigned)num_TC_blocks) {
                    auto sparse_A_ptr = cast_smem_ptr_to_uint(
                        &sparse_A[current_buffer * 8 + laneid * 4]);
                    int32_t offset_spa_packed =
                        i * BLK_H * BLK_W / 32 + spa_packed_start;
                    auto packed_ptr =
                        &hspa_packed[offset_spa_packed + laneid * 4];
                    asm volatile(
                        "{\n\t"
                        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                        "\t}" ::"r"(sparse_A_ptr),
                        "l"(packed_ptr));
                }

                int32_t offset_ind = i * BLK_W + ind_start;
                if (laneid < BLK_W * 2 &&
                    i + laneid / BLK_W < (unsigned)num_TC_blocks) {
                    auto src_ofs = hind[laneid + offset_ind];
                    vx_memcpy_async(
                        dense_X_ptr + laneid * MAX_FEATURE_DIM_PER_BLOCK_PADDED,
                        input + src_ofs * embedding_dim + start_col_blk * BLK_H,
                        tma_load_size, ld_bar);
                }
                cp_async_mbarrier_arrive(ld_bar);
                mbarrier_arrive(ld_bar);
            }
        }
    } else {
#pragma unroll
        for (int32_t b = 0; b < NUM_BUFFERS; b++)
            mbarrier_arrive(bars[NUM_BUFFERS + b]);

        uint32_t a_frag[2][4];
        uint32_t b_frag[2][MAX_MMAS_PER_WARP][2][2];
        float    acc_frag[MAX_MMAS_PER_WARP][2][4];
        static_assert(sizeof(float[2][4]) == 32);

        /* precompute per-thread indices for the swizzle MMA layout */
        const int32_t frag_row0 = laneid / 4;
        const int32_t frag_col0 = laneid % 4;
        const auto spa_float_start_dd = blks_offsets[row] * BLK_H * BLK_W;

        const int32_t num_blocks = step_blocks;
        const int32_t num_mmas =
            num_blocks / CONSUMER_WARPS_PER_BLOCK +
            (wid < num_blocks % CONSUMER_WARPS_PER_BLOCK);

#pragma unroll
        for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++)
#pragma unroll
            for (int32_t ii = 0; ii < 2; ++ii)
#pragma unroll
                for (int32_t v = 0; v < 4; ++v)
                    acc_frag[f][ii][v] = 0.0f;

        const auto group_id        = laneid >> 2;
        const auto lane_id_in_group = laneid % 4;

        for (int32_t i = 0; i < num_TC_blocks; i += 2) {
            current_buffer = (current_buffer + 1) % NUM_BUFFERS;
            auto &ld_bar   = bars[current_buffer];
            auto &mma_bar  = bars[NUM_BUFFERS + current_buffer];
            auto dense_X_ptr  =
                &dense_X[current_buffer * NUM_DENSE_X_SHARED_PER_BUFFER_PADDED];
            {
                mbarrier_arrive_and_wait(ld_bar);

                /* load A fragment from global float hspa (weighted) */
#pragma unroll
                for (int32_t k = 0; k < 2; k++) {
                    if (i + k < num_TC_blocks) {
                        int32_t base = (i + k) * BLK_H * BLK_W + spa_float_start_dd;
                        a_frag[k][0] = __float_as_uint(hspa_float[base + frag_row0       * BLK_W + frag_col0]);
                        a_frag[k][1] = __float_as_uint(hspa_float[base + (frag_row0 + 8) * BLK_W + frag_col0]);
                        a_frag[k][2] = __float_as_uint(hspa_float[base + frag_row0       * BLK_W + (frag_col0 + 4)]);
                        a_frag[k][3] = __float_as_uint(hspa_float[base + (frag_row0 + 8) * BLK_W + (frag_col0 + 4)]);
                    } else {
                        a_frag[k][0] = a_frag[k][1] = a_frag[k][2] = a_frag[k][3] = 0;
                    }
#pragma unroll
                    for (int32_t t = 0; t < 4; t++)
                        asm volatile("cvt.rna.tf32.f32 %0, %0;\n"
                                     : "+r"(a_frag[k][t]));
                }
#pragma unroll
                for (int32_t k = 0; k < 2; k++) {
                    auto dense_base =
                        dense_X_ptr + wid * BLK_H +
                        k * BLK_W * MAX_FEATURE_DIM_PER_BLOCK_PADDED;
#pragma unroll
                    for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) {
                        if (f < num_mmas && i + k < num_TC_blocks) {
                            b_frag[k][f][0][0] = *reinterpret_cast<uint32_t *>(
                                dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id +
                                lane_id_in_group * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
                            b_frag[k][f][0][1] = *reinterpret_cast<uint32_t *>(
                                dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id +
                                (lane_id_in_group + 4) *
                                    MAX_FEATURE_DIM_PER_BLOCK_PADDED);
                            b_frag[k][f][1][0] = *reinterpret_cast<uint32_t *>(
                                dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id + 8 +
                                lane_id_in_group * MAX_FEATURE_DIM_PER_BLOCK_PADDED);
                            b_frag[k][f][1][1] = *reinterpret_cast<uint32_t *>(
                                dense_base + f * DENSE_X_STRIDE_PER_MMA + group_id + 8 +
                                (lane_id_in_group + 4) *
                                    MAX_FEATURE_DIM_PER_BLOCK_PADDED);
#pragma unroll
                            for (int32_t t = 0; t < 2; t++)
#pragma unroll
                                for (int32_t v = 0; v < 2; v++)
                                    asm volatile("cvt.rna.tf32.f32 %0, %0;"
                                                 : "+r"(b_frag[k][f][t][v]));
                            MMA<16, 8, 8>::mma(a_frag[k], b_frag[k][f][0],
                                               acc_frag[f][0]);
                            MMA<16, 8, 8>::mma(a_frag[k], b_frag[k][f][1],
                                               acc_frag[f][1]);
                        }
                    }
                }
                mbarrier_arrive(mma_bar);
            }
        }

        auto offset_base =
            output + row * BLK_H * embedding_dim + (start_col_blk + wid) * BLK_H;

#pragma unroll
        for (int32_t f = 0; f < MAX_MMAS_PER_WARP; f++) {
            if (f < num_mmas) {
#pragma unroll
                for (int32_t ii = 0; ii < 2; ii++) {
#pragma unroll
                    for (int32_t v = 0; v < 4; v += 2) {
                        auto r = group_id + (v / 2) * 8;
                        auto c = (lane_id_in_group * 2) + ii * 8;
                        auto ptr = offset_base + f * DENSE_X_STRIDE_PER_MMA +
                                   r * embedding_dim + c;
                        double val = *(double *)&acc_frag[f][ii][v];
                        asm volatile("st.global.wt.f64 [%0], %1;\n" ::"l"(ptr),
                                     "d"(val));
                    }
                }
            }
        }
    }

    if (tid == 0) {
#pragma unroll
        for (int32_t i = 0; i < NUM_BARRIERS; i++)
            invalidate(bars[i]);
    }
}

/* ================================================================
 *  Section 5 : voltrix_spmm_forward_cuda  (dispatch)
 * ================================================================ */

static void
voltrix_spmm_forward_cuda(
    const int *blks_offsets,
    const uint32_t *hspa_packed,
    const float *hspa_float,
    const int32_t *hind,
    int num_nodes, int num_edges, int embedding_dim,
    const float *input, float *output,
    int32_t model, cudaStream_t stream) {

    constexpr auto extra_dynamic_shared_memory = 226 * 1024;

    if (model == 0) {
        constexpr int32_t NUM_AGENTS              = 2;
        constexpr int32_t PADDING_SIZE            = 8;
        constexpr int32_t NUM_BUFFERS             = 2;
        constexpr int32_t MAX_MMAS_PER_WARP       = 8;
        constexpr int32_t CONSUMER_WARPS_PER_BLOCK = 4;
        constexpr int32_t NUM_THREAD_BLOCKS       = 114 * 2;
        using Config = PersistenKernelConfig<float, NUM_AGENTS, NUM_BUFFERS,
                                             MAX_MMAS_PER_WARP,
                                             CONSUMER_WARPS_PER_BLOCK,
                                             NUM_THREAD_BLOCKS, PADDING_SIZE>;
        const dim3 GRID  = dim3(num_nodes / BLK_H, 1, 1);
        constexpr dim3 BLOCK = Config::BLOCK;
        constexpr int32_t DENSE_X_SHARED_MEMORY_SIZE =
            Config::DENSE_X_SHARED_MEMORY_SIZE;
        static auto spmm_func =
            spmm_mma161616_spa_swizzle_d<NUM_AGENTS, NUM_BUFFERS,
                                         MAX_MMAS_PER_WARP,
                                         CONSUMER_WARPS_PER_BLOCK,
                                         PADDING_SIZE>;
        cudaFuncSetAttribute(spmm_func,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             extra_dynamic_shared_memory);
        spmm_func<<<GRID, BLOCK, DENSE_X_SHARED_MEMORY_SIZE * 2, stream>>>(
            blks_offsets, hspa_packed, hspa_float, hind,
            num_nodes, num_edges, embedding_dim,
            input, output);
    } else if (model == 1) {
        constexpr int32_t NUM_AGENTS              = 2;
        constexpr int32_t PADDING_SIZE            = 8;
        constexpr int32_t NUM_BUFFERS             = 2;
        constexpr int32_t MAX_MMAS_PER_WARP       = 4;
        constexpr int32_t CONSUMER_WARPS_PER_BLOCK = 4;
        constexpr int32_t NUM_THREAD_BLOCKS       = 114 * 2;
        using Config = PersistenKernelConfig<float, NUM_AGENTS, NUM_BUFFERS,
                                             MAX_MMAS_PER_WARP,
                                             CONSUMER_WARPS_PER_BLOCK,
                                             NUM_THREAD_BLOCKS, PADDING_SIZE>;
        const dim3 GRID  = dim3(num_nodes / BLK_H, 1, 1);
        constexpr dim3 BLOCK = Config::BLOCK;
        constexpr int32_t DENSE_X_SHARED_MEMORY_SIZE =
            Config::DENSE_X_SHARED_MEMORY_SIZE;
        static auto spmm_func =
            spmm_mma161616_spa_swizzle_d<NUM_AGENTS, NUM_BUFFERS,
                                         MAX_MMAS_PER_WARP,
                                         CONSUMER_WARPS_PER_BLOCK,
                                         PADDING_SIZE>;
        cudaFuncSetAttribute(spmm_func,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             extra_dynamic_shared_memory);
        spmm_func<<<GRID, BLOCK, DENSE_X_SHARED_MEMORY_SIZE * 2, stream>>>(
            blks_offsets, hspa_packed, hspa_float, hind,
            num_nodes, num_edges, embedding_dim,
            input, output);
    } else if (model == 2) {
        constexpr int32_t NUM_AGENTS              = 2;
        constexpr int32_t PADDING_SIZE            = 8;
        constexpr int32_t NUM_BUFFERS             = 4;
        constexpr int32_t MAX_MMAS_PER_WARP       = 2;
        constexpr int32_t CONSUMER_WARPS_PER_BLOCK = 4;
        constexpr int32_t NUM_THREAD_BLOCKS       = 114 * 2;
        using Config = PersistenKernelConfig<float, NUM_AGENTS, NUM_BUFFERS,
                                             MAX_MMAS_PER_WARP,
                                             CONSUMER_WARPS_PER_BLOCK,
                                             NUM_THREAD_BLOCKS, PADDING_SIZE>;
        const dim3 GRID  = dim3(
            num_nodes / BLK_H,
            vx_div_round_up(static_cast<int32_t>(embedding_dim),
                            (int32_t)Config::Traits::MAX_FEATURE_DIM_PER_BLOCK),
            1);
        constexpr dim3 BLOCK = Config::BLOCK;
        constexpr int32_t DENSE_X_SHARED_MEMORY_SIZE =
            Config::DENSE_X_SHARED_MEMORY_SIZE;
        static auto spmm_func =
            spmm_mma161616_spa_swizzle_dd<NUM_AGENTS, NUM_BUFFERS,
                                          MAX_MMAS_PER_WARP,
                                          CONSUMER_WARPS_PER_BLOCK,
                                          PADDING_SIZE>;
        cudaFuncSetAttribute(spmm_func,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             extra_dynamic_shared_memory);
        spmm_func<<<GRID, BLOCK, DENSE_X_SHARED_MEMORY_SIZE * 2, stream>>>(
            blks_offsets, hspa_packed, hspa_float, hind,
            num_nodes, num_edges, embedding_dim,
            input, output);
    } else {
        fprintf(stderr, "Invalid model %d\n", model);
        exit(1);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Voltrix kernel launch failed (model %d): %s\n",
                model, cudaGetErrorString(err));
    }
}

/* ================================================================
 *  Section 6 : Preprocessing kernels  (from bmat_kernels.cuh)
 * ================================================================ */

/* --- hmat_cuda_kernel : build hspa (float) and hind -------------------- */
__global__ void
hmat_cuda_kernel(
    const int *__restrict__ nodePointer,
    const int *__restrict__ edgeList,
    const int *__restrict__ blockPartition,
    const int *__restrict__ edgeToColumn,
    const int *__restrict__ edgeToRow,
    const int *__restrict__ Pointer1,
    const float *__restrict__ edgeValues,
    const int numNodes, const int numEdges,
    float *hspa, int *hind) {

    const unsigned bid  = blockIdx.x;
    const unsigned wid  = threadIdx.y;
    const unsigned laneid = threadIdx.x;
    const unsigned tid  = threadIdx.y * blockDim.x + laneid;
    const unsigned warpSz = blockDim.x;
    const unsigned threadPerBlock = blockDim.x * blockDim.y;

    const unsigned nIdx_start = bid * BLK_H;
    const unsigned nIdx_end   = min((unsigned)(bid + 1) * BLK_H, (unsigned)numNodes);
    const unsigned eIdx_start = nodePointer[nIdx_start];
    const unsigned eIdx_end   = nodePointer[nIdx_end];
    const unsigned num_TC_blocks = Pointer1[bid + 1] - Pointer1[bid];

    const unsigned spa_start = Pointer1[bid] * BLK_H * BLK_W;
    const unsigned ind_start = Pointer1[bid] * BLK_W;

    for (unsigned i = 0; i < num_TC_blocks; i++) {
        int offset_spa = i * BLK_H * BLK_W + spa_start;
        int offset_ind = i * BLK_W + ind_start;

        if (tid < (unsigned)BLK_W)
            hind[tid + offset_ind] = 0;
        __syncthreads();

#pragma unroll
        for (unsigned idx = tid; idx < (unsigned)(BLK_W * BLK_H); idx += threadPerBlock)
            hspa[idx + offset_spa] = 0;
        __syncthreads();

#pragma unroll
        for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end;
             eIdx += threadPerBlock) {
            unsigned col = edgeToColumn[eIdx];
            if (i * BLK_W <= col && col < (i + 1) * BLK_W) {
                unsigned row_local = edgeToRow[eIdx] % BLK_H;
                unsigned col_local = col % BLK_W;
                hspa[row_local * BLK_W + col_local + offset_spa] = edgeValues[eIdx];
                hind[col_local + offset_ind] = edgeList[eIdx];
            }
        }
        __syncthreads();
    }
}

/* --- hmat_convert_uint32_swizzle_cuda_kernel : pack + swizzle ---------- */
__global__ void
hmat_convert_uint32_swizzle_cuda_kernel(
    const int *__restrict__ Pointer1,
    const float *__restrict__ hspa,
    uint32_t *packed_hspa) {

    const unsigned bid  = blockIdx.x;
    const unsigned tid  = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned threadPerBlock = blockDim.x * blockDim.y;

    const unsigned num_TC_blocks     = Pointer1[bid + 1] - Pointer1[bid];
    const unsigned spa_start         = Pointer1[bid] * BLK_H * BLK_W;
    const unsigned spa_packed_start  = Pointer1[bid] * BLK_H * BLK_W / 32;

    for (unsigned i = 0; i < num_TC_blocks; i++) {
        int offset_spa        = i * BLK_H * BLK_W + spa_start;
        int offset_spa_packed = i * BLK_H * BLK_W / 32 + spa_packed_start;

#pragma unroll
        for (unsigned idx = tid; idx < (unsigned)(BLK_W * BLK_H / 32);
             idx += threadPerBlock) {
            uint32_t byte = 0;
            for (int bit = 0; bit < 32; ++bit) {
                int group_id            = bit >> 2;
                int thread_id_in_group  = bit % 4;
                int row = group_id + 8 * (idx % 2);
                int col = thread_id_in_group + 4 * (idx / 2);
                float val = hspa[row * BLK_W + col + offset_spa];
                if (fabsf(val) > 1e-5f)
                    byte |= (1 << bit);
            }
            packed_hspa[idx + offset_spa_packed] = byte;
        }
    }
}

/* ================================================================
 *  Section 7 : CPU preprocessing  (from bmat_kernels.cuh)
 * ================================================================ */

static std::map<unsigned, unsigned>
inplace_deduplication(unsigned *array, unsigned length) {
    int loc = 0, cur = 1;
    std::map<unsigned, unsigned> nb2col;
    nb2col[array[0]] = 0;
    while (cur < (int)length) {
        if (array[cur] != array[cur - 1]) {
            loc++;
            array[loc] = array[cur];
            nb2col[array[cur]] = loc;
        }
        cur++;
    }
    return nb2col;
}

static void
voltrix_preprocess(
    const int32_t *edgeList, const int32_t *nodePointer,
    int num_nodes,
    int32_t *blockPartition, int32_t *edgeToColumn,
    int32_t *edgeToRow, int32_t *Pointer1) {

    unsigned block_counter = 0;

#pragma omp parallel for
    for (int nid = 0; nid < num_nodes; nid++) {
        for (int eid = nodePointer[nid]; eid < nodePointer[nid + 1]; eid++)
            edgeToRow[eid] = nid;
    }

#pragma omp parallel for reduction(+ : block_counter)
    for (int iter = 0; iter < num_nodes; iter += BLK_H) {
        unsigned windowId      = iter / BLK_H;
        unsigned block_start   = nodePointer[iter];
        unsigned block_end     = nodePointer[std::min(iter + BLK_H, num_nodes)];
        unsigned num_window_edges = block_end - block_start;

        unsigned *neighbor_window =
            (unsigned *)malloc(num_window_edges * sizeof(unsigned));
        memcpy(neighbor_window, &edgeList[block_start],
               num_window_edges * sizeof(unsigned));

        thrust::sort(neighbor_window, neighbor_window + num_window_edges);

        std::map<unsigned, unsigned> clean_edges2col =
            inplace_deduplication(neighbor_window, num_window_edges);

        blockPartition[windowId] =
            (clean_edges2col.size() + BLK_W - 1) / BLK_W;
        block_counter += blockPartition[windowId];

        for (unsigned e_index = block_start; e_index < block_end; e_index++) {
            unsigned eid = edgeList[e_index];
            edgeToColumn[e_index] = clean_edges2col[eid];
        }
        free(neighbor_window);
    }

    const size_t num_row_windows = (num_nodes + BLK_H - 1) / BLK_H;
    std::vector<int> bp_vec(blockPartition, blockPartition + num_row_windows);
    thrust::inclusive_scan(bp_vec.begin(), bp_vec.end(), Pointer1 + 1);
    Pointer1[0] = 0;
}

/* ================================================================
 *  Section 8 : GPU wrapper functions for preprocessing
 * ================================================================ */

static void
run_hmat_cuda(const int32_t *d_nodePointer, const int32_t *d_edgeList,
              const int32_t *d_blockPartition, const int32_t *d_edgeToColumn,
              const int32_t *d_edgeToRow, const int32_t *d_Pointer1,
              const float *d_edgeValues,
              int32_t num_row_windows, int num_nodes, int num_edges,
              float *d_hspa, int *d_hind) {
    dim3 grid(num_row_windows, 1, 1);
    dim3 block(WARP_SIZE, WPB, 1);
    hmat_cuda_kernel<<<grid, block>>>(
        d_nodePointer, d_edgeList, d_blockPartition,
        d_edgeToColumn, d_edgeToRow, d_Pointer1,
        d_edgeValues,
        num_nodes, num_edges, d_hspa, d_hind);
    cudaDeviceSynchronize();
}

static void
run_hmat_packed_swizzle_cuda(int32_t num_row_windows,
                             const int32_t *d_Pointer1,
                             const float *d_hspa,
                             uint32_t *d_hspa_packed) {
    dim3 grid(num_row_windows, 1, 1);
    dim3 block(WARP_SIZE, WPB, 1);
    hmat_convert_uint32_swizzle_cuda_kernel<<<grid, block>>>(
        d_Pointer1, d_hspa, d_hspa_packed);
    cudaDeviceSynchronize();
}

/* ================================================================
 *  Section 9 : CSV loader  (shared with DTC / FlashSparse tests)
 * ================================================================ */

struct CsrGraph {
    int num_nodes, num_edges;
    std::vector<int> row_ptr, col_idx;
    std::vector<float> vals;
};

static CsrGraph load_csr_csv(const char *path) {
    CsrGraph g;
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    std::string line;
    int ln = 0;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            float v = std::stof(tok);
            if (ln == 0) g.row_ptr.push_back((int)v);
            else if (ln == 1) g.col_idx.push_back((int)v);
            else g.vals.push_back(v);
        }
        ln++;
    }
    g.num_nodes = (int)g.row_ptr.size() - 1;
    g.num_edges = (int)g.col_idx.size();
    return g;
}

/* ================================================================
 *  Section 10 : cuSPARSE reference
 * ================================================================ */

static void cusparse_spmm_ref(
    int M, int N, int K, int nnz,
    const int *d_row, const int *d_col, const float *d_val,
    const float *d_B, float *d_C) {

    cusparseHandle_t h;  cusparseCreate(&h);
    cusparseSpMatDescr_t A;
    cusparseDnMatDescr_t matB, matC;
    cusparseCreateCsr(&A, M, N, nnz,
                      (void *)d_row, (void *)d_col, (void *)d_val,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matB, N, K, K, (void *)d_B,
                        CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, M, K, K, (void *)d_C,
                        CUDA_R_32F, CUSPARSE_ORDER_ROW);
    float alpha = 1.0f, beta = 0.0f;
    size_t buf_size = 0;
    cusparseSpMM_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, A, matB, &beta, matC,
                            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                            &buf_size);
    void *buf = nullptr;
    if (buf_size) cudaMalloc(&buf, buf_size);
    cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, A, matB, &beta, matC,
                 CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buf);
    cudaDeviceSynchronize();
    if (buf) cudaFree(buf);
    cusparseDestroySpMat(A);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(h);
}

/* ================================================================
 *  Section 11 : GPU timer
 * ================================================================ */

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void tick()  { cudaEventRecord(start); }
    float tock() {
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

/* ================================================================
 *  Section 12 : Result struct & table helpers
 * ================================================================ */

struct VxRunResult {
    int num_nodes = 0;
    int num_edges = 0;
    int embedding_dim = 0;
    double preprocess_ms = 0.0;
    double cusp_ms = 0.0, cusp_gflops = 0.0;
    int best_model = -1;
    double vx_ms = 0.0, vx_gflops = 0.0, vx_rmse = 0.0;
};

static std::string extract_dataset_name(const std::string &path) {
    size_t slash = path.find_last_of('/');
    std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) base = base.substr(0, dot);
    return base;
}

static void print_table_header() {
    printf("%-14s %8s %10s %4s | %9s %9s | %9s %9s %10s | %9s\n",
           "Dataset", "Nodes", "Edges", "K",
           "cuSP/ms", "cuSP/GF",
           "VX/ms", "VX/GF", "VX/RMSE",
           "Prep/ms");
    printf("%-14s %8s %10s %4s | %9s %9s | %9s %9s %10s | %9s\n",
           "--------------", "--------", "----------", "----",
           "---------", "---------",
           "---------", "---------", "----------",
           "---------");
}

static void print_table_row(const char *name, const VxRunResult &R) {
    printf("%-14s %8d %10d %4d | %9.4f %9.2f | %9.4f %9.2f %10.2e | %9.2f\n",
           name, R.num_nodes, R.num_edges, R.embedding_dim,
           R.cusp_ms, R.cusp_gflops,
           R.vx_ms, R.vx_gflops, R.vx_rmse,
           R.preprocess_ms);
}

/* ================================================================
 *  Section 13 : run_benchmark
 * ================================================================ */

static VxRunResult run_benchmark(const char *csv_path, int K) {
    /* --- load graph --------------------------------------------------- */
    CsrGraph g = load_csr_csv(csv_path);
    int orig_nodes = g.num_nodes;
    int orig_edges = g.num_edges;

    /* pad to multiple of BLK_H */
    int padded_nodes = ((orig_nodes + BLK_H - 1) / BLK_H) * BLK_H;
    g.row_ptr.resize(padded_nodes + 1, g.row_ptr.back());
    g.num_nodes = padded_nodes;

    /* also pad K to multiple of BLK_H (16) since kernel requires it */
    int padded_K = ((K + BLK_H - 1) / BLK_H) * BLK_H;

    VxRunResult R;
    R.num_nodes     = orig_nodes;
    R.num_edges     = orig_edges;
    R.embedding_dim = K;

    int N = padded_nodes;
    int E = orig_edges;
    int num_row_windows = N / BLK_H;

    /* --- CPU preprocessing -------------------------------------------- */
    std::vector<int32_t> edgeToColumn(E, 0), edgeToRow(E, 0);
    std::vector<int32_t> blockPartition(num_row_windows, 0);
    std::vector<int32_t> Pointer1(num_row_windows + 1, 0);

    GpuTimer ptimer;
    ptimer.tick();

    voltrix_preprocess(g.col_idx.data(), g.row_ptr.data(), N,
                       blockPartition.data(), edgeToColumn.data(),
                       edgeToRow.data(), Pointer1.data());

    int total_blocks = Pointer1[num_row_windows];

    /* --- allocate GPU preprocessing buffers --------------------------- */
    int32_t *d_nodePointer, *d_edgeList, *d_blockPartition;
    int32_t *d_edgeToColumn, *d_edgeToRow, *d_Pointer1;
    float *d_hspa;
    int *d_hind;
    uint32_t *d_hspa_packed;

    cudaMalloc(&d_nodePointer,    (N + 1) * sizeof(int32_t));
    cudaMalloc(&d_edgeList,       E * sizeof(int32_t));
    cudaMalloc(&d_blockPartition, num_row_windows * sizeof(int32_t));
    cudaMalloc(&d_edgeToColumn,   E * sizeof(int32_t));
    cudaMalloc(&d_edgeToRow,      E * sizeof(int32_t));
    cudaMalloc(&d_Pointer1,       (num_row_windows + 1) * sizeof(int32_t));
    cudaMalloc(&d_hspa,           (size_t)total_blocks * BLK_H * BLK_W * sizeof(float));
    cudaMalloc(&d_hind,           (size_t)total_blocks * BLK_W * sizeof(int));
    cudaMalloc(&d_hspa_packed,    (size_t)total_blocks * BLK_H * BLK_W / 32 * sizeof(uint32_t));

    cudaMemcpy(d_nodePointer,    g.row_ptr.data(),      (N + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeList,       g.col_idx.data(),      E * sizeof(int32_t),       cudaMemcpyHostToDevice);
    cudaMemcpy(d_blockPartition, blockPartition.data(),  num_row_windows * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeToColumn,   edgeToColumn.data(),    E * sizeof(int32_t),       cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeToRow,      edgeToRow.data(),       E * sizeof(int32_t),       cudaMemcpyHostToDevice);
    cudaMemcpy(d_Pointer1,       Pointer1.data(),        (num_row_windows + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);

    /* copy edge values to GPU */
    float *d_edgeValues;
    cudaMalloc(&d_edgeValues, E * sizeof(float));
    cudaMemcpy(d_edgeValues, g.vals.data(), E * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_hspa, 0, (size_t)total_blocks * BLK_H * BLK_W * sizeof(float));
    cudaMemset(d_hind, 0, (size_t)total_blocks * BLK_W * sizeof(int));
    cudaMemset(d_hspa_packed, 0, (size_t)total_blocks * BLK_H * BLK_W / 32 * sizeof(uint32_t));

    /* --- GPU preprocessing: hmat + pack -------------------------------- */
    run_hmat_cuda(d_nodePointer, d_edgeList, d_blockPartition,
                  d_edgeToColumn, d_edgeToRow, d_Pointer1,
                  d_edgeValues,
                  num_row_windows, N, E, d_hspa, d_hind);

    run_hmat_packed_swizzle_cuda(num_row_windows, d_Pointer1, d_hspa, d_hspa_packed);

    float prep_ms = ptimer.tock();
    R.preprocess_ms = prep_ms;

    /* --- allocate input / output -------------------------------------- */
    size_t B_bytes = (size_t)N * padded_K * sizeof(float);
    size_t C_bytes = (size_t)N * padded_K * sizeof(float);

    float *d_B, *d_C_cusp, *d_C_vx;
    cudaMalloc(&d_B,      B_bytes);
    cudaMalloc(&d_C_cusp, C_bytes);
    cudaMalloc(&d_C_vx,   C_bytes);

    /* random dense matrix */
    {
        std::vector<float> h_B(N * padded_K);
        srand(42);
        for (auto &v : h_B) v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        cudaMemcpy(d_B, h_B.data(), B_bytes, cudaMemcpyHostToDevice);
    }

    /* --- cuSPARSE reference ------------------------------------------- */
    /* cuSPARSE uses actual edge values */
    float *d_row_f, *d_col_f, *d_val_f;
    {
        cudaMalloc(&d_row_f, (N + 1) * sizeof(int));
        cudaMalloc(&d_col_f, E * sizeof(int));
        cudaMalloc(&d_val_f, E * sizeof(float));
        cudaMemcpy(d_row_f, g.row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_f, g.col_idx.data(), E * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val_f, g.vals.data(),    E * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaMemset(d_C_cusp, 0, C_bytes);
    cusparse_spmm_ref(N, N, padded_K, E,
                      (int *)d_row_f, (int *)d_col_f, (float *)d_val_f,
                      d_B, d_C_cusp);

    /* cuSPARSE timing */
    const int WARMUP = 10, ITERS = 100;
    GpuTimer timer;
    for (int i = 0; i < WARMUP; i++) {
        cudaMemset(d_C_cusp, 0, C_bytes);
        cusparse_spmm_ref(N, N, padded_K, E,
                          (int *)d_row_f, (int *)d_col_f, (float *)d_val_f,
                          d_B, d_C_cusp);
    }
    /* get reference output */
    cusparse_spmm_ref(N, N, padded_K, E,
                      (int *)d_row_f, (int *)d_col_f, (float *)d_val_f,
                      d_B, d_C_cusp);
    std::vector<float> h_ref(N * padded_K);
    cudaMemcpy(h_ref.data(), d_C_cusp, C_bytes, cudaMemcpyDeviceToHost);

    timer.tick();
    for (int i = 0; i < ITERS; i++) {
        cusparse_spmm_ref(N, N, padded_K, E,
                          (int *)d_row_f, (int *)d_col_f, (float *)d_val_f,
                          d_B, d_C_cusp);
    }
    float cusp_ms = timer.tock() / ITERS;
    double cusp_gflops = 2.0 * E * padded_K / (cusp_ms * 1e6);
    R.cusp_ms     = cusp_ms;
    R.cusp_gflops = cusp_gflops;

    /* --- Voltrix SpMM : try all 3 models, pick best ------------------- */
    float best_vx_ms   = 1e30f;
    int   best_model   = -1;
    double best_rmse   = 0.0;
    double best_gflops = 0.0;
    std::vector<float> h_vx(N * padded_K);

    for (int model = 0; model < 3; model++) {
        /* warmup */
        cudaMemset(d_C_vx, 0, C_bytes);
        voltrix_spmm_forward_cuda(d_Pointer1, d_hspa_packed,
                                  d_hspa,
                                  (int32_t *)d_hind, N, E, padded_K,
                                  d_B, d_C_vx, model, 0);
        cudaDeviceSynchronize();
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            cudaGetLastError(); // clear
            continue;
        }

        /* accuracy check */
        cudaMemset(d_C_vx, 0, C_bytes);
        voltrix_spmm_forward_cuda(d_Pointer1, d_hspa_packed,
                                  d_hspa,
                                  (int32_t *)d_hind, N, E, padded_K,
                                  d_B, d_C_vx, model, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(h_vx.data(), d_C_vx, C_bytes, cudaMemcpyDeviceToHost);

        double sq_sum = 0;
        int count = orig_nodes * padded_K;
        for (int r = 0; r < orig_nodes; r++)
            for (int c = 0; c < padded_K; c++) {
                double d = (double)h_vx[r * padded_K + c] - (double)h_ref[r * padded_K + c];
                sq_sum += d * d;
            }
        double rmse = std::sqrt(sq_sum / count);

        /* timing */
        for (int i = 0; i < WARMUP; i++) {
            voltrix_spmm_forward_cuda(d_Pointer1, d_hspa_packed,
                                      d_hspa,
                                      (int32_t *)d_hind, N, E, padded_K,
                                      d_B, d_C_vx, model, 0);
        }
        cudaDeviceSynchronize();

        timer.tick();
        for (int i = 0; i < ITERS; i++) {
            voltrix_spmm_forward_cuda(d_Pointer1, d_hspa_packed,
                                      d_hspa,
                                      (int32_t *)d_hind, N, E, padded_K,
                                      d_B, d_C_vx, model, 0);
        }
        float ms = timer.tock() / ITERS;
        double gflops = 2.0 * E * padded_K / (ms * 1e6);

        if (ms < best_vx_ms) {
            best_vx_ms   = ms;
            best_model   = model;
            best_rmse    = rmse;
            best_gflops  = gflops;
        }
    }

    R.best_model = best_model;
    R.vx_ms      = best_vx_ms;
    R.vx_gflops  = best_gflops;
    R.vx_rmse    = best_rmse;

    /* --- cleanup ------------------------------------------------------ */
    cudaFree(d_nodePointer);  cudaFree(d_edgeList);
    cudaFree(d_blockPartition); cudaFree(d_edgeToColumn);
    cudaFree(d_edgeToRow);    cudaFree(d_Pointer1);
    cudaFree(d_edgeValues);
    cudaFree(d_hspa);         cudaFree(d_hind);
    cudaFree(d_hspa_packed);
    cudaFree(d_B);  cudaFree(d_C_cusp);  cudaFree(d_C_vx);
    cudaFree(d_row_f); cudaFree(d_col_f); cudaFree(d_val_f);
    return R;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <csr.csv> <K> [csr2.csv ...]\n", argv[0]);
        return 1;
    }

    int K = std::atoi(argv[2]);

    std::vector<std::string> csv_paths;
    csv_paths.push_back(argv[1]);
    for (int i = 3; i < argc; i++)
        csv_paths.push_back(argv[i]);

    print_table_header();

    for (const auto &csv_path : csv_paths) {
        std::string ds_name = extract_dataset_name(csv_path);
        VxRunResult R = run_benchmark(csv_path.c_str(), K);
        print_table_row(ds_name.c_str(), R);
    }

    return 0;
}
