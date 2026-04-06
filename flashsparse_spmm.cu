#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <mma.h>

using namespace nvcuda;

namespace {

constexpr int FS_BLK_H = 8;
constexpr int FS_BLK_W = 4;
constexpr int FS_BAL_PART = 32;
constexpr int FS_EPOCHS = 10;

struct PerfResult { float time_ms; float gflops; };
struct AccResult { double max_abs; double mean_abs; double rmse; };

struct FsPreprocessResult {
    int original_nodes = 0;
    int padded_nodes = 0;
    int num_edges = 0;
    int window_count = 0;
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<float> values;
    std::vector<int> balance_row_offsets;
    std::vector<int> balance_window_row;
    std::vector<int> balance_atomic;
    double preprocess_ms = 0.0;
};

struct FsRunResult {
    int num_nodes = 0;
    int num_edges = 0;
    int embedding_dim = 0;
    double preprocess_ms = 0.0;
    PerfResult cusparse{0, 0};
    PerfResult flashsparse{0, 0};
    AccResult flashsparse_acc{0, 0, 0};
    PerfResult flashsparse_bal{0, 0};
    AccResult flashsparse_bal_acc{0, 0, 0};
};

struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void Start() { cudaEventRecord(start); }
    void Stop() { cudaEventRecord(stop); }
    float Elapsed() {
        float elapsed = 0.0f;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

inline void cuda_check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

inline void cusparse_check(cusparseStatus_t err, const char *what) {
    if (err != CUSPARSE_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s failed with cuSPARSE status %d\n", what, (int)err);
        std::exit(1);
    }
}

static int round_up(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

static void load_csr_csv(const std::string &path,
                         std::vector<int> &rowPtr,
                         std::vector<int> &col,
                         std::vector<float> &vals) {
    std::fstream fin(path, std::ios::in);
    if (!fin.is_open()) {
        std::fprintf(stderr, "Error: cannot open %s\n", path.c_str());
        std::exit(1);
    }
    std::string line, word;

    std::getline(fin, line);
    std::stringstream ss1(line);
    while (std::getline(ss1, word, ',')) rowPtr.push_back(std::stoi(word));

    std::getline(fin, line);
    std::stringstream ss2(line);
    while (std::getline(ss2, word, ',')) col.push_back(std::stoi(word));

    if (std::getline(fin, line) && !line.empty()) {
        std::stringstream ss3(line);
        while (std::getline(ss3, word, ',')) vals.push_back(std::stof(word));
    }
    if (vals.empty()) vals.assign(col.size(), 1.0f);
}

static std::string extract_dataset_name(const std::string &path) {
    size_t slash = path.find_last_of('/');
    std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) base = base.substr(0, dot);
    return base;
}

static AccResult compare_results(const float *d_ref, const float *d_test, size_t count) {
    std::vector<float> h_ref(count), h_test(count);
    cuda_check(cudaMemcpy(h_ref.data(), d_ref, count * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy ref");
    cuda_check(cudaMemcpy(h_test.data(), d_test, count * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy test");

    double max_abs = 0.0, sum_abs = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double diff = static_cast<double>(h_test[i]) - static_cast<double>(h_ref[i]);
        double adiff = std::fabs(diff);
        if (adiff > max_abs) max_abs = adiff;
        sum_abs += adiff;
        sum_sq += diff * diff;
    }
    return {max_abs, sum_abs / static_cast<double>(count), std::sqrt(sum_sq / static_cast<double>(count))};
}

static PerfResult cusparse_spmm_reference(const std::vector<int> &rowPtr,
                                          const std::vector<int> &col,
                                          int num_rows,
                                          int num_cols,
                                          int embedding_dim,
                                          const float *d_X,
                                          float *d_C,
                                          int epochs) {
    int *d_rowPtr = nullptr, *d_col = nullptr;
    float *d_vals = nullptr;
    cuda_check(cudaMalloc(&d_rowPtr, rowPtr.size() * sizeof(int)), "cudaMalloc rowPtr");
    cuda_check(cudaMalloc(&d_col, col.size() * sizeof(int)), "cudaMalloc col");
    cuda_check(cudaMalloc(&d_vals, col.size() * sizeof(float)), "cudaMalloc vals");
    cuda_check(cudaMemcpy(d_rowPtr, rowPtr.data(), rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice), "copy rowPtr");
    cuda_check(cudaMemcpy(d_col, col.data(), col.size() * sizeof(int), cudaMemcpyHostToDevice), "copy col");

    std::vector<float> h_vals(col.size(), 1.0f);
    cuda_check(cudaMemcpy(d_vals, h_vals.data(), h_vals.size() * sizeof(float), cudaMemcpyHostToDevice), "copy vals");

    cusparseHandle_t handle;
    cusparse_check(cusparseCreate(&handle), "cusparseCreate");

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    cusparse_check(cusparseCreateCsr(&matA,
                                     num_rows,
                                     num_cols,
                                     static_cast<int64_t>(col.size()),
                                     d_rowPtr,
                                     d_col,
                                     d_vals,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F),
                   "cusparseCreateCsr");
    cusparse_check(cusparseCreateDnMat(&matB, num_cols, embedding_dim, embedding_dim,
                                       const_cast<float *>(d_X), CUDA_R_32F, CUSPARSE_ORDER_ROW),
                   "cusparseCreateDnMat B");
    cusparse_check(cusparseCreateDnMat(&matC, num_rows, embedding_dim, embedding_dim,
                                       d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW),
                   "cusparseCreateDnMat C");

    float alpha = 1.0f, beta = 0.0f;
    size_t buffer_size = 0;
    void *d_buffer = nullptr;
    cusparse_check(cusparseSpMM_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           matA,
                                           matB,
                                           &beta,
                                           matC,
                                           CUDA_R_32F,
                                           CUSPARSE_SPMM_ALG_DEFAULT,
                                           &buffer_size),
                   "cusparseSpMM_bufferSize");
    cuda_check(cudaMalloc(&d_buffer, buffer_size), "cudaMalloc cusparse buffer");

    for (int i = 0; i < 5; ++i) {
        cuda_check(cudaMemset(d_C, 0, (size_t)num_rows * embedding_dim * sizeof(float)), "warmup memset C");
        cusparse_check(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    matA,
                                    matB,
                                    &beta,
                                    matC,
                                    CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT,
                                    d_buffer),
                       "cusparseSpMM warmup");
    }
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize cusparse warmup");

    GpuTimer timer;
    timer.Start();
    for (int i = 0; i < epochs; ++i) {
        cuda_check(cudaMemset(d_C, 0, (size_t)num_rows * embedding_dim * sizeof(float)), "timed memset C");
        cusparse_check(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    matA,
                                    matB,
                                    &beta,
                                    matC,
                                    CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT,
                                    d_buffer),
                       "cusparseSpMM");
    }
    timer.Stop();

    float time_ms = timer.Elapsed() / epochs;
    float flop = static_cast<float>(col.size()) * static_cast<float>(embedding_dim) * 2.0f;
    float gflops = (flop * 1000.0f) / (time_ms * 1e9f);

    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroySpMat(matA);
    cusparseDestroy(handle);
    cudaFree(d_buffer);
    cudaFree(d_rowPtr);
    cudaFree(d_col);
    cudaFree(d_vals);

    return {time_ms, gflops};
}

static FsPreprocessResult preprocess_flashsparse(const std::vector<int> &rowPtr,
                                                 const std::vector<int> &col) {
    FsPreprocessResult result;
    result.original_nodes = static_cast<int>(rowPtr.size()) - 1;
    result.num_edges = static_cast<int>(col.size());
    result.padded_nodes = round_up(result.original_nodes, FS_BLK_H);
    result.window_count = result.padded_nodes / FS_BLK_H;

    std::vector<int> padded_rowPtr = rowPtr;
    padded_rowPtr.resize(result.padded_nodes + 1, rowPtr.back());

    result.row_offsets.assign(result.window_count + 1, 0);
    result.balance_row_offsets.assign(1, 0);

    auto start = std::chrono::high_resolution_clock::now();

    for (int win = 0; win < result.window_count; ++win) {
        int row_begin = win * FS_BLK_H;
        int row_end = std::min(row_begin + FS_BLK_H, result.original_nodes);
        int edge_begin = padded_rowPtr[row_begin];
        int edge_end = padded_rowPtr[std::min(row_begin + FS_BLK_H, result.padded_nodes)];
        if (edge_begin == edge_end) {
            result.row_offsets[win + 1] = result.row_offsets[win];
            continue;
        }

        std::vector<int> unique_cols(col.begin() + edge_begin, col.begin() + edge_end);
        std::sort(unique_cols.begin(), unique_cols.end());
        unique_cols.erase(std::unique(unique_cols.begin(), unique_cols.end()), unique_cols.end());

        int num_vector = static_cast<int>(unique_cols.size());
        int vector_start = static_cast<int>(result.col_indices.size());
        result.row_offsets[win + 1] = vector_start + num_vector;
        result.col_indices.insert(result.col_indices.end(), unique_cols.begin(), unique_cols.end());

        size_t values_base = result.values.size();
        result.values.resize(result.values.size() + static_cast<size_t>(num_vector) * FS_BLK_H, 0.0f);

        int block_num = (num_vector + FS_BLK_W - 1) / FS_BLK_W;
        int remaining_blocks = block_num;
        if (block_num == 0) {
            result.balance_window_row.push_back(win);
            result.balance_atomic.push_back(0);
            result.balance_row_offsets.push_back(result.balance_row_offsets.back());
        } else {
            int full_group_vectors = FS_BAL_PART * FS_BLK_W;
            bool single_group = block_num <= FS_BAL_PART;
            int consumed_vectors = 0;
            while (remaining_blocks > 0) {
                int group_blocks = std::min(remaining_blocks, FS_BAL_PART);
                int group_vectors = std::min(full_group_vectors, num_vector - consumed_vectors);
                result.balance_window_row.push_back(win);
                result.balance_atomic.push_back(single_group ? 0 : 1);
                result.balance_row_offsets.push_back(result.balance_row_offsets.back() + group_vectors);
                consumed_vectors += group_vectors;
                remaining_blocks -= group_blocks;
            }
        }

        for (int row = row_begin; row < row_end; ++row) {
            for (int eid = rowPtr[row]; eid < rowPtr[row + 1]; ++eid) {
                int mapped_col = static_cast<int>(std::lower_bound(unique_cols.begin(), unique_cols.end(), col[eid]) - unique_cols.begin());
                int tcblock_id = mapped_col / FS_BLK_W;
                int row_local = row - row_begin;
                int col_local = mapped_col % FS_BLK_W;
                int residue = num_vector % FS_BLK_W;
                size_t offset = 0;
                if (residue > 0 && mapped_col >= num_vector - residue) {
                    offset = values_base + static_cast<size_t>(tcblock_id) * FS_BLK_H * FS_BLK_W + static_cast<size_t>(row_local) * residue + col_local;
                } else {
                    offset = values_base + static_cast<size_t>(tcblock_id) * FS_BLK_H * FS_BLK_W + static_cast<size_t>(row_local) * FS_BLK_W + col_local;
                }
                result.values[offset] = 1.0f;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.preprocess_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

struct mmaDenseTile_tf32_v2_map {
    const float *values_;
    const int *column_idxs_;
    const int rhs_cols_;
    const int lane_id_;
    const int warpin_id;
    const int warp_id;
    const float2 *matrix_base_;
    float *dense_tile_;
    float *sparse_fragment_;

    __device__ __forceinline__ mmaDenseTile_tf32_v2_map(
        long row_offset_vec,
        const float *values,
        const int *column_idxs,
        int rhs_cols,
        int offset,
        int lane_id,
        const float *matrix,
        float *dense_tile,
        float *sparse_fragment)
        : values_((values + row_offset_vec * 8) + (lane_id & 31)),
          column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31) % 4)),
          rhs_cols_(rhs_cols),
          lane_id_(lane_id),
          warpin_id(lane_id & 31),
          warp_id(lane_id >> 5),
          matrix_base_(reinterpret_cast<const float2 *>(matrix + offset)),
          dense_tile_(dense_tile),
          sparse_fragment_(sparse_fragment) {}

    __device__ __forceinline__ void Fetch(int, int) {
        sparse_fragment_[0] = __ldg(values_);
        const long row_offsets_ = __ldg(column_idxs_);
        values_ += 32;
        column_idxs_ += 4;
        const int global_offset = (warp_id << 3) + (warpin_id / 4);
        const long offset = (row_offsets_ * rhs_cols_ / 2) + global_offset;
        float2 temp = __ldg(matrix_base_ + offset);
        dense_tile_[0] = temp.x;
        dense_tile_[1] = temp.y;
    }

    __device__ __forceinline__ void ResidueLoad(int, int, int residue) {
        int col_offset = warpin_id % 4;
        long row_offsets_ = -1;
        values_ -= (4 - residue) * (warpin_id / 4);
        if (col_offset < residue) {
            sparse_fragment_[0] = __ldg(values_);
            row_offsets_ = __ldg(column_idxs_);
        }
        const int global_offset = (warp_id << 3) + (warpin_id / 4);
        if (row_offsets_ >= 0) {
            const long offset = (row_offsets_ * rhs_cols_ / 2) + global_offset;
            float2 temp = __ldg(matrix_base_ + offset);
            dense_tile_[0] = temp.x;
            dense_tile_[1] = temp.y;
        } else {
            dense_tile_[0] = 0.0f;
            dense_tile_[1] = 0.0f;
        }
    }
};

struct mmaComputeUtils_tf32_v2 {
    uint32_t *rhs_fragment;
    float *output_fragment_;
    uint32_t *lhs_fragment;

    __device__ __forceinline__ mmaComputeUtils_tf32_v2(
        float *dense_tile,
        float *output_fragment,
        int,
        float *sparse_fragment)
        : rhs_fragment(reinterpret_cast<uint32_t *>(dense_tile)),
          output_fragment_(output_fragment),
          lhs_fragment(reinterpret_cast<uint32_t *>(sparse_fragment)) {}

    __device__ __forceinline__ void TileMAC() {
        asm volatile(
            "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), "r"(lhs_fragment[0]),
              "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));
    }

    __device__ __forceinline__ void TileMACResidue() { TileMAC(); }
};

struct mmaOutputTile_tf32_map {
    int warp_id;
    int warpin_id;
    int wrow_offset;
    int wcol_offset;
    const float *output_fragment_;
    float *output_matrix_;

    __device__ __forceinline__ mmaOutputTile_tf32_map(int lane_id, float *output_fragment)
        : warp_id(lane_id >> 5),
          warpin_id(lane_id & 31),
          wrow_offset((warpin_id >> 2) * 2),
          wcol_offset((warpin_id & 3) << 1),
          output_fragment_(output_fragment),
          output_matrix_(nullptr) {}

    __device__ __forceinline__ void Store(long m_index_vec, long column_offset,
                                          long cols, float *output_matrix,
                                          int rowEdge, int colEdge) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            long row = ((m_index_vec << 3) + wcol_offset + i);
            long col = column_offset + wrow_offset + (warp_id << 4);
            output_matrix_ = output_matrix + (row * cols) + col;
            if (row < rowEdge) {
                if (col < colEdge) *output_matrix_ = output_fragment_[i];
                if ((col + 1) < colEdge) *(output_matrix_ + 1) = output_fragment_[i + 2];
            }
        }
    }
};

template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_tf32_map(
    const int *__restrict__ row_offsets,
    const int *__restrict__ col_indices,
    const float *__restrict__ values,
    const float *__restrict__ rhs_matrix,
    float *__restrict__ output_matrix,
    int dimN,
    int dimM,
    long nOri,
    int mOri) {
    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;
    if ((dimN_index + ((lane_id / 32 + 1) * 16)) > dimN) return;
    if ((blockIdx.z * 200 + blockIdx.y) >= dimM) return;

    int m_index_vec = (blockIdx.z * 200) + blockIdx.y;
    int row_offset_vec = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;
    if (nonzeros == 0) return;

    float sparse_fragment[1] = {0.0f};
    float dense_fragment[2] = {0.0f, 0.0f};
    mmaDenseTile_tf32_v2_map dense_tile_loader(row_offset_vec, values, col_indices,
                                               nOri, dimN_index, lane_id, rhs_matrix,
                                               dense_fragment, sparse_fragment);
    float output_fragment[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    mmaComputeUtils_tf32_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    int steps = nonzeros >> 2;
    int residue = nonzeros % 4;
    for (int i = 0; i < steps; ++i) {
        dense_tile_loader.Fetch(nOri, dimN_index);
        __syncwarp();
        computer.TileMAC();
    }
    if (residue > 0) {
        dense_tile_loader.ResidueLoad(nOri, dimN_index, residue);
        __syncwarp();
        computer.TileMACResidue();
    }

    mmaOutputTile_tf32_map output_tile_storer(lane_id, output_fragment);
    output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix, mOri, nOri);
}

template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_tf32_balance(
    const int *__restrict__ row_offsets,
    const int *__restrict__ col_indices,
    const float *__restrict__ values,
    const int *__restrict__ t_window_row,
    const int *__restrict__ t_atomic,
    const float *__restrict__ rhs_matrix,
    float *__restrict__ output_matrix,
    int dimN,
    int parts_t,
    long nOri,
    int mOri,
    int splitk) {
    int m_index_vec = (blockIdx.z * splitk) + blockIdx.y;
    if (m_index_vec >= parts_t) return;

    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;
    int warp_id = lane_id >> 5;
    if ((dimN_index + ((warp_id + 1) * 16)) > dimN) return;
    int warpin_id = lane_id & 31;

    int row_offset_vec = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;
    if (nonzeros == 0) return;

    float sparse_fragment[1] = {0.0f};
    float dense_fragment[2] = {0.0f, 0.0f};
    mmaDenseTile_tf32_v2_map dense_tile_loader(row_offset_vec, values, col_indices,
                                               nOri, dimN_index, lane_id, rhs_matrix,
                                               dense_fragment, sparse_fragment);
    float output_fragment[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    mmaComputeUtils_tf32_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    int steps = nonzeros >> 2;
    int residue = nonzeros % 4;
    for (int i = 0; i < steps; ++i) {
        dense_tile_loader.Fetch(nOri, dimN_index);
        __syncwarp();
        computer.TileMAC();
    }
    if (residue > 0) {
        dense_tile_loader.ResidueLoad(nOri, dimN_index, residue);
        __syncwarp();
        computer.TileMACResidue();
    }

    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    int cur_t_atomic = __ldg(t_atomic + m_index_vec);
    int row = (cur_m_index_vec << 3) + (warpin_id % 4) * 2;
    int col = dimN_index + warp_id * 16 + (warpin_id / 4) * 2;
    if (row < mOri) {
        float *output_ptr = output_matrix + (row * nOri) + col;
        if (cur_t_atomic == 0) {
            if (col < nOri) *output_ptr = output_fragment[0];
            if ((col + 1) < nOri) *(output_ptr + 1) = output_fragment[2];
            if ((row + 1) < mOri) {
                output_ptr += nOri;
                if (col < nOri) *output_ptr = output_fragment[1];
                if ((col + 1) < nOri) *(output_ptr + 1) = output_fragment[3];
            }
        } else {
            if (col < nOri) atomicAdd(output_ptr, output_fragment[0]);
            if ((col + 1) < nOri) atomicAdd(output_ptr + 1, output_fragment[2]);
            if ((row + 1) < mOri) {
                output_ptr += nOri;
                if (col < nOri) atomicAdd(output_ptr, output_fragment[1]);
                if ((col + 1) < nOri) atomicAdd(output_ptr + 1, output_fragment[3]);
            }
        }
    }
}

static PerfResult flashsparse_tf32_map_forward(const int *d_row_offsets,
                                               const int *d_col_indices,
                                               const float *d_values,
                                               const float *d_X,
                                               float *d_C,
                                               int window_count,
                                               int embedding_dim,
                                               int original_nodes,
                                               int epochs) {
    int n1 = embedding_dim;
    if ((embedding_dim & 15) != 0) n1 = ((embedding_dim >> 4) + 1) << 4;
    int grid_x = (n1 >> 6) + 1;
    if (n1 % 64 == 0) grid_x -= 1;
    dim3 grid_dim(grid_x, 200, (window_count / 200) + 1);
    dim3 block_dim(128, 1, 1);

    size_t out_bytes = static_cast<size_t>(original_nodes) * embedding_dim * sizeof(float);
    for (int iter = 0; iter < 5; ++iter) {
        cuda_check(cudaMemset(d_C, 0, out_bytes), "flashsparse warmup memset");
        spmm_forward_cuda_kernel_tf32_map<64><<<grid_dim, block_dim>>>(
            d_row_offsets, d_col_indices, d_values, d_X, d_C,
            n1, window_count, embedding_dim, original_nodes);
    }
    cuda_check(cudaDeviceSynchronize(), "flashsparse warmup sync");

    GpuTimer timer;
    timer.Start();
    for (int iter = 0; iter < epochs; ++iter) {
        cuda_check(cudaMemset(d_C, 0, out_bytes), "flashsparse timed memset");
        spmm_forward_cuda_kernel_tf32_map<64><<<grid_dim, block_dim>>>(
            d_row_offsets, d_col_indices, d_values, d_X, d_C,
            n1, window_count, embedding_dim, original_nodes);
    }
    timer.Stop();
    cuda_check(cudaGetLastError(), "flashsparse kernel launch");

    float time_ms = timer.Elapsed() / epochs;
    float flop = static_cast<float>(d_row_offsets ? 1 : 1); (void)flop;
    return {time_ms, 0.0f};
}

static PerfResult flashsparse_tf32_balance_forward(const int *d_row_offsets,
                                                   const int *d_col_indices,
                                                   const float *d_values,
                                                   const int *d_t_window_row,
                                                   const int *d_t_atomic,
                                                   int parts_t,
                                                   const float *d_X,
                                                   float *d_C,
                                                   int embedding_dim,
                                                   int original_nodes,
                                                   int epochs) {
    int n1 = embedding_dim;
    if ((embedding_dim & 15) != 0) n1 = ((embedding_dim >> 4) + 1) << 4;
    int grid_x = (n1 >> 6) + 1;
    if (n1 % 64 == 0) grid_x -= 1;
    int splitk_t = (parts_t < 500000) ? 8 : (((parts_t / 1250000) + 1) * 20);
    dim3 grid_dim(grid_x, splitk_t, (parts_t / splitk_t) + 1);
    dim3 block_dim(128, 1, 1);

    size_t out_bytes = static_cast<size_t>(original_nodes) * embedding_dim * sizeof(float);
    for (int iter = 0; iter < 5; ++iter) {
        cuda_check(cudaMemset(d_C, 0, out_bytes), "flashsparse bal warmup memset");
        spmm_forward_cuda_kernel_tf32_balance<64><<<grid_dim, block_dim>>>(
            d_row_offsets, d_col_indices, d_values,
            d_t_window_row, d_t_atomic, d_X, d_C,
            n1, parts_t, embedding_dim, original_nodes, splitk_t);
    }
    cuda_check(cudaDeviceSynchronize(), "flashsparse bal warmup sync");

    GpuTimer timer;
    timer.Start();
    for (int iter = 0; iter < epochs; ++iter) {
        cuda_check(cudaMemset(d_C, 0, out_bytes), "flashsparse bal timed memset");
        spmm_forward_cuda_kernel_tf32_balance<64><<<grid_dim, block_dim>>>(
            d_row_offsets, d_col_indices, d_values,
            d_t_window_row, d_t_atomic, d_X, d_C,
            n1, parts_t, embedding_dim, original_nodes, splitk_t);
    }
    timer.Stop();
    cuda_check(cudaGetLastError(), "flashsparse bal kernel launch");

    return {timer.Elapsed() / epochs, 0.0f};
}

static FsRunResult run_flashsparse_dataset(const FsPreprocessResult &prep,
                                           int embedding_dim,
                                           const float *d_X,
                                           float *d_C,
                                           float *d_C_ref,
                                           int epochs,
                                           const std::vector<int> &orig_rowPtr,
                                           const std::vector<int> &orig_col) {
    FsRunResult out;
    out.num_nodes = prep.original_nodes;
    out.num_edges = prep.num_edges;
    out.embedding_dim = embedding_dim;
    out.preprocess_ms = prep.preprocess_ms;

    int *d_row_offsets = nullptr, *d_col_indices = nullptr;
    float *d_values = nullptr;
    cuda_check(cudaMalloc(&d_row_offsets, prep.row_offsets.size() * sizeof(int)), "cudaMalloc fs row_offsets");
    cuda_check(cudaMalloc(&d_col_indices, prep.col_indices.size() * sizeof(int)), "cudaMalloc fs col_indices");
    cuda_check(cudaMalloc(&d_values, prep.values.size() * sizeof(float)), "cudaMalloc fs values");
    cuda_check(cudaMemcpy(d_row_offsets, prep.row_offsets.data(), prep.row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice), "copy fs row_offsets");
    cuda_check(cudaMemcpy(d_col_indices, prep.col_indices.data(), prep.col_indices.size() * sizeof(int), cudaMemcpyHostToDevice), "copy fs col_indices");
    cuda_check(cudaMemcpy(d_values, prep.values.data(), prep.values.size() * sizeof(float), cudaMemcpyHostToDevice), "copy fs values");

    int *d_bal_row_offsets = nullptr, *d_t_window_row = nullptr, *d_t_atomic = nullptr;
    cuda_check(cudaMalloc(&d_bal_row_offsets, prep.balance_row_offsets.size() * sizeof(int)), "cudaMalloc fs bal row_offsets");
    cuda_check(cudaMalloc(&d_t_window_row, prep.balance_window_row.size() * sizeof(int)), "cudaMalloc fs bal rows");
    cuda_check(cudaMalloc(&d_t_atomic, prep.balance_atomic.size() * sizeof(int)), "cudaMalloc fs bal atomic");
    cuda_check(cudaMemcpy(d_bal_row_offsets, prep.balance_row_offsets.data(), prep.balance_row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice), "copy fs bal row_offsets");
    cuda_check(cudaMemcpy(d_t_window_row, prep.balance_window_row.data(), prep.balance_window_row.size() * sizeof(int), cudaMemcpyHostToDevice), "copy fs bal rows");
    cuda_check(cudaMemcpy(d_t_atomic, prep.balance_atomic.data(), prep.balance_atomic.size() * sizeof(int), cudaMemcpyHostToDevice), "copy fs bal atomic");

    out.cusparse = cusparse_spmm_reference(orig_rowPtr, orig_col,
                                           prep.original_nodes, prep.original_nodes,
                                           embedding_dim, d_X, d_C_ref, epochs);

    out.flashsparse = flashsparse_tf32_map_forward(d_row_offsets, d_col_indices, d_values,
                                                   d_X, d_C, prep.window_count,
                                                   embedding_dim, prep.original_nodes, epochs);
    out.flashsparse.gflops = (static_cast<float>(prep.num_edges) * embedding_dim * 2.0f * 1000.0f) /
                             (out.flashsparse.time_ms * 1e9f);
    out.flashsparse_acc = compare_results(d_C_ref, d_C,
                                          static_cast<size_t>(prep.original_nodes) * embedding_dim);

    out.flashsparse_bal = flashsparse_tf32_balance_forward(d_bal_row_offsets, d_col_indices, d_values,
                                                           d_t_window_row, d_t_atomic,
                                                           static_cast<int>(prep.balance_window_row.size()),
                                                           d_X, d_C, embedding_dim,
                                                           prep.original_nodes, epochs);
    out.flashsparse_bal.gflops = (static_cast<float>(prep.num_edges) * embedding_dim * 2.0f * 1000.0f) /
                                 (out.flashsparse_bal.time_ms * 1e9f);
    out.flashsparse_bal_acc = compare_results(d_C_ref, d_C,
                                              static_cast<size_t>(prep.original_nodes) * embedding_dim);

    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_bal_row_offsets);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);
    return out;
}

static void print_table_header() {
    std::printf("%-14s %8s %10s %4s | %9s %9s | %9s %9s %10s | %9s %9s %10s | %9s\n",
                "Dataset", "Nodes", "Edges", "K",
                "cuSP/ms", "cuSP/GF",
                "FS/ms", "FS/GF", "FS/RMSE",
                "Bal/ms", "Bal/GF", "Bal/RMSE",
                "Prep/ms");
    std::printf("%-14s %8s %10s %4s | %9s %9s | %9s %9s %10s | %9s %9s %10s | %9s\n",
                "--------------", "--------", "----------", "----",
                "---------", "---------",
                "---------", "---------", "----------",
                "---------", "---------", "----------",
                "---------");
}

static void print_table_row(const char *name, const FsRunResult &R) {
    std::printf("%-14s %8d %10d %4d | %9.4f %9.2f | %9.4f %9.2f %10.2e | %9.4f %9.2f %10.2e | %9.2f\n",
                name, R.num_nodes, R.num_edges, R.embedding_dim,
                R.cusparse.time_ms, R.cusparse.gflops,
                R.flashsparse.time_ms, R.flashsparse.gflops, R.flashsparse_acc.rmse,
                R.flashsparse_bal.time_ms, R.flashsparse_bal.gflops, R.flashsparse_bal_acc.rmse,
                R.preprocess_ms);
}

} // namespace

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <csr.csv> <embedding_dim> [<csr2.csv> ...]\n", argv[0]);
        return 1;
    }

    int embedding_dim = std::atoi(argv[2]);
    if (embedding_dim <= 0 || (embedding_dim & 1)) {
        std::fprintf(stderr, "Embedding dim must be positive and even (got %d).\n", embedding_dim);
        return 1;
    }

    std::vector<std::string> csv_paths;
    csv_paths.push_back(argv[1]);
    for (int i = 3; i < argc; ++i) csv_paths.push_back(argv[i]);

    print_table_header();

    for (const auto &csv_path : csv_paths) {
        std::vector<int> rowPtr, col;
        std::vector<float> vals;
        load_csr_csv(csv_path, rowPtr, col, vals);

        FsPreprocessResult prep = preprocess_flashsparse(rowPtr, col);

        std::vector<float> h_X(static_cast<size_t>(prep.original_nodes) * embedding_dim);
        std::srand(42);
        for (auto &v : h_X) v = 2.0f * std::rand() / static_cast<float>(RAND_MAX) - 1.0f;

        float *d_X = nullptr, *d_C = nullptr, *d_C_ref = nullptr;
        size_t matrix_bytes = static_cast<size_t>(prep.original_nodes) * embedding_dim * sizeof(float);
        cuda_check(cudaMalloc(&d_X, matrix_bytes), "cudaMalloc X");
        cuda_check(cudaMalloc(&d_C, matrix_bytes), "cudaMalloc C");
        cuda_check(cudaMalloc(&d_C_ref, matrix_bytes), "cudaMalloc C_ref");
        cuda_check(cudaMemcpy(d_X, h_X.data(), matrix_bytes, cudaMemcpyHostToDevice), "copy X");
        cuda_check(cudaMemset(d_C, 0, matrix_bytes), "memset C");
        cuda_check(cudaMemset(d_C_ref, 0, matrix_bytes), "memset C_ref");

        FsRunResult R = run_flashsparse_dataset(prep, embedding_dim, d_X, d_C, d_C_ref,
                                                FS_EPOCHS, rowPtr, col);
        std::string ds_name = extract_dataset_name(csv_path);
        print_table_row(ds_name.c_str(), R);

        cudaFree(d_X);
        cudaFree(d_C);
        cudaFree(d_C_ref);
    }

    return 0;
}
