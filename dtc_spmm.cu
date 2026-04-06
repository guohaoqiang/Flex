///////////////////////////////////////////////////////////////////////////////
// DTC-SpMM Kernel Extraction for Flex Project
// Extracted from: https://github.com/HPMLL/DTC-SpMM_ASPLOS24
// Paper: "DTC-SpMM: Bridging the Gap in Accelerating General Sparse Matrix
//         Multiplication with Tensor Cores" (ASPLOS'24)
//
// Standalone CUDA baseline — no PyTorch dependency.
///////////////////////////////////////////////////////////////////////////////

#include "dtc_spmm.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <vector>

//////////////////////////////////////////////////////////////////////
/// Result structures for tabular output
//////////////////////////////////////////////////////////////////////

struct SpmmResult  { float time_ms; float gflops; };
struct AccResult   { double max_abs; double mean_abs; double rmse; };

struct DtcRunResult {
    int    num_nodes, num_edges, embedding_dim;
    int    tc_blocks;
    double preprocess_sec;
    SpmmResult  cusparse;
    SpmmResult  dtc;
    AccResult   dtc_acc;
    SpmmResult  dtc_bal;
    AccResult   dtc_bal_acc;
    const char *dtc_plan;
    const char *bal_plan;
};

//////////////////////////////////////////////////////////////////////
/// Thrust-replacement utilities (avoid CUB/thrust due to CUDA 13.2
/// + GCC 14 overload ambiguity in cub::block_load_to_shared)
//////////////////////////////////////////////////////////////////////

/// Simple kernel to fill a float array with a constant value.
__global__ void dtc_fill_float_kernel(float *arr, int n, float val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) arr[tid] = val;
}

/// Fill device float array with a constant value (replaces thrust::fill).
static inline void dtc_fill_float(float *d_arr, int n, float val) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dtc_fill_float_kernel<<<blocks, threads>>>(d_arr, n, val);
}

/// In-place inclusive prefix sum on a device int array.
/// Copies to host, scans, copies back (arrays are small enough).
static inline void dtc_inclusive_scan_int(int *d_arr, int n) {
    std::vector<int> h(n);
    cudaMemcpy(h.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 1; i < n; i++) h[i] += h[i - 1];
    cudaMemcpy(d_arr, h.data(), n * sizeof(int), cudaMemcpyHostToDevice);
}

//////////////////////////////////////////////////////////////////////
/// Timer
//////////////////////////////////////////////////////////////////////

struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;
  GpuTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
  ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
  void Start() { cudaEventRecord(start); }
  void Stop()  { cudaEventRecord(stop); }
  float Elapsed() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

//////////////////////////////////////////////////////////////////////
/// Preprocessing GPU Kernels
//////////////////////////////////////////////////////////////////////

__global__ void roundup_to_multiple_of_eight(int *input, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    int rounded_value = ((input[tid] + 7) / 8) * 8;
    input[tid] = rounded_value;
  }
}

__global__ void get_padding_tileid_kernel(int *ori_offset, uint8_t *ori_tileid,
                                          int *padded_offset,
                                          uint8_t *padded_tileid, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    int s = ori_offset[tid];
    int e = ori_offset[tid + 1];
    int s1 = padded_offset[tid];
    for (int i = 0; i < e - s; i++) {
      padded_tileid[s1 + i] = ori_tileid[s + i];
    }
  }
}

__global__ void fill_edgeToRow(int *edgeToRow, int *nodePointer,
                               int num_nodes) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int nid = tid / 32;
  int laneid = tid % 32;
  // check a valid node range.
  if (nid < num_nodes) {
#pragma unroll
    for (int eid = nodePointer[nid] + laneid; eid < nodePointer[nid + 1];
         eid += 32) {
      edgeToRow[eid] = nid;
    }
  }
}
/*Generate segment*/
__global__ void fill_segment(int *nodePointer, int *seg_out, int blockSize_h,
                             int blockSize_w, int num_nodes) {
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = nodePointer[winId * blockSize_h];
  unsigned block_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = block_end - block_start;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  for (unsigned idx = tid; idx < num_window_edges; idx += threadPerBlock) {
    seg_out[block_start + idx] = winId;
  }
}
void fill_segment_cuda(int *nodePointer, int *seg_out, int blockSize_h,
                       int blockSize_w, int num_nodes) {
  int block_size = 512;
  int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
  fill_segment<<<window_count, block_size>>>(nodePointer, seg_out, blockSize_h,
                                             blockSize_w, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/*Generate TCblock_rowid*/
__global__ void generate_tcblock_rowid(int *rowwindow_offset,
                                       int *tcblock_rowid,
                                       int num_row_windows) {
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = rowwindow_offset[winId];
  unsigned block_end = rowwindow_offset[min(winId + 1, num_row_windows)];
  unsigned num_blocks = block_end - block_start;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  for (unsigned idx = tid; idx < num_blocks; idx += threadPerBlock) {
    tcblock_rowid[block_start + idx] = winId;
  }
}
void generate_tcblock_rowid_cuda(int *rowwindow_offset, int *tcblock_rowid,
                                 int num_row_windows) {
  int block_size = 512;
  int window_count = num_row_windows;
  generate_tcblock_rowid<<<window_count, block_size>>>(
      rowwindow_offset, tcblock_rowid, num_row_windows);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/* Generate edge2column*/
__device__ __forceinline__ int binarysearch(int *arr, int size, int target) {
  int left = 0;
  int right = size - 1;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (arr[mid] == target) {
      while (mid > 0 && arr[mid - 1] == target) {
        mid--;
      }
      return mid;
    } else if (arr[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return -1;
}
__device__ __forceinline__ void inplace_deduplication(int *array, int length,
                                                      int *loc) {
  int cur = 1;
  while (cur < length) {
    if (array[cur] != array[cur - 1]) {
      (*loc)++;
      array[(*loc)] = array[cur];
    }
    cur++;
  }
}
__global__ void generate_edgetocolumn(int *nodePointer, int *edgelist,
                                      int *edgelist_sort, int *edgetocol,
                                      int *blockpartition, int *blocknum,
                                      int blockSize_h, int blockSize_w,
                                      int num_nodes) {
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = nodePointer[winId * blockSize_h];
  unsigned block_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = block_end - block_start;
  if (num_window_edges == 0)
    return;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  int *start = edgelist_sort + block_start;
  int size = 0;
  inplace_deduplication(start, num_window_edges, &size);
  int num = (size + blockSize_w) / blockSize_w;
  atomicAdd(blocknum, num);
  blockpartition[winId] = num;
  for (unsigned idx = block_start; idx < block_end; idx += 1) {
    int index = binarysearch(start, size + 1, edgelist[idx]);
    edgetocol[idx] = index;
  }
}
void generate_edgetocolumn_cuda(int *nodePointer, int *edgelist,
                                int *edgelist_sort, int *edgetocol,
                                int *blockpartition, int *blocknum,
                                int blockSize_h, int blockSize_w,
                                int num_nodes) {
  int block_size = 1;
  int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
  int block_size1 = 128;
  int block_count1 = (window_count + 127) / 128;
  generate_edgetocolumn<<<window_count, block_size>>>(
      nodePointer, edgelist, edgelist_sort, edgetocol, blockpartition, blocknum,
      blockSize_h, blockSize_w, num_nodes);
  // generate_edgetocolumn_v1<<< window_count, block_size >>> (nodePointer,
  // edgelist, edgelist_sort, edgetocol, blockpartition, blocknum, blockSize_h,
  // blockSize_w, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/*Generate TC offset, tileid and AtoB*/
__global__ void generate_tcoffset_id_atob(
    int *nodePointer, int *rowwindow_offset, int *edgeToColumn, int *edgeToRow,
    int *edgeList, int *tcblock_offset, uint8_t *tcblocktile_id,
    int *sparseatob, int max_block, int num_nodes, int blockSize_h,
    int blockSize_w, int num_row_windows) {
  extern __shared__ int pos_ptr[];
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = rowwindow_offset[winId];
  unsigned block_end = rowwindow_offset[min(winId + 1, num_row_windows)];
  unsigned num_blocks = block_end - block_start;
  if (num_blocks == 0) {
    return;
  }
  int *tcblock_offset_ptr = pos_ptr + num_blocks;
  int *tcblock_offset_global_ptr = tcblock_offset + block_start;
  int *tcblock_nnz_ptr = pos_ptr + num_blocks + 1;
  unsigned element_start = nodePointer[winId * blockSize_h];
  unsigned element_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = element_end - element_start;
  if (num_window_edges == 0) {
    return;
  }
  for (int i = 0; i < 2 * num_blocks + 1; i++) {
    pos_ptr[i] = 0;
  }
  for (unsigned e_index = element_start; e_index < element_end; e_index++) {
    unsigned col = edgeToColumn[e_index]; // new col
    tcblock_nnz_ptr[col / blockSize_w]++;
  }
  for (int i = 0; i < num_blocks; i++) {
    tcblock_offset_global_ptr[i] = tcblock_nnz_ptr[i];
  }
  auto tileid = tcblocktile_id + element_start;
  auto sparse_AToB = sparseatob + block_start * blockSize_w;
  for (int i = 0; i < num_blocks; i++) {
    tcblock_nnz_ptr[i] += tcblock_nnz_ptr[i - 1];
  }
  for (unsigned e_index = element_start; e_index < element_end; e_index++) {
    unsigned col = edgeToColumn[e_index]; // new col
    unsigned tcblock_id = col / blockSize_w;
    unsigned row_local = edgeToRow[e_index] % blockSize_h;
    unsigned col_local = col % blockSize_w;
    tileid[tcblock_offset_ptr[tcblock_id] + pos_ptr[tcblock_id]] =
        (uint8_t)(row_local * blockSize_w + col_local);
    sparse_AToB[tcblock_id * blockSize_w + col_local] = edgeList[e_index];
    pos_ptr[tcblock_id]++;
  }
}
void generate_tcoffset_id_atob_cuda(int *nodePointer, int *rowwindow_offset,
                                    int *edgeToColumn, int *edgeToRow,
                                    int *edgeList, int *tcblock_offset,
                                    uint8_t *tcblock_tileid, int *sparseatob,
                                    int max_block, int num_nodes,
                                    int blockSize_h, int blockSize_w,
                                    int num_row_windows) {
  int block_size = 1;
  int window_count = num_row_windows;
  const int dynamic_shared_size = (2 * max_block + 1) * sizeof(int);
  // (quiet — result printed in table)
  if (dynamic_shared_size > 98304) {
    int maxbytes = 131072; // 96 KB
    cudaFuncSetAttribute(generate_tcoffset_id_atob,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  } else if (dynamic_shared_size > 65536) {
    int maxbytes = 98304; // 96 KB
    cudaFuncSetAttribute(generate_tcoffset_id_atob,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  } else if (dynamic_shared_size > 32768) {
    int maxbytes = 65536; // 128 KB
    cudaFuncSetAttribute(generate_tcoffset_id_atob,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  }
  generate_tcoffset_id_atob<<<window_count, block_size, dynamic_shared_size>>>(
      nodePointer, rowwindow_offset, edgeToColumn, edgeToRow, edgeList,
      tcblock_offset, tcblock_tileid, sparseatob, max_block, num_nodes,
      blockSize_h, blockSize_w, num_row_windows);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}
void padding_up_8(int *input, int size) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  roundup_to_multiple_of_eight<<<blocksPerGrid, threadsPerBlock>>>(input, size);
}
void get_padding_tileid(int *ori_offset, uint8_t *ori_tileid,
                        int *padded_offset, uint8_t *padded_tileid, int size) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  get_padding_tileid_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      ori_offset, ori_tileid, padded_offset, padded_tileid, size);
}


void fill_edgeToRow_cuda(int *edgeToRow, int *nodePointer, int num_nodes) {
  int wrap_size = 32;
  int block_size = 1024;
  int grid_size = (num_nodes * wrap_size + block_size - 1) / block_size;
  fill_edgeToRow<<<grid_size, block_size>>>(edgeToRow, nodePointer, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}




//////////////////////////////////////////////////////////////////////
/// CPU Preprocessing (from TC-GNN, adapted)
/// No torch dependency — pure CPU.
//////////////////////////////////////////////////////////////////////

static std::map<unsigned, unsigned>
inplace_deduplication_cpu(unsigned *array, unsigned length) {
  int loc = 0, cur = 1;
  std::map<unsigned, unsigned> nb2col;
  nb2col[array[0]] = 0;
  while ((unsigned)cur < length) {
    if (array[cur] != array[cur - 1]) {
      loc++;
      array[loc] = array[cur];
      nb2col[array[cur]] = loc;
    }
    cur++;
  }
  return nb2col;
}

static int cpu_preprocess(
    const int *edgeList, const int *nodePointer,
    int num_nodes, int blockSize_h, int blockSize_w,
    int *blockPartition, int *edgeToColumn, int *edgeToRow)
{
    unsigned block_counter = 0;
    for (unsigned nid = 0; nid < (unsigned)num_nodes; nid++) {
        for (int eid = nodePointer[nid]; eid < nodePointer[nid+1]; eid++)
            edgeToRow[eid] = nid;
    }
    for (unsigned iter = 0; iter < (unsigned)num_nodes + 1; iter += blockSize_h) {
        unsigned windowId = iter / blockSize_h;
        unsigned block_start = nodePointer[iter];
        unsigned block_end = nodePointer[std::min(iter + (unsigned)blockSize_h, (unsigned)num_nodes)];
        unsigned num_window_edges = block_end - block_start;
        if (num_window_edges == 0) {
            blockPartition[windowId] = 0;
            continue;
        }
        unsigned *neighbor_window = (unsigned *)malloc(num_window_edges * sizeof(unsigned));
        memcpy(neighbor_window, &edgeList[block_start], num_window_edges * sizeof(unsigned));
        std::sort(neighbor_window, neighbor_window + num_window_edges);
        std::map<unsigned, unsigned> clean_edges2col =
            inplace_deduplication_cpu(neighbor_window, num_window_edges);
        blockPartition[windowId] = (clean_edges2col.size() + blockSize_w - 1) / blockSize_w;
        block_counter += blockPartition[windowId];
        for (unsigned e_index = block_start; e_index < block_end; e_index++) {
            unsigned eid = edgeList[e_index];
            edgeToColumn[e_index] = clean_edges2col[eid];
        }
        free(neighbor_window);
    }
    // (quiet — result printed in table)
    return block_counter;
}



//////////////////////////////////////////////////////////////////////
/// Forward SpMM Kernels
//////////////////////////////////////////////////////////////////////

#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


// ===== float_nonsplit: with_value_double_buffer =====

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) + wid * BLK_H;
	unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	 
		// if (tid < BLK_W) {
		//   sparse_AToX_index[tid] = numNodes + 1;
		// }
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[(int)TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;

		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));
	
			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));
	
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
			source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));
	
			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
		}
	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   
	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  int id_local = (((int)TCblocktile_id[eIdx])<<2);
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(&valuesA[eIdx]));	  
	    }
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(&sparse_AToX_idx[sparse_AToX_idx_start + tid]));	
		}

		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[2]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[3]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );

	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));

		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[2]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[3]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );


	uint32_t o_off1 = bid * BLK_H * embedding_dim + wid * BLK_H;
	uint32_t o_off2 = o_off1 + 8;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
		uint32_t off = row_d * embedding_dim + col_d;
		output[o_off1 + off] = frag_D[i];
		output[o_off2 + off] = frag_D[i + 4];
	}
}



// ===== float_split: with_value_double_buffer_split =====

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_split(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	int off_y = (blockIdx.y << 7);								
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned off = wid * BLK_W * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) + wid * BLK_H + off_y;
	unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = (numNodes + 1) * embedding_dim;
		}
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[tid] * embedding_dim;	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;

		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off];
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));
	
			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));
	
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1];
			source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));
	
			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			else
			  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
		}
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   
	    if (tid < BLK_W) {
		  sparse_AToX_index[(smem_sel_next << 3) + tid] = (numNodes + 1) * embedding_dim;
	    }
	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  int id_local = (((int)TCblocktile_id[eIdx])<<2);
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
	    }
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx+sparse_AToX_idx_start+tid));	
		}


		// asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		// asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		// asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		// asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[2]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[3]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );

	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off];
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));

		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1];
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[2]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[3]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );




	uint32_t o_off1 = bid * BLK_H * embedding_dim + wid * BLK_H + off_y;
	uint32_t o_off2 = o_off1 + 8;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
		uint32_t off = row_d * embedding_dim + col_d;
		output[o_off1 + off] = frag_D[i];
		output[o_off2 + off] = frag_D[i + 4];
	}
}



// ===== float2_nonsplit: with_value_double_buffer_float2 =====

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float2(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 	// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *input,		                        // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned off = wid * BLK_W * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 2 + wid * BLK_H;
	// unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	  
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
			}
		}

	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   
	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  int id_local = (((int)TCblocktile_id[eIdx])<<2);
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
	    }
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx + sparse_AToX_idx_start + tid));	
		}


		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[2]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[3]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );

	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[2]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[3]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * BLK_H;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group<<2) + ((i & 0x1)<<1);
		uint32_t off = row_d * embedding_dim + col_d;
		output[o_off + off] = frag_D[i];
		output[o_off + off + 1] = frag_D[i + 4];
	}
}



// ===== float2_split: with_value_double_buffer_float2_split =====

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float2_split(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	int off_y = (blockIdx.y << 6);
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned off = wid * BLK_W * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 2 + wid * BLK_H + off_y;
	// unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	  
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
			}
		}

	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   
	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  int id_local = (((int)TCblocktile_id[eIdx])<<2);
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
	    }
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx + sparse_AToX_idx_start + tid));	
		}


		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[2]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[3]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );

	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[2]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[3]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * BLK_H + off_y;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group<<2) + ((i & 0x1)<<1);
		uint32_t off = row_d * embedding_dim + col_d;
		output[o_off + off] = frag_D[i];
		output[o_off + off + 1] = frag_D[i + 4];
	}
}



// ===== float4_nonsplit: with_value_double_buffer_float4 =====

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / 32;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 4 + wid * 32;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[8]; // 8 * 8 * 2  / 32 = 4
	float frag_D[16] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j - lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
			unsigned source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
			}

		}

	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   

	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  int id_local = (((int)TCblocktile_id[eIdx])<<2);
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
	    }
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx+sparse_AToX_idx_start+tid));	
		}

		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[4]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[5]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );
		asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[2]), "r"(frag_B[6]), 
            "f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[3]), "r"(frag_B[7]), 
            "f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
        );
	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[4]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[5]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[2]), "r"(frag_B[6]), 
		  "f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[3]), "r"(frag_B[7]), 
		  "f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
	  );

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * 32;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group << 3) + ((i & 0x1)<<2);
		uint32_t off = row_d * embedding_dim + col_d;
		uint32_t off_set = o_off + off;
		output[off_set] = frag_D[i];
		output[off_set + 1] = frag_D[i + 4];
		output[off_set + 2] = frag_D[i + 8];
		output[off_set + 3] = frag_D[i + 12];
	}
}



// ===== float4_split: with_value_double_buffer_float4_split =====

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split(
	const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	int off_y = (blockIdx.y << 7);
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / 32;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 4 + wid * 32 + off_y;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[8]; // 8 * 8 * 2  / 32 = 4
	float frag_D[16] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j - lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
			unsigned source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
			}

		}

	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   

	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  int id_local = (((int)TCblocktile_id[eIdx])<<2);
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
	    }
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx+sparse_AToX_idx_start+tid));	
		}

		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[4]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[5]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );
		asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[2]), "r"(frag_B[6]), 
            "f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[3]), "r"(frag_B[7]), 
            "f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
        );
	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim; // TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[4]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[5]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[2]), "r"(frag_B[6]), 
		  "f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[3]), "r"(frag_B[7]), 
		  "f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
	  );

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * 32 + off_y;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group << 3) + ((i & 0x1)<<2);
		uint32_t off = row_d * embedding_dim + col_d;
		uint32_t off_set = o_off + off;
		output[off_set] = frag_D[i];
		output[off_set + 1] = frag_D[i + 4];
		output[off_set + 2] = frag_D[i + 8];
		output[off_set + 3] = frag_D[i + 12];
	}
}



// ===== balance: strict_balance_withv =====

__global__ void spmm_forward_cuda_kernel_improved_ptx_uint8_v1_strict_balance_withv(
	const int *__restrict__ TCblock_rowid, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA,
	const int tc_count,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	const float *__restrict__ input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = bid * TCBLOCK_PER_WARP;
	const unsigned hb = min((bid + 1) * TCBLOCK_PER_WARP, tc_count);
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	__shared__ int tc_rowid[TCBLOCK_PER_WARP];
	unsigned wid_BLK_H = wid * BLK_H;
	unsigned off = wid_BLK_H * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) + wid_BLK_H;
	unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	#pragma unroll
	for (unsigned idx = tid; idx < TCBLOCK_PER_WARP; idx += threadPerBlock) {
	  int ptr = lb + idx;
	  if (ptr < hb) {
		tc_rowid[idx] = __ldg(TCblock_rowid + ptr);
	  }
	}
	__syncthreads();
	unsigned former_row_id = tc_rowid[0];
	unsigned current_rid = former_row_id;
	for (unsigned j = lb; j < hb; j++) {
	  current_rid = tc_rowid[j - lb];
	  unsigned eIdx_start = TCblock_offset[j];			
	  unsigned eIdx_end = TCblock_offset[j + 1];
	  unsigned sparse_AToX_idx_start = j * BLK_W;	 
	  if (current_rid != former_row_id) {
		uint32_t o_off1 = former_row_id * BLK_H * embedding_dim + wid_BLK_H;
		uint32_t o_off2 = o_off1 + 8;
		if (wid < dimTileNum)
		#pragma unroll
		for(int i = 0; i < 4; i++) {
			uint32_t row_d = 0;
			if( i < 2 ) {
				row_d = group_id;
			} else {
				row_d = group_id + 8;
			}
			uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
			uint32_t off = row_d * embedding_dim + col_d;
			atomicAdd(output + o_off1 + off, frag_D[i]);
			atomicAdd(output + o_off2 + off, frag_D[i + 4]);
			frag_D[i] = 0.0;
			frag_D[i + 4] = 0.0;
		}
		former_row_id = current_rid;
	  }
	  #pragma unroll
	  for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		sparse_A[idx] = 0.0;
	  }
	  if (tid < BLK_W) {
		sparse_AToX_index[tid] = (numNodes + 1) * embedding_dim;
	  }
	  __syncthreads();
	  #pragma unroll
	  for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		sparse_A[(int)TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	
	  }
	  #pragma unroll
	  for (unsigned eIdx = sparse_AToX_idx_start + tid; eIdx < sparse_AToX_idx_start + BLK_W; eIdx += threadPerBlock) {
		sparse_AToX_index[tid] = sparse_AToX_idx[eIdx] * embedding_dim;	
	  }
	  __syncthreads();
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[sparse_A_idx]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[sparse_A_idx1]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[sparse_A_idx2]));
	  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[sparse_A_idx3]));
	  __syncthreads();
	  if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[dense_rowIdx_off];						// TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;

		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[0]), "r"(frag_A[1]), 
			  "r"(frag_B[0]), 
			  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
		);
		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[0]), "r"(frag_A[1]),
			  "r"(frag_B[1]), 
			  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
		);

		dense_rowIdx = sparse_AToX_index[dense_rowIdx_off1];						// TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[2]), "r"(frag_A[3]), 
			  "r"(frag_B[2]), 
			  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
		);
		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		else
		  asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
		asm volatile(
			"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[2]), "r"(frag_A[3]), 
			  "r"(frag_B[3]), 
			  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
		  );
	  }
	}
	uint32_t o_off1 = current_rid * BLK_H * embedding_dim + wid_BLK_H;
	uint32_t o_off2 = o_off1 + 8;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
		uint32_t off = row_d * embedding_dim + col_d;
		atomicAdd(output + o_off1 + off, frag_D[i]);
		atomicAdd(output + o_off2 + off, frag_D[i + 4]);
	}
}



// ===== balance: float4_split_balance =====

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split_balance(
	const int *__restrict__ TCblock_rowid, 		// offset of each row window.
	const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
	const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
	const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
	const float *__restrict__ valuesA, 		
	const int tc_count,
	const int numNodes,
	const int numEdges,
	const int embedding_dim,				    // embedding dimension.
	float *input,		    // input feature matrix.
	float *output							    // output feature matrix.
) {
    int bid = blockIdx.x;								// block_index == row_window_index
	int off_y = (blockIdx.y << 7);
	const unsigned lb = bid * TCBLOCK_PER_WARP;
	const unsigned hb = min((bid + 1) * TCBLOCK_PER_WARP, tc_count);
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / 32;              // number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	__shared__ int tc_rowid[TCBLOCK_PER_WARP];
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 4 + wid * 32 + off_y;
	uint32_t group_id = (laneid >> 2);
    uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
    uint32_t frag_B[8]; // 8 * 8 * 2  / 32 = 4
	float frag_D[16] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	#pragma unroll
	for (unsigned idx = tid; idx < TCBLOCK_PER_WARP; idx += threadPerBlock) {
	  int ptr = lb + idx;
	  if (ptr < hb) {
		tc_rowid[idx] = __ldg(TCblock_rowid + ptr);
	  }
	}
	__syncthreads();
	unsigned former_row_id = tc_rowid[0];
	unsigned current_rid = former_row_id;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
// pre loop
    {
		unsigned sparse_AToX_idx_start = lb * BLK_W;	
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];  // set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
		  sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
        int smem_sel_next = ( (j - lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim; // TC_block_col to dense_tile_row.
			unsigned source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim; // TC_block_col to dense_tile_row.
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
			}

		}

	    eIdx_start = TCblock_offset[j];			
	    eIdx_end = TCblock_offset[j+1];
	    unsigned sparse_AToX_idx_start = j * BLK_W;	   

	    #pragma unroll
	    for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
		  sparse_A[(smem_sel_next << 7) + idx] = 0.0;
	    }
	    __syncthreads();
	    #pragma unroll
	    for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
		  int id_local = (((int)TCblocktile_id[eIdx])<<2);
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
	    }
		if (tid < BLK_W) {	
		  asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx+sparse_AToX_idx_start+tid));	
		}

		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[0]), "r"(frag_B[4]), 
            "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[1]), "r"(frag_B[5]), 
            "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
        );
		asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[2]), "r"(frag_B[6]), 
            "f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
        );
	    asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
          : "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
            "r"(frag_B[3]), "r"(frag_B[7]), 
            "f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
        );


		current_rid = tc_rowid[j - lb];
		if (current_rid != former_row_id) {
			uint32_t o_off = former_row_id * BLK_H * embedding_dim + wid * 32 + off_y;
			if (wid < dimTileNum)
			#pragma unroll
			for(int i = 0; i < 4; i++) {
				uint32_t row_d = 0;
				if( i < 2 ) {
					row_d = group_id;
				} else {
					row_d = group_id + 8;
				}
				uint32_t col_d = (tid_in_group << 3) + ((i & 0x1)<<2);
				uint32_t off = row_d * embedding_dim + col_d;
				uint32_t off_set = o_off + off;
				atomicAdd(output + off_set, frag_D[i]);
				atomicAdd(output + off_set + 1, frag_D[i + 4]);
				atomicAdd(output + off_set + 2, frag_D[i + 8]);
				atomicAdd(output + off_set + 3, frag_D[i + 12]);
				frag_D[i] = 0.0;
				frag_D[i + 4] = 0.0;
				frag_D[i + 8] = 0.0;
				frag_D[i + 12] = 0.0;
			}
			former_row_id = current_rid;
		}

	    asm ("cp.async.commit_group;\n"::);
	    asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}
//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[0]), "r"(frag_B[4]), 
		  "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[1]), "r"(frag_B[5]), 
		  "f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[2]), "r"(frag_B[6]), 
		  "f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
	  );
	  asm volatile(
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
		: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
		  "r"(frag_B[3]), "r"(frag_B[7]), 
		  "f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
	  );

	uint32_t o_off = current_rid * BLK_H * embedding_dim + wid * 32 + off_y;
	if (wid < dimTileNum)
	#pragma unroll
	for(int i = 0; i < 4; i++) {
		uint32_t row_d = 0;
		if( i < 2 ) {
			row_d = group_id;
		} else {
			row_d = group_id + 8;
		}
		uint32_t col_d = (tid_in_group << 3) + ((i & 0x1)<<2);
		uint32_t off = row_d * embedding_dim + col_d;
		uint32_t off_set = o_off + off;
		atomicAdd(output + off_set, frag_D[i]);
		atomicAdd(output + off_set + 1, frag_D[i + 4]);
		atomicAdd(output + off_set + 2, frag_D[i + 8]);
		atomicAdd(output + off_set + 3, frag_D[i + 12]);
	}
}




//////////////////////////////////////////////////////////////////////
/// Dispatch Functions (no PyTorch)
//////////////////////////////////////////////////////////////////////

static SpmmResult dtc_spmm_forward(
    int *Rowwindow_offset,
    uint8_t *TCblocktile_id,
    int *TCblock_offset,
    int *sparse_AToX_idx,
    int num_row_windows,
    int num_nodes,
    int num_edges,
    int embedding_dim,
    float *input,
    float *output,
    const char *exeplan)
{
    const int WARPperBlock = embedding_dim / BLK_H;
    const int WARPperBlock1 = embedding_dim / 32;
    dim3 grid(num_row_windows, 1, 1);
    dim3 block(WARP_SIZE, WARPperBlock, 1);
    dim3 grid_split(num_row_windows, WARPperBlock / 4, 1);
    dim3 block_split(WARP_SIZE, 4, 1);
    dim3 grid_float4(num_row_windows, 1, 1);
    dim3 block_float4(WARP_SIZE, WARPperBlock1, 1);
    dim3 grid_float4_split(num_row_windows, WARPperBlock1 / 4, 1);
    dim3 block_float4_split(WARP_SIZE, 4, 1);

    // valuesA = all ones (unweighted adjacency)
    float *valuesA = nullptr;
    cudaMalloc(&valuesA, num_edges * sizeof(float));
    dtc_fill_float(valuesA, num_edges, 1.0f);

    std::string plan(exeplan);
    GpuTimer timer;

    if (plan == "float_nonsplit") {
        timer.Start();
        for (int i = 0; i < DTC_EXE_TIME; i++) {
            spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer
                <<<grid, block>>>(
                Rowwindow_offset, TCblocktile_id, TCblock_offset,
                sparse_AToX_idx, valuesA, num_nodes, num_edges,
                embedding_dim, input, output);
        }
        timer.Stop();
    } else if (plan == "float2_nonsplit") {
        timer.Start();
        for (int i = 0; i < DTC_EXE_TIME; i++) {
            spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float2
                <<<grid, block>>>(
                Rowwindow_offset, TCblocktile_id, TCblock_offset,
                sparse_AToX_idx, valuesA, num_nodes, num_edges,
                embedding_dim, input, output);
        }
        timer.Stop();
    } else if (plan == "float2_split") {
        timer.Start();
        if (embedding_dim >= 64)
        for (int i = 0; i < DTC_EXE_TIME; i++) {
            spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float2_split
                <<<grid_split, block_split>>>(
                Rowwindow_offset, TCblocktile_id, TCblock_offset,
                sparse_AToX_idx, valuesA, num_nodes, num_edges,
                embedding_dim, input, output);
        }
        timer.Stop();
    } else if (plan == "float4_nonsplit") {
        timer.Start();
        for (int i = 0; i < DTC_EXE_TIME; i++) {
            spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4
                <<<grid_float4, block_float4>>>(
                Rowwindow_offset, TCblocktile_id, TCblock_offset,
                sparse_AToX_idx, valuesA, num_nodes, num_edges,
                embedding_dim, input, output);
        }
        timer.Stop();
    } else if (plan == "float4_split") {
        timer.Start();
        if (embedding_dim >= 128)
        for (int i = 0; i < DTC_EXE_TIME; i++) {
            spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split
                <<<grid_float4_split, block_float4_split>>>(
                Rowwindow_offset, TCblocktile_id, TCblock_offset,
                sparse_AToX_idx, valuesA, num_nodes, num_edges,
                embedding_dim, input, output);
        }
        timer.Stop();
    } else {
        printf("DTC-SpMM: unsupported exeplan '%s'\n", exeplan);
        cudaFree(valuesA);
        return {0, 0};
    }

    float dtc_time = timer.Elapsed() / DTC_EXE_TIME;
    float spmm_flop = float(num_edges) * float(embedding_dim) * 2.0f;
    float throughput = (spmm_flop * 1000.0f) / (dtc_time * 1e9f);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    cudaFree(valuesA);
    return {dtc_time, throughput};
}

static SpmmResult dtc_spmm_balance_forward(
    int *TCblock_rowid,
    uint8_t *TCblocktile_id,
    int *TCblock_offset,
    int *sparse_AToX_idx,
    int tc_count,
    int num_nodes,
    int num_edges,
    int embedding_dim,
    float *input,
    float *output,
    const char *exeplan)
{
    const int WARPperBlock = embedding_dim / BLK_H;
    const int WARPperBlock1 = embedding_dim / 32;
    dim3 grid((tc_count + TCBLOCK_PER_WARP - 1) / TCBLOCK_PER_WARP, 1, 1);
    dim3 block(WARP_SIZE, WARPperBlock, 1);
    dim3 grid_float4_split((tc_count + TCBLOCK_PER_WARP - 1) / TCBLOCK_PER_WARP,
                           WARPperBlock1 / 4, 1);
    dim3 block_float4_split(WARP_SIZE, 4, 1);

    float *valuesA = nullptr;
    cudaMalloc(&valuesA, num_edges * sizeof(float));
    dtc_fill_float(valuesA, num_edges, 1.0f);

    std::string plan(exeplan);
    GpuTimer timer;

    size_t out_bytes = (size_t)num_nodes * embedding_dim * sizeof(float);

    if (plan == "float_nonsplit") {
        timer.Start();
        for (int i = 0; i < DTC_EXE_TIME; i++) {
            cudaMemset(output, 0, out_bytes);
            spmm_forward_cuda_kernel_improved_ptx_uint8_v1_strict_balance_withv
                <<<grid, block>>>(
                TCblock_rowid, TCblocktile_id, TCblock_offset,
                sparse_AToX_idx, valuesA, tc_count, num_nodes,
                num_edges, embedding_dim, input, output);
        }
        timer.Stop();
    } else if (plan == "float4_split") {
        timer.Start();
        if (embedding_dim >= 128)
        for (int i = 0; i < DTC_EXE_TIME; i++) {
            cudaMemset(output, 0, out_bytes);
            spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split_balance
                <<<grid_float4_split, block_float4_split>>>(
                TCblock_rowid, TCblocktile_id, TCblock_offset,
                sparse_AToX_idx, valuesA, tc_count, num_nodes,
                num_edges, embedding_dim, input, output);
        }
        timer.Stop();
    } else {
        printf("DTC-SpMM balance: unsupported exeplan '%s'\n", exeplan);
        cudaFree(valuesA);
        return {0, 0};
    }

    float dtc_time = timer.Elapsed() / DTC_EXE_TIME;
    float spmm_flop = float(num_edges) * float(embedding_dim) * 2.0f;
    float throughput = (spmm_flop * 1000.0f) / (dtc_time * 1e9f);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    cudaFree(valuesA);
    return {dtc_time, throughput};
}


//////////////////////////////////////////////////////////////////////
/// Main entry: dtc_spmm_run
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
/// cuSPARSE reference SpMM  (C = A × X, A in CSR, values = all 1s)
//////////////////////////////////////////////////////////////////////
#include <cusparse.h>

#define CUSPARSE_CHECK(call)                                                   \
    do {                                                                       \
        cusparseStatus_t err = (call);                                         \
        if (err != CUSPARSE_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuSPARSE error %d at %s:%d\n",                    \
                    (int)err, __FILE__, __LINE__);                             \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

/// Run cuSPARSE CSR SpMM:  C (m×k) = A (m×m, CSR) × X (m×k)
/// Values of A are all 1.0 (unweighted adjacency, same as DTC-SpMM).
/// Result is written to d_C and timed.
static SpmmResult cusparse_spmm_reference(
    const int *h_rowPtr, const int *h_col,
    int num_nodes, int num_edges, int embedding_dim,
    float *d_X, float *d_C)
{

    // Upload CSR to device
    int *d_rowPtr = nullptr, *d_col = nullptr;
    float *d_vals = nullptr;
    cudaMalloc(&d_rowPtr, (num_nodes + 1) * sizeof(int));
    cudaMalloc(&d_col, num_edges * sizeof(int));
    cudaMalloc(&d_vals, num_edges * sizeof(float));
    cudaMemcpy(d_rowPtr, h_rowPtr, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, h_col, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    dtc_fill_float(d_vals, num_edges, 1.0f);

    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    // Sparse matrix A (m×m)
    cusparseSpMatDescr_t matA;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &matA, num_nodes, num_nodes, num_edges,
        d_rowPtr, d_col, d_vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Dense matrix X (m×k, row-major)
    cusparseDnMatDescr_t matX;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matX, num_nodes, embedding_dim, /*ld=*/embedding_dim,
        d_X, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // Dense matrix C (m×k, row-major)
    cusparseDnMatDescr_t matC;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matC, num_nodes, embedding_dim, /*ld=*/embedding_dim,
        d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    float alpha = 1.0f, beta = 0.0f;
    size_t bufSize = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matX, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufSize));

    void *d_buf = nullptr;
    if (bufSize > 0) cudaMalloc(&d_buf, bufSize);

    // Warm-up
    CUSPARSE_CHECK(cusparseSpMM(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matX, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_buf));
    cudaDeviceSynchronize();

    // Timed runs
    GpuTimer timer;
    timer.Start();
    for (int i = 0; i < DTC_EXE_TIME; i++) {
        CUSPARSE_CHECK(cusparseSpMM(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matX, &beta, matC,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_buf));
    }
    timer.Stop();
    float avg_ms = timer.Elapsed() / DTC_EXE_TIME;
    float flop = float(num_edges) * float(embedding_dim) * 2.0f;
    float gflops = (flop * 1000.0f) / (avg_ms * 1e9f);

    // Cleanup cuSPARSE objects (keep d_C with result)
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matX);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);
    if (d_buf) cudaFree(d_buf);
    cudaFree(d_rowPtr);
    cudaFree(d_col);
    cudaFree(d_vals);

    return {avg_ms, gflops};
}

//////////////////////////////////////////////////////////////////////
/// Accuracy comparison between two GPU buffers
//////////////////////////////////////////////////////////////////////

/// Compare d_test against d_ref (both num_nodes × embedding_dim floats).
/// Prints max absolute error, mean absolute error, and RMSE.
static AccResult compare_results(
    const float *d_ref, const float *d_test,
    int num_nodes, int embedding_dim)
{
    size_t n = (size_t)num_nodes * embedding_dim;
    std::vector<float> h_ref(n), h_test(n);
    cudaMemcpy(h_ref.data(),  d_ref,  n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_test.data(), d_test, n * sizeof(float), cudaMemcpyDeviceToHost);

    double max_abs = 0.0, sum_abs = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = (double)h_test[i] - (double)h_ref[i];
        double adiff = fabs(diff);
        if (adiff > max_abs) max_abs = adiff;
        sum_abs += adiff;
        sum_sq += diff * diff;
    }
    double mean_abs = sum_abs / n;
    double rmse = sqrt(sum_sq / n);
    return {max_abs, mean_abs, rmse};
}

//////////////////////////////////////////////////////////////////////
/// Main DTC-SpMM driver (standalone — no DataLoader dependency)
//////////////////////////////////////////////////////////////////////

/// Run all SpMM variants and collect results into DtcRunResult.
static DtcRunResult dtc_spmm_run(
    const char *dataset_name,
    const int *h_rowPtr, const int *h_col,
    int num_nodes, int num_edges, int embedding_dim,
    float *d_X, float *d_C, float *d_C_ref,
    SpmmResult cusparse_res)
{
    DtcRunResult R = {};
    R.num_nodes = num_nodes;
    R.num_edges = num_edges;
    R.embedding_dim = embedding_dim;
    R.cusparse = cusparse_res;

    int num_row_windows = (num_nodes + BLK_H - 1) / BLK_H;

    int *d_rowPtr = nullptr, *d_col = nullptr;
    cudaMalloc(&d_rowPtr, (num_nodes + 1) * sizeof(int));
    cudaMalloc(&d_col, num_edges * sizeof(int));
    cudaMemcpy(d_rowPtr, h_rowPtr, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, h_col, num_edges * sizeof(int), cudaMemcpyHostToDevice);

    // --- CPU Preprocessing ---
    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<int> h_blockPartition(num_row_windows, 0);
    std::vector<int> h_edgeToColumn(num_edges, 0);
    std::vector<int> h_edgeToRow(num_edges, 0);

    int block_counter = cpu_preprocess(
        h_col, h_rowPtr,
        num_nodes, BLK_H, BLK_W,
        h_blockPartition.data(), h_edgeToColumn.data(), h_edgeToRow.data());

    auto t_end = std::chrono::high_resolution_clock::now();
    R.preprocess_sec = std::chrono::duration<double>(t_end - t_start).count();
    R.tc_blocks = block_counter;

    if (block_counter == 0) {
        fprintf(stderr, "  [%s] No TC blocks found — skipping.\n", dataset_name);
        cudaFree(d_rowPtr); cudaFree(d_col);
        return R;
    }

    // --- Upload preprocessing results to GPU ---
    int *d_blockPartition = nullptr, *d_edgeToColumn = nullptr, *d_edgeToRow = nullptr;
    cudaMalloc(&d_blockPartition, num_row_windows * sizeof(int));
    cudaMalloc(&d_edgeToColumn, num_edges * sizeof(int));
    cudaMalloc(&d_edgeToRow, num_edges * sizeof(int));
    cudaMemcpy(d_blockPartition, h_blockPartition.data(), num_row_windows * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeToColumn, h_edgeToColumn.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeToRow, h_edgeToRow.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);

    // --- Compute RowWindowOffset (prefix sum of blockPartition) ---
    int *d_rowwindow_offset = nullptr;
    cudaMalloc(&d_rowwindow_offset, (num_row_windows + 1) * sizeof(int));
    cudaMemset(d_rowwindow_offset, 0, (num_row_windows + 1) * sizeof(int));
    cudaMemcpy(d_rowwindow_offset + 1, d_blockPartition,
               num_row_windows * sizeof(int), cudaMemcpyDeviceToDevice);
    dtc_inclusive_scan_int(d_rowwindow_offset + 1, num_row_windows);

    // --- Generate TCblock_rowid ---
    int *d_tcblock_rowid = nullptr;
    cudaMalloc(&d_tcblock_rowid, block_counter * sizeof(int));
    cudaMemset(d_tcblock_rowid, 0, block_counter * sizeof(int));
    generate_tcblock_rowid_cuda(d_rowwindow_offset, d_tcblock_rowid, num_row_windows);

    // --- Find max_blocks per row window ---
    int max_blocks = *std::max_element(h_blockPartition.begin(), h_blockPartition.end());

    // --- Generate tcblock_offset, tcblocktile_id, sparse_AToX_index ---
    uint8_t *d_tcblocktile_id = nullptr;
    cudaMalloc(&d_tcblocktile_id, num_edges * sizeof(uint8_t));
    cudaMemset(d_tcblocktile_id, 0, num_edges * sizeof(uint8_t));

    int *d_tcblock_offset = nullptr;
    cudaMalloc(&d_tcblock_offset, (block_counter + 1) * sizeof(int));
    cudaMemset(d_tcblock_offset, 0, (block_counter + 1) * sizeof(int));

    int *d_sparse_AToX_index = nullptr;
    cudaMalloc(&d_sparse_AToX_index, block_counter * BLK_W * sizeof(int));
    cudaMemset(d_sparse_AToX_index, 0, block_counter * BLK_W * sizeof(int));

    generate_tcoffset_id_atob_cuda(
        d_rowPtr, d_rowwindow_offset, d_edgeToColumn, d_edgeToRow, d_col,
        d_tcblock_offset + 1, d_tcblocktile_id, d_sparse_AToX_index,
        max_blocks, num_nodes, BLK_H, BLK_W, num_row_windows);

    dtc_inclusive_scan_int(d_tcblock_offset, block_counter + 1);
    cudaDeviceSynchronize();

    // --- Select execution plan based on embedding_dim ---
    const char *plan;
    if (embedding_dim <= 32)       plan = "float_nonsplit";
    else if (embedding_dim <= 64)  plan = "float2_nonsplit";
    else if (embedding_dim <= 128) plan = "float4_nonsplit";
    else                           plan = "float4_split";
    R.dtc_plan = plan;

    // --- Run non-balanced version ---
    cudaMemset(d_C, 0, (size_t)num_nodes * embedding_dim * sizeof(float));
    R.dtc = dtc_spmm_forward(
        d_rowwindow_offset, d_tcblocktile_id, d_tcblock_offset,
        d_sparse_AToX_index, num_row_windows, num_nodes, num_edges,
        embedding_dim, d_X, d_C, plan);
    if (d_C_ref)
        R.dtc_acc = compare_results(d_C_ref, d_C, num_nodes, embedding_dim);

    // --- Run balanced version ---
    cudaMemset(d_C, 0, (size_t)num_nodes * embedding_dim * sizeof(float));
    const char *balance_plan;
    if (embedding_dim >= 128) balance_plan = "float4_split";
    else                      balance_plan = "float_nonsplit";
    R.bal_plan = balance_plan;

    R.dtc_bal = dtc_spmm_balance_forward(
        d_tcblock_rowid, d_tcblocktile_id, d_tcblock_offset,
        d_sparse_AToX_index, block_counter, num_nodes, num_edges,
        embedding_dim, d_X, d_C, balance_plan);
    if (d_C_ref)
        R.dtc_bal_acc = compare_results(d_C_ref, d_C, num_nodes, embedding_dim);

    // --- Cleanup ---
    cudaFree(d_rowPtr);
    cudaFree(d_col);
    cudaFree(d_blockPartition);
    cudaFree(d_edgeToColumn);
    cudaFree(d_edgeToRow);
    cudaFree(d_rowwindow_offset);
    cudaFree(d_tcblock_rowid);
    cudaFree(d_tcblocktile_id);
    cudaFree(d_tcblock_offset);
    cudaFree(d_sparse_AToX_index);
    return R;
}

//////////////////////////////////////////////////////////////////////
/// Standalone CSV loader & main()
//////////////////////////////////////////////////////////////////////

/// Load a Flex-format CSV file (line 1: rowPtr, line 2: col, line 3: vals).
/// Returns CSR arrays as int vectors.
static void load_csr_csv(const std::string &path,
                         std::vector<int> &rowPtr,
                         std::vector<int> &col,
                         std::vector<float> &vals)
{
    std::fstream fin(path, std::ios::in);
    if (!fin.is_open()) {
        fprintf(stderr, "Error: cannot open %s\n", path.c_str());
        exit(1);
    }
    std::string line, word;

    // Line 1: row pointers
    std::getline(fin, line);
    std::stringstream ss1(line);
    while (std::getline(ss1, word, ','))
        rowPtr.push_back(std::stoi(word));

    // Line 2: column indices
    std::getline(fin, line);
    std::stringstream ss2(line);
    while (std::getline(ss2, word, ','))
        col.push_back(std::stoi(word));

    // Line 3: values (optional — fill with 1.0 if missing)
    if (std::getline(fin, line) && !line.empty()) {
        std::stringstream ss3(line);
        while (std::getline(ss3, word, ','))
            vals.push_back(std::stof(word));
    }
    if (vals.empty()) {
        vals.assign(col.size(), 1.0f);
    }
    fin.close();
}

/// Extract a short dataset name from a file path (e.g. "./data/flickr.csv" → "flickr").
static std::string extract_dataset_name(const std::string &path) {
    size_t slash = path.find_last_of('/');
    std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) base = base.substr(0, dot);
    return base;
}

/// Print the table header.
static void print_table_header() {
    printf("%-14s %8s %10s %4s | %9s %9s | %9s %9s %10s | %9s %9s %10s\n",
           "Dataset", "Nodes", "Edges", "K",
           "cuSP/ms", "cuSP/GF",
           "DTC/ms", "DTC/GF", "DTC/RMSE",
           "Bal/ms", "Bal/GF", "Bal/RMSE");
    printf("%-14s %8s %10s %4s | %9s %9s | %9s %9s %10s | %9s %9s %10s\n",
           "--------------", "--------", "----------", "----",
           "---------", "---------",
           "---------", "---------", "----------",
           "---------", "---------", "----------");
}

/// Print one result row.
static void print_table_row(const char *name, const DtcRunResult &R) {
    printf("%-14s %8d %10d %4d | %9.4f %9.2f | %9.4f %9.2f %10.2e | %9.4f %9.2f %10.2e\n",
           name, R.num_nodes, R.num_edges, R.embedding_dim,
           R.cusparse.time_ms, R.cusparse.gflops,
           R.dtc.time_ms, R.dtc.gflops, R.dtc_acc.rmse,
           R.dtc_bal.time_ms, R.dtc_bal.gflops, R.dtc_bal_acc.rmse);
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <csr.csv> <embedding_dim> [<csr2.csv> ...]\n", argv[0]);
        return 1;
    }
    const int embedding_dim = atoi(argv[2]);

    // Collect all CSV paths (allow multiple datasets on one invocation)
    std::vector<std::string> csv_paths;
    csv_paths.push_back(argv[1]);
    for (int i = 3; i < argc; ++i)
        csv_paths.push_back(argv[i]);

    print_table_header();

    for (const auto &csv_path : csv_paths) {
        std::string ds_name = extract_dataset_name(csv_path);

        // --- Load CSR from CSV ---
        std::vector<int> h_rowPtr, h_col;
        std::vector<float> h_vals;
        load_csr_csv(csv_path, h_rowPtr, h_col, h_vals);

        int num_nodes = (int)h_rowPtr.size() - 1;
        int num_edges = (int)h_col.size();

        // --- Allocate dense X (random), C (DTC output), C_ref (cuSPARSE) on GPU ---
        size_t X_bytes = (size_t)num_nodes * embedding_dim * sizeof(float);
        size_t C_bytes = X_bytes;

        std::vector<float> h_X(num_nodes * embedding_dim);
        srand(42);
        for (auto &v : h_X) v = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;

        float *d_X = nullptr, *d_C = nullptr, *d_C_ref = nullptr;
        cudaMalloc(&d_X, X_bytes);
        cudaMalloc(&d_C, C_bytes);
        cudaMalloc(&d_C_ref, C_bytes);
        cudaMemcpy(d_X, h_X.data(), X_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_C, 0, C_bytes);
        cudaMemset(d_C_ref, 0, C_bytes);

        // --- cuSPARSE reference ---
        SpmmResult cusparse_res = cusparse_spmm_reference(
            h_rowPtr.data(), h_col.data(),
            num_nodes, num_edges, embedding_dim,
            d_X, d_C_ref);

        // --- DTC-SpMM (all variants + accuracy check) ---
        DtcRunResult R = dtc_spmm_run(
            ds_name.c_str(),
            h_rowPtr.data(), h_col.data(),
            num_nodes, num_edges, embedding_dim,
            d_X, d_C, d_C_ref,
            cusparse_res);

        print_table_row(ds_name.c_str(), R);

        cudaFree(d_X);
        cudaFree(d_C);
        cudaFree(d_C_ref);
    }

    return 0;
}
