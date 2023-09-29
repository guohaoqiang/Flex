#include "flex.cuh"
#include <ranges>
#include <set>
/*
__device__ __forceinline__
uint32_t glm_u32addr(const void *glm_ptr) {
    uint32_t addr;
    asm ("{.reg .u64 u64addr;\n"
         " cvta.to.global.u64 u64addr, %1;\n"
         " cvt.u32.u64 %0, u64addr;}\n"
         : "=r"(addr)
         : "l"(glm_ptr)
    );
    return addr;
}
__device__ __forceinline__
void ldg64(int &reg0, int &reg1, const uint32_t &addr) {
    asm volatile (
        "ld.global.v2.u32 {%0, %1}, [%2];\n"
        : "=r"(reg0), "=r"(reg1)
        : "r"(addr)
    );
}
*/


// Return Streaming Multiprocessor (aka SM, MP) ID.
__device__ uint32_t
smid_get()
{
  uint smid = 0;
  asm( "mov.u32 %0, %%smid;" : "=r" (smid) );
  return smid;
}

typedef uint32_t pClock_t;

constexpr size_t timing_item_size = 16;

struct __align__(timing_item_size) Timing_Item {
  pClock_t time_start;
  uint32_t smid_start;
  pClock_t time_end;
  uint32_t smid_end; // To detect preemption.
};
static_assert( timing_item_size == sizeof(Timing_Item) );

struct Timing {
  Timing_Item *timing_items;
};

__constant__ Timing timing_dev;

__device__ void
timing_start()
{
  constexpr int wp_sz = 32;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if ( threadIdx.x % wp_sz == 0 )
    {
      Timing_Item& ti = timing_dev.timing_items[tid/wp_sz];
      ti.time_start = clock();
      ti.smid_start = smid_get();
    }
  __syncwarp();
}
__device__ void
timing_end()
{
  constexpr int wp_sz = 32;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __syncwarp();
  if ( threadIdx.x % wp_sz == 0 )
    {
      Timing_Item& ti = timing_dev.timing_items[tid/wp_sz];
      ti.time_end = clock();
      ti.smid_end = smid_get();
    }
}
/*************** CSR ge-spmm (top) *******************/
__global__ 
void spmm_test0()
{
    // this code is adapted from "https://github.com/hgyhungry/ge-spmm.git"
    
    // kernel for k in [1,32)
    // grid: dim3( (m+128/k-1)/(128/k), 1, 1  ) 
    // block: dim3( k, 128/k, 1 )
   
    const Mat_POD& md = mat_dev; 

    timing_start();
    int rid = blockDim.y*blockIdx.x+threadIdx.y;
    if (rid<md.m) {
        int cid = (blockIdx.y<<5)+threadIdx.x;
        int lb = md.csr_rowPtr_dev[rid];
        int hb = md.csr_rowPtr_dev[(rid+1)];
        int offset = 0;
        float acc=0;
        if (blockIdx.y!=gridDim.y-1){
            for (int ptr = lb; ptr<hb; ptr++) {
                offset = md.csr_col_dev[ptr]*md.k+cid;
                acc += md.csr_vals_dev[ptr]*md.mat_b_dev[offset];
            }
            md.mat_c_dev[(rid*md.k+cid)] = acc;
        }
        else {
            for (int ptr = lb; ptr<hb; ptr++) {
                if (cid<md.k) {
                offset = md.csr_col_dev[ptr]*md.k+cid;}
                acc += md.csr_vals_dev[ptr]*md.mat_b_dev[offset];
            }
            if (cid<md.k) {
            md.mat_c_dev[(rid*md.k+cid)] = acc;}
        }
    }
    timing_end();
}

__global__ 
void spmm_test1()
{
    // this code is adapted from "https://github.com/hgyhungry/ge-spmm.git"
    
    // kernel for k in [32,64)
    // grid: dim3( (m+4-1)/4, (k+31)/32, 1  ) 
    // block: dim3( 32,4,1 )
    // shared m: 32*4*(sizeof(int)+sizeof(float)) 
    const Mat_POD& md = mat_dev; 
    
    timing_start();
    extern __shared__ int sh[];
    int *colInd_sh = sh;
    float *val_sh = (float *)&sh[(blockDim.y<<5)];
    int shmem_offset = (threadIdx.y<<5);
    int thread_idx = shmem_offset+threadIdx.x;

    int rid = blockDim.y*blockIdx.x+threadIdx.y;

    if (rid<md.m) {
        int cid = (blockIdx.y<<5)+threadIdx.x;
        int lb = md.csr_rowPtr_dev[rid];
        int hb = md.csr_rowPtr_dev[(rid+1)];
        int ptr = lb+threadIdx.x;
        int offset;
        float acc=0;

        if (blockIdx.y != gridDim.y-1) {
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = md.csr_vals_dev[ptr];
                    colInd_sh[thread_idx] = md.k*md.csr_col_dev[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    acc += val_sh[(shmem_offset+kk)]*md.mat_b_dev[offset];
                }
                __syncwarp();
            }
            md.mat_c_dev[(rid*md.k+cid)] = acc;
        }
        else {
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = md.csr_vals_dev[ptr];
                    colInd_sh[thread_idx] = md.k*md.csr_col_dev[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    if (cid<md.k) {
                    acc += val_sh[(shmem_offset+kk)]*md.mat_b_dev[offset];
                    }
                }
                __syncwarp();
            }
            if (cid<md.k) {
            md.mat_c_dev[(rid*md.k+cid)] = acc;
            }
        }
    }
    timing_end();
}

__global__ 
void spmm_test2()
{

    // this code is adapted from "https://github.com/hgyhungry/ge-spmm.git"
    
    // kernel for k in [64,+oo)
    // grid: dim3( (m+8-1)/8, (k+63)/64, 1  ) 
    // block: dim3( 32,8,1 )
    // shared m: 32*8*(sizeof(int)+sizeof(float)) 

    const Mat_POD& md = mat_dev; 
    
    timing_start();
    extern __shared__ int sh[];
    int *colInd_sh = sh;
    float *val_sh = (float *)&sh[(blockDim.y<<5)];
    int shmem_offset = (threadIdx.y<<5);
    int thread_idx = shmem_offset+threadIdx.x;

    int rid = blockDim.y*blockIdx.x+threadIdx.y;

   if (rid<md.m) {
        int cid = (blockIdx.y<<6)+threadIdx.x;
        int lb = md.csr_rowPtr_dev[rid];
        int hb = md.csr_rowPtr_dev[(rid+1)];
        int ptr = lb+threadIdx.x;
        int offset;
        float acc1=0, acc2=0, val;

        if (blockIdx.y != gridDim.y-1) {
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = md.csr_vals_dev[ptr];
                    colInd_sh[thread_idx] = md.k*md.csr_col_dev[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    val = val_sh[(shmem_offset+kk)];
                    acc1 += val*md.mat_b_dev[offset];
                    acc2 += val*md.mat_b_dev[offset+32];
                }
                __syncwarp();
            }
            offset = rid*md.k+cid;
            md.mat_c_dev[offset] = acc1;
            md.mat_c_dev[offset+32] = acc2;
        }
        else {
            int nout = (md.k-cid+31)/32;
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = md.csr_vals_dev[ptr];
                    colInd_sh[thread_idx] = md.k*md.csr_col_dev[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    val = val_sh[(shmem_offset+kk)];
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    if (nout>0) {
                    acc1 += val*md.mat_b_dev[offset];
                    }
                    if (nout>1) {
                    acc2 += val*md.mat_b_dev[offset+32];
                    }
                }
                __syncwarp();
            }
            offset = rid*md.k+cid;
            if (nout>0) {
                md.mat_c_dev[offset] = acc1;
            }
            if (nout>1) {
                md.mat_c_dev[(offset+32)] = acc2;
            }
        }
    }
    timing_end();
}
/*************** CSR ge-spmm (bottom) *******************/

__global__
void flexspmm_v9_permuteX(){
    // preprocess dense mat B. out-of-place permutation of B rows
    const Mat_POD& md = mat_dev;
    const int rows_p_blk = blockDim.x / 32; // a warp moves a row
    const int lane_id = threadIdx.x % 32;
	for (int row_idx=blockIdx.x*rows_p_blk+threadIdx.x/32; row_idx<md.n; row_idx += (gridDim.x*rows_p_blk)){ // over C rows
      
        int tgt_row = md.voMp_dev[row_idx];  
        for (int i = lane_id; i<md.k; i += 32){
            md.shadow_b_dev[ row_idx*md.k+i ] = md.mat_b_dev[ tgt_row*md.k+i ]; 
        }
	} // end C row loops    
}

__global__
void flexspmm_vec_permuteX(){
    // preprocess dense mat B. out-of-place permutation of B rows
    const Mat_POD& md = mat_dev;
    const int rows_p_blk = blockDim.x / 32; // a warp moves a row
    const int lane_id = threadIdx.x % 32;
	for (int row_idx=blockIdx.x*rows_p_blk+threadIdx.x/32; row_idx<md.n; row_idx += (gridDim.x*rows_p_blk)){ // over C rows
      
        int tgt_row = md.voMp_dev[row_idx];  
        float *shadow_b_addr = &md.shadow_b_dev[ row_idx*md.k ];
        float *mat_b_addr = &md.mat_b_dev[ tgt_row*md.k ];
        for (int i = lane_id; i<md.k/4; i += 32){
            reinterpret_cast<float4*>(shadow_b_addr)[ i ] = reinterpret_cast<float4*>(mat_b_addr)[ i ]; 
        }

        // for k is not a multiple of 128, TBD
        //int remainder = md.k%4;

	} // end C row loops    
}

// args:
//		tileRowPtr: tile ptr for the 1st tile in each row
//		tileNnz: ptr for the 1st non zero entry of each tile
// 		nnzTile: #nnz of each tile
// 		bitMap: mark B rows required by the each tile
// 		tileCol: column idx of each tile. 
//      rcOffset: row and column indexfor each non-zero entry
//		vals: non-zero entries
// 		m: height of sparseMat
// 		n: width of sparseMat
// 		mat_b: input dense mat
//		k: width of mat_b
//		mat_c: output dense mat
// A: sparse, m * n
// B: dense, n * k   (k << n)
template<int tm, int tn, int warps>
__global__
void flexspmm_cuda_wo_pre_v4(){
    const Mat_POD& md = mat_dev;
	const uint32_t WARPSZ = 32;
	const uint32_t lane_id = threadIdx.x % WARPSZ;
    const uint32_t warp_id = threadIdx.x / WARPSZ;
	//const uint32_t warps = (blockDim.x + WARPSZ - 1)/WARPSZ;

	// now we restrain "tn" in {4,8,16,32}
	__shared__ float curB[warps][tn*32]; // 2 warps && each warp needs tn*8*4 matB float entries
	float res[tm];
	#pragma unroll
	for (int i=0; i<tm; ++i){
		res[i] = 0;
	}

	int computeWidth = 1; // # of C entries to be computed by a thread
	int tileRows_perBlk = 1; // # row tiles per block
	for (int row_idx=blockIdx.x*tileRows_perBlk; row_idx<(md.m+tm-1)/tm; row_idx += (gridDim.x*tileRows_perBlk)){ // over C rows
	   
        int tile_curR_id = 0, tile_nxtR_id = 0;
        tile_curR_id = md.tileRowPtr_dev[row_idx]; 
        tile_nxtR_id = md.tileRowPtr_dev[row_idx+1]; 

        for (int col_idx=warp_id*(32*computeWidth); col_idx<md.k; col_idx += warps*(32*computeWidth)){  // over C tile columns
             
            int tiles = 0;

            for (int tile_id=tile_curR_id; tile_id<tile_nxtR_id; tile_id+=tiles){

                uint32_t mask_tiles = __ballot_sync(FULL_MASK, tile_id+lane_id<tile_nxtR_id);
                tiles = __popc(mask_tiles); // maximum # tiles can be loaded in cur row 
                
                int start_of_tile = 0, nnz_of_tile = 0, bitmap_of_tile = 0, col_of_tile = 0;
                if (tile_curR_id+lane_id<tile_nxtR_id){
                    // load as many as as tile info of cur tile-row
                    start_of_tile = md.tileNnz_dev[tile_id+lane_id];
                    nnz_of_tile = md.nnzTile_dev[tile_id+lane_id];
                    bitmap_of_tile = md.bitMap_dev[tile_id+lane_id];
                    col_of_tile = md.tileColIdx_dev[tile_id+lane_id];
                }

                // use all loaded tiles
                for(int tile_cnt = 0; tile_cnt<tiles; ++tile_cnt){
                    int start_cur_tile = __shfl_sync(FULL_MASK, start_of_tile, tile_cnt);
                    int nnz_cur_tile = __shfl_sync(FULL_MASK, nnz_of_tile, tile_cnt);
                    int bitmap_cur_tile = __shfl_sync(FULL_MASK, bitmap_of_tile, tile_cnt);
                    int col_cur_tile = __shfl_sync(FULL_MASK, col_of_tile, tile_cnt);
                    
					// load requiring B rows to smem
					for (int j=0; j<tn; ++j){
						if ((bitmap_cur_tile & (1<<j)) && col_idx+lane_id<md.k){
                            curB[warp_id][j*32+lane_id] = md.mat_b_dev[(col_cur_tile+j)*md.k + col_idx + lane_id];
						}
					}
					//__syncwarp(); // I doubt if it is necessary besause warp is the scheduling unit

					// visit all nz of the sparse tile
					int steps = 1;
                    int cur_end = start_cur_tile+nnz_cur_tile;
					for (int kk=start_cur_tile; kk<cur_end; kk+=steps){
					    uint32_t mask_join = __ballot_sync(FULL_MASK, kk+lane_id<cur_end);
                		steps = __popc(mask_join);

                		float val = 0;
                		int rcidx = 0;
                        if (kk+lane_id<cur_end){
                		    // load sparse nnz from glb mem
                		    val = md.vals_dev[kk+lane_id];
                		    rcidx = md.rcOffset_dev[kk+lane_id];
                        }
                		// exchange nnz within a warp && perfom FMA
                		for (int it=0; it<steps; ++it){
                			float v = __shfl_sync(FULL_MASK, val, it);
                			int rc = __shfl_sync(FULL_MASK, rcidx, it);

                			res[rc>>16] += v * curB[warp_id][(rc & 0x0000ffff)*32 + lane_id];
                		}
					}// end visiting all nz in a sparse tile
                    
                }// end visiting all loaded sparse tiles
            }// end visiting all sparse tiles in cur tile-row
            
			// store C tiles back to global mem
            //#pragma unroll
            for (int c=0; c<tm; ++c){
                if (row_idx*tm+c<md.m){
                    md.mat_c_dev[(row_idx*tm+c)*md.k+col_idx+lane_id] = res[c];
                }
                res[c] = 0;
            }
    
		} // end C column loops
	} // end C row loops
}
// args:
//		tileRowPtr: tile ptr for the 1st tile in each row
//		tileNnz: ptr for the 1st non zero entry of each tile
// 		nnzTile: #nnz of each tile
// 		bitMap: mark B rows required by the each tile
// 		tileCol: column idx of each tile. 
//      rcOffset: row and column indexfor each non-zero entry
//		vals: non-zero entries
// 		m: height of sparseMat
// 		n: width of sparseMat
// 		mat_b: input dense mat
//		k: width of mat_b
//		mat_c: output dense mat
// A: sparse, m * n
// B: dense, n * k   (k << n)
template<int tm, int tn, int warps>
__global__
void flexspmm_cuda_wo_pre_v5(){
    const Mat_POD& md = mat_dev;
	const uint32_t WARPSZ = 32;
	const uint32_t lane_id = threadIdx.x % WARPSZ;
    const uint32_t warp_id = threadIdx.x / WARPSZ;
	//const uint32_t warps = (blockDim.x + WARPSZ - 1)/WARPSZ;

    timing_start();

	// now we restrain "tn" in {4,8,16,32}
	__shared__ float curB[warps][tn*32]; // 2 warps && each warp needs tn*8*4 matB float entries
	float res[tm];
	#pragma unroll
	for (int i=0; i<tm; ++i){
		res[i] = 0;
	}

	int computeWidth = 1; // # of C entries to be computed by a thread
	int tileRows_perBlk = 1; // # row tiles per block
	for (int row_idx=blockIdx.x*tileRows_perBlk; row_idx<(md.m+tm-1)/tm; row_idx += (gridDim.x*tileRows_perBlk)){ // over C rows
	   
        int tile_curR_id = 0, tile_nxtR_id = 0;
        tile_curR_id = md.tileRowPtr_dev[row_idx]; 
        tile_nxtR_id = md.tileRowPtr_dev[row_idx+1]; 

        for (int col_idx=warp_id*(32*computeWidth); col_idx<md.k; col_idx += warps*(32*computeWidth)){  // over C tile columns
            
            int tiles = 0;

            for (int tile_id=tile_curR_id; tile_id<tile_nxtR_id; tile_id+=tiles){

                uint32_t mask_tiles = __ballot_sync(FULL_MASK, tile_id+lane_id<tile_nxtR_id);
                tiles = __popc(mask_tiles); // maximum # tiles can be loaded in cur row 
                
                int start_of_tile = 0, nnz_of_tile = 0, bitmap_of_tile = 0, col_of_tile = 0;
                if (tile_curR_id+lane_id<tile_nxtR_id){
                    // load as many as as tile info of cur tile-row
                    start_of_tile = md.tileNnz_dev[tile_id+lane_id];
                    nnz_of_tile = md.nnzTile_dev[tile_id+lane_id];
                    bitmap_of_tile = md.bitMap_dev[tile_id+lane_id];
                    col_of_tile = md.tileColIdx_dev[tile_id+lane_id];
                }

                // use all loaded tiles
                for(int tile_cnt = 0; tile_cnt<tiles; ++tile_cnt){
                    int start_cur_tile = __shfl_sync(FULL_MASK, start_of_tile, tile_cnt);
                    int nnz_cur_tile = __shfl_sync(FULL_MASK, nnz_of_tile, tile_cnt);
                    int bitmap_cur_tile = __shfl_sync(FULL_MASK, bitmap_of_tile, tile_cnt);
                    int col_cur_tile = __shfl_sync(FULL_MASK, col_of_tile, tile_cnt);
                    
					// load requiring B rows to smem
					for (int j=0; j<tn; ++j){
						if ((bitmap_cur_tile & (1<<j)) && col_idx+lane_id<md.k){
                            curB[warp_id][j*32+lane_id] = md.mat_b_dev[(col_cur_tile+j)*md.k + col_idx + lane_id];
						}
					}

					// visit all nz of the sparse tile
                    if (nnz_cur_tile==1){
                        // load sparse nnz from glb mem
                        float val = md.vals_dev[start_cur_tile];
                        int rcidx = md.rcOffset_dev[start_cur_tile];
                        res[rcidx>>16] += val * curB[warp_id][(rcidx & 0x0000ffff)*32 + lane_id];
                         
                    }else if (nnz_cur_tile==2){    
                        // load sparse nnz from glb mem
                        float val1 = md.vals_dev[start_cur_tile];
                        float val2 = md.vals_dev[start_cur_tile+1];
                        int rcidx1 = md.rcOffset_dev[start_cur_tile];
                        int rcidx2 = md.rcOffset_dev[start_cur_tile+1];
                        res[rcidx1>>16] += val1 * curB[warp_id][(rcidx1 & 0x0000ffff)*32 + lane_id];
                        res[rcidx2>>16] += val2 * curB[warp_id][(rcidx2 & 0x0000ffff)*32 + lane_id];
                    }else if (nnz_cur_tile==3){
                        // load sparse nnz from glb mem
                        float val1 = md.vals_dev[start_cur_tile];
                        float val2 = md.vals_dev[start_cur_tile+1];
                        float val3 = md.vals_dev[start_cur_tile+2];
                        int rcidx1 = md.rcOffset_dev[start_cur_tile];
                        int rcidx2 = md.rcOffset_dev[start_cur_tile+1];
                        int rcidx3 = md.rcOffset_dev[start_cur_tile+2];
                        res[rcidx1>>16] += val1 * curB[warp_id][(rcidx1 & 0x0000ffff)*32 + lane_id];
                        res[rcidx2>>16] += val2 * curB[warp_id][(rcidx2 & 0x0000ffff)*32 + lane_id]; 
                        res[rcidx3>>16] += val3 * curB[warp_id][(rcidx3 & 0x0000ffff)*32 + lane_id]; 
                    }else if(nnz_cur_tile==4){
                        // load sparse nnz from glb mem
                        float val1 = md.vals_dev[start_cur_tile];
                        float val2 = md.vals_dev[start_cur_tile+1];
                        float val3 = md.vals_dev[start_cur_tile+2];
                        float val4 = md.vals_dev[start_cur_tile+3];
                        int rcidx1 = md.rcOffset_dev[start_cur_tile];
                        int rcidx2 = md.rcOffset_dev[start_cur_tile+1];
                        int rcidx3 = md.rcOffset_dev[start_cur_tile+2];
                        int rcidx4 = md.rcOffset_dev[start_cur_tile+3];
                        res[rcidx1>>16] += val1 * curB[warp_id][(rcidx1 & 0x0000ffff)*32 + lane_id];
                        res[rcidx2>>16] += val2 * curB[warp_id][(rcidx2 & 0x0000ffff)*32 + lane_id]; 
                        res[rcidx3>>16] += val3 * curB[warp_id][(rcidx3 & 0x0000ffff)*32 + lane_id]; 
                        res[rcidx4>>16] += val4 * curB[warp_id][(rcidx4 & 0x0000ffff)*32 + lane_id]; 
                    }else{ 
                        int steps = 1;
                        int cur_end = start_cur_tile+nnz_cur_tile;
                        for (int kk=start_cur_tile; kk<cur_end; kk+=steps){
                            uint32_t mask_join = __ballot_sync(FULL_MASK, kk+lane_id<cur_end);
                            steps = __popc(mask_join);

                            float val = 0;
                            int rcidx = 0;
                            if (kk+lane_id<cur_end){
                                // load sparse nnz from glb mem
                                val = md.vals_dev[kk+lane_id];
                                rcidx = md.rcOffset_dev[kk+lane_id];
                            }
                            // exchange nnz within a warp && perfom FMA
                            for (int it=0; it<steps; ++it){
                                float v = __shfl_sync(FULL_MASK, val, it);
                                int rc = __shfl_sync(FULL_MASK, rcidx, it);

                                res[rc>>16] += v * curB[warp_id][(rc & 0x0000ffff)*32 + lane_id];
                            }
                        }// end visiting all nz in a sparse tile
                    }
                }// end visiting all loaded sparse tiles
            }// end visiting all sparse tiles in cur tile-row
            
			// store C tiles back to global mem
            //#pragma unroll
            for (int c=0; c<tm; ++c){
                if (row_idx*tm+c<md.m){
                    md.mat_c_dev[(row_idx*tm+c)*md.k+col_idx+lane_id] = res[c];
                }
                res[c] = 0;
            }
    
		} // end C column loops
	} // end C row loops

        timing_end();

}
// args:
//		tileRowPtr: tile ptr for the 1st tile in each row
//		tileNnz: ptr for the 1st non zero entry of each tile
// 		nnzTile: #nnz of each tile
// 		bitMap: mark B rows required by the each tile
// 		tileCol: column idx of each tile. 
//      rcOffset: row and column indexfor each non-zero entry
//		vals: non-zero entries
// 		m: height of sparseMat
// 		n: width of sparseMat
// 		mat_b: input dense mat
//		k: width of mat_b
//		mat_c: output dense mat
// A: sparse, m * n
// B: dense, n * k   (k << n)
template<int tm, int tn, int warps>
__global__
void flexspmm_cuda_wo_pre_v6(){
    const Mat_POD& md = mat_dev;
	const uint32_t WARPSZ = 32;
	const uint32_t lane_id = threadIdx.x % WARPSZ;
    const uint32_t warp_id = threadIdx.x / WARPSZ;
	//const uint32_t warps = (blockDim.x + WARPSZ - 1)/WARPSZ;

    timing_start();

	// now we restrain "tn" in {4,8,16,32}
	__shared__ float curB[warps][tn*32]; // 2 warps && each warp needs tn*8*4 matB float entries
	float res[tm];
	#pragma unroll
	for (int i=0; i<tm; ++i){
		res[i] = 0;
	}

	int computeWidth = 1; // # of C entries to be computed by a thread
	int tileRows_perBlk = 1; // # row tiles per block
	for (int row_idx=blockIdx.x*tileRows_perBlk; row_idx<(md.m+tm-1)/tm; row_idx += (gridDim.x*tileRows_perBlk)){ // over C rows
	   
        int tile_curR_id = 0, tile_nxtR_id = 0;
        tile_curR_id = md.tileRowPtr_dev[row_idx]; 
        tile_nxtR_id = md.tileRowPtr_dev[row_idx+1]; 

        for (int col_idx=warp_id*(32*computeWidth); col_idx<md.k; col_idx += warps*(32*computeWidth)){  // over C tile columns
            
            int tiles = 0;

            for (int tile_id=tile_curR_id; tile_id<tile_nxtR_id; tile_id+=tiles){

                uint32_t mask_tiles = __ballot_sync(FULL_MASK, tile_id+lane_id<tile_nxtR_id);
                tiles = __popc(mask_tiles); // maximum # tiles can be loaded in cur row 
                
                int start_of_tile = 0, nnz_of_tile = 0, bitmap_of_tile = 0, col_of_tile = 0;
                if (tile_curR_id+lane_id<tile_nxtR_id){
                    // load as many as as tile info of cur tile-row
                    start_of_tile = md.tileNnz_dev[tile_id+lane_id];
                    nnz_of_tile = md.nnzTile_dev[tile_id+lane_id];
                    bitmap_of_tile = md.bitMap_dev[tile_id+lane_id];
                    col_of_tile = md.tileColIdx_dev[tile_id+lane_id];
                }

                // use all loaded tiles
                for(int tile_cnt = 0; tile_cnt<tiles; ++tile_cnt){
                    int start_cur_tile = __shfl_sync(FULL_MASK, start_of_tile, tile_cnt);
                    int nnz_cur_tile = __shfl_sync(FULL_MASK, nnz_of_tile, tile_cnt);
                    int bitmap_cur_tile = __shfl_sync(FULL_MASK, bitmap_of_tile, tile_cnt);
                    int col_cur_tile = __shfl_sync(FULL_MASK, col_of_tile, tile_cnt);
                   
                    if (nnz_cur_tile>4){ 
					    // load requiring B rows to smem
					    for (int j=0; j<tn; ++j){
						    if ((bitmap_cur_tile & (1<<j)) && col_idx+lane_id<md.k){
                                curB[warp_id][j*32+lane_id] = md.mat_b_dev[(col_cur_tile+j)*md.k + col_idx + lane_id];
						    }
					    }
                    }
                    auto do_n = [&](int n)
                     {
                       for ( int z=0; z<n; z++ )
                         {
                           float val = md.vals_dev[start_cur_tile+z];
                           int rcidx = md.rcOffset_dev[start_cur_tile+z];
                           int x_row = col_cur_tile + (rcidx & 0xffff);
                           res[rcidx>>16] += val
                             * md.mat_b_dev[x_row*md.k + col_idx + lane_id];
                           //res[rcidx>>16] += val * curB[warp_id][(rcidx & 0x0000ffff)*32 + lane_id];
                         }
                     };
					// visit all nz of the sparse tile
                    if (nnz_cur_tile==1){
                        do_n(1);
                    }else if (nnz_cur_tile==2){    
                        do_n(2);
                    }else if (nnz_cur_tile==3){
                        do_n(3);
                    }else if(nnz_cur_tile==4){
                        do_n(4);
                    }else{ 
                        int steps = 1;
                        int cur_end = start_cur_tile+nnz_cur_tile;
                        for (int kk=start_cur_tile; kk<cur_end; kk+=steps){
                            uint32_t mask_join = __ballot_sync(FULL_MASK, kk+lane_id<cur_end);
                            steps = __popc(mask_join);

                            float val = 0;
                            int rcidx = 0;
                            if (kk+lane_id<cur_end){
                                // load sparse nnz from glb mem
                                val = md.vals_dev[kk+lane_id];
                                rcidx = md.rcOffset_dev[kk+lane_id];
                            }
                            // exchange nnz within a warp && perfom FMA
                            for (int it=0; it<steps; ++it){
                                float v = __shfl_sync(FULL_MASK, val, it);
                                int rc = __shfl_sync(FULL_MASK, rcidx, it);

                                res[rc>>16] += v * curB[warp_id][(rc & 0x0000ffff)*32 + lane_id];
                            }
                        }// end visiting all nz in a sparse tile
                    }
                }// end visiting all loaded sparse tiles
            }// end visiting all sparse tiles in cur tile-row
            
			// store C tiles back to global mem
            //#pragma unroll
            for (int c=0; c<tm; ++c){
                if (row_idx*tm+c<md.m){
                    md.mat_c_dev[(row_idx*tm+c)*md.k+col_idx+lane_id] = res[c];
                }
                res[c] = 0;
            }
    
		} // end C column loops
	} // end C row loops

        timing_end();
}
// args:
//		tileRowPtr: tile ptr for the 1st tile in each row
//		tileNnz: ptr for the 1st non zero entry of each tile
// 		nnzTile: #nnz of each tile
// 		tileCol: column idx of each tile. 
//      rcOffset: row and column indexfor each non-zero entry
//		vals: non-zero entries
// 		m: height of sparseMat
// 		n: width of sparseMat
// 		mat_b: input dense mat
//		k: width of mat_b
//		mat_c: output dense mat
// A: sparse, m * n
// B: dense, n * k   (k << n)
template<int tm, int tn, int warps>
__global__
void flexspmm_cuda_wo_pre_v7(){
    const Mat_POD& md = mat_dev;
	const uint32_t WARPSZ = 32;
	const uint32_t lane_id = threadIdx.x % WARPSZ;
    const uint32_t warp_id = threadIdx.x / WARPSZ;
	//const uint32_t warps = (blockDim.x + WARPSZ - 1)/WARPSZ;

    timing_start();

	// now we restrain "tn" in {4,8,16,32}
	//__shared__ float curB[warps][tn*32]; // 2 warps && each warp needs tn*8*4 matB float entries
	float res[tm];
	#pragma unroll
	for (int i=0; i<tm; ++i){
		res[i] = 0;
	}

	int computeWidth = 1; // # of C entries to be computed by a thread
	int tileRows_perBlk = 1; // # row tiles per block
	for (int row_idx=blockIdx.x*tileRows_perBlk; row_idx<(md.m+tm-1)/tm; row_idx += (gridDim.x*tileRows_perBlk)){ // over C rows
	   
        int tile_curR_id = 0, tile_nxtR_id = 0;
        tile_curR_id = md.tileRowPtr_dev[row_idx]; 
        tile_nxtR_id = md.tileRowPtr_dev[row_idx+1]; 

        for (int col_idx=warp_id*(32*computeWidth); col_idx<md.k; col_idx += warps*(32*computeWidth)){  // over C tile columns
            
            int tiles = 0;

            for (int tile_id=tile_curR_id; tile_id<tile_nxtR_id; tile_id+=tiles){

                uint32_t mask_tiles = __ballot_sync(FULL_MASK, tile_id+lane_id<tile_nxtR_id);
                tiles = __popc(mask_tiles); // maximum # tiles can be loaded in cur row 
                
                int start_of_tile = 0, nnz_of_tile = 0, col_of_tile = 0;
                if (tile_curR_id+lane_id<tile_nxtR_id){
                    // load as many as as tile info of cur tile-row
                    start_of_tile = md.tileNnz_dev[tile_id+lane_id];
                    nnz_of_tile = md.nnzTile_dev[tile_id+lane_id];
                    //bitmap_of_tile = md.bitMap_dev[tile_id+lane_id];
                    col_of_tile = md.tileColIdx_dev[tile_id+lane_id];
                }

                // use all loaded tiles
                for(int tile_cnt = 0; tile_cnt<tiles; ++tile_cnt){
                    int start_cur_tile = __shfl_sync(FULL_MASK, start_of_tile, tile_cnt);
                    int nnz_cur_tile = __shfl_sync(FULL_MASK, nnz_of_tile, tile_cnt);
                    //int bitmap_cur_tile = __shfl_sync(FULL_MASK, bitmap_of_tile, tile_cnt);
                    int col_cur_tile = __shfl_sync(FULL_MASK, col_of_tile, tile_cnt);
                    auto do_n = [&](int n)
                     {
                       for ( int z=0; z<n; z++ )
                         {
                           float val = md.vals_dev[start_cur_tile+z];
                           int rcidx = md.rcOffset_dev[start_cur_tile+z];
                           int x_row = col_cur_tile + (rcidx & 0xffff);
                           res[rcidx>>16] += val
                             * md.mat_b_dev[x_row*md.k + col_idx + lane_id];
                         }
                     };
					// visit all nz of the sparse tile
                    if (nnz_cur_tile==1){
                        do_n(1);
                    }else if (nnz_cur_tile==2){    
                        do_n(2);
                    }else if (nnz_cur_tile==3){
                        do_n(3);
                    }else if(nnz_cur_tile==4){
                        do_n(4);
                    }else{ 
                        int steps = 1;
                        int cur_end = start_cur_tile+nnz_cur_tile;
                        for (int kk=start_cur_tile; kk<cur_end; kk+=steps){
                            uint32_t mask_join = __ballot_sync(FULL_MASK, kk+lane_id<cur_end);
                            steps = __popc(mask_join);

                            float val = 0;
                            int rcidx = 0;
                            if (kk+lane_id<cur_end){
                                // load sparse nnz from glb mem
                                val = md.vals_dev[kk+lane_id];
                                rcidx = md.rcOffset_dev[kk+lane_id];
                            }
                            // exchange nnz within a warp && perfom FMA
                            for (int it=0; it<steps; ++it){
                                float v = __shfl_sync(FULL_MASK, val, it);
                                int rc = __shfl_sync(FULL_MASK, rcidx, it);

                                //res[rc>>16] += v * curB[warp_id][(rc & 0x0000ffff)*32 + lane_id];
                                res[rc>>16] += v * md.mat_b_dev[(col_cur_tile+(rc & 0xffff))*md.k + col_idx + lane_id];
                            }
                        }// end visiting all nz in a sparse tile
                    }
                }// end visiting all loaded sparse tiles
            }// end visiting all sparse tiles in cur tile-row
            
			// store C tiles back to global mem
            //#pragma unroll
            for (int c=0; c<tm; ++c){
                if (row_idx*tm+c<md.m){
                    md.mat_c_dev[(row_idx*tm+c)*md.k+col_idx+lane_id] = res[c];
                }
                res[c] = 0;
            }
    
		} // end C column loops
	} // end C row loops

        timing_end();
}

struct RC16 {
  __host__ __device__
  RC16( uint32_t rcpacked ):r( rcpacked & 0xffff ),c( rcpacked >> 16 ){};
  const uint32_t r, c;
};

// args:
//		tileRowPtr: tile ptr for the 1st tile in each row
//		tileNnz: ptr for the 1st non zero entry of each tile
// 		nnzTile: #nnz of each tile
// 		tileCol: column idx of each tile. 
//      rcOffset: row and column indexfor each non-zero entry
//		vals: non-zero entries
// 		m: height of sparseMat
// 		n: width of sparseMat
// 		mat_b: input dense mat
//		k: width of mat_b
//		mat_c: output dense mat
// A: sparse, m * n
// B: dense, n * k   (k << n)
template<int tm, int tn, int warps>
__global__
void flexspmm_cuda_wo_pre_v8(){
    const Mat_POD& md = mat_dev;
	const uint32_t WARPSZ = 32;
	const uint32_t lane_id = threadIdx.x % WARPSZ;
    const uint32_t warp_id = threadIdx.x / WARPSZ;

    timing_start();

	float res[tm];
	#pragma unroll
	for (int i=0; i<tm; ++i){
		res[i] = 0;
	}

	int computeWidth = 1; // # of C entries to be computed by a thread
	int tileRows_perBlk = 1; // # row tiles per block
	for (int row_idx=blockIdx.x*tileRows_perBlk; row_idx<(md.m+tm-1)/tm; row_idx += (gridDim.x*tileRows_perBlk)){ // over C rows
	   
        int tile_curR_id = 0, tile_nxtR_id = 0;
        tile_curR_id = md.tileRowPtr_dev[row_idx]; 
        tile_nxtR_id = md.tileRowPtr_dev[row_idx+1]; 

        for (int col_idx=warp_id*(32*computeWidth); col_idx<md.k; col_idx += warps*(32*computeWidth)){  // over C tile columns
            
            int tiles = 0;

            for (int tile_id=tile_curR_id; tile_id<tile_nxtR_id; tile_id+=tiles){

                uint32_t mask_tiles = __ballot_sync(FULL_MASK, tile_id+lane_id<tile_nxtR_id);
                tiles = __popc(mask_tiles); // maximum # tiles can be loaded in cur row 
                
                int start_of_tile = 0, nnz_of_tile = 0, col_of_tile = 0;
                if (tile_curR_id+lane_id<tile_nxtR_id){
                    // load as many as as tile info of cur tile-row
                    start_of_tile = md.tileNnz_dev[tile_id+lane_id];
                    nnz_of_tile = md.nnzTile_dev[tile_id+lane_id];
                    //bitmap_of_tile = md.bitMap_dev[tile_id+lane_id];
                    col_of_tile = md.tileColIdx_dev[tile_id+lane_id];
                }

                // use all loaded tiles
                for(int tile_cnt = 0; tile_cnt<tiles; ++tile_cnt){
                    int start_cur_tile = __shfl_sync(FULL_MASK, start_of_tile, tile_cnt);
                    int nnz_cur_tile = __shfl_sync(FULL_MASK, nnz_of_tile, tile_cnt);
                    int col_cur_tile = __shfl_sync(FULL_MASK, col_of_tile, tile_cnt);
                    // visit all nz of the sparse tile

                    const int n_rounds = nnz_cur_tile / WARPSZ;
                    for ( int rnd = 0;  rnd < n_rounds;  rnd++ )
                      {
                        // load sparse nnz from glb mem
                        const int vidx = start_cur_tile + rnd*WARPSZ + lane_id;
                        const float val = md.vals_dev[vidx];
                        const int rcidx = md.rcOffset_dev[vidx];

                        // exchange nnz within a warp && perfom FMA
                        for (int it=0; it<WARPSZ; ++it){
                          float v = __shfl_sync(FULL_MASK, val, it);
                          RC16 rc( __shfl_sync(FULL_MASK, rcidx, it) );
                          res[rc.c] +=
                            v * md.mat_b_dev[(col_cur_tile+rc.r)*md.k + col_idx + lane_id];
                        }
                      }

                    const uint nnz_remaining = nnz_cur_tile % WARPSZ;
                    const int vidx_base = start_cur_tile + n_rounds * WARPSZ;

                    auto do_n = [&](int n)
                     {
                       for ( int z=0; z<n; z++ )
                         {
                           float val = md.vals_dev[ vidx_base + z ];
                           RC16 rc( md.rcOffset_dev[ vidx_base + z ] );
                           int x_row = col_cur_tile + rc.r;
                           res[ rc.c ] +=
                             val * md.mat_b_dev[x_row*md.k + col_idx + lane_id];
                         }
                     };

                    if ( nnz_remaining ){
                      #define C5(n) C4(n) C4(n+16)
                      #define C4(n) C3(n) C3(n+8)
                      #define C3(n) C2(n) C2(n+4)
                      #define C2(n) C1(n) C1(n+2)
                      #define C1(n) C0(n) C0(n+1)
                      #define C0(n) case n: do_n(n); break;
                      switch ( nnz_remaining ) {
                        C2(1);  // This generates do_n(1) to do_n(4)
                      default: do_n( nnz_remaining ); break;
                      }
                      #undef C5
                      #undef C4
                      #undef C3
                      #undef C2
                      #undef C1
                      #undef C0
                    }

                }// end visiting all loaded sparse tiles
            }// end visiting all sparse tiles in cur tile-row
            
			// store C tiles back to global mem
            //#pragma unroll
            for (int c=0; c<tm; ++c){
                if (row_idx*tm+c<md.m){
                    md.mat_c_dev[(row_idx*tm+c)*md.k+col_idx+lane_id] = res[c];
                }
                res[c] = 0;
            }
    
		} // end C column loops
	} // end C row loops

        timing_end();
}
// args:
//		tileRowPtr: tile ptr for the 1st tile in each row
//		tileNnz: ptr for the 1st non zero entry of each tile
// 		nnzTile: #nnz of each tile
// 		tileCol: column idx of each tile. 
//      rcOffset: row and column indexfor each non-zero entry
//		vals: non-zero entries
// 		m: height of sparseMat
// 		n: width of sparseMat
// 		mat_b: input dense mat
//		k: width of mat_b
//		mat_c: output dense mat
// A: sparse, m * n
// B: dense, n * k   (k << n)
template<int tm, int tn, int warps>
__global__
void flexspmm_cuda_wo_pre_v9(){
    // requires preprocess dense mat B

    const Mat_POD& md = mat_dev;
	const uint32_t WARPSZ = 32;
	const uint32_t lane_id = threadIdx.x % WARPSZ;
    const uint32_t warp_id = threadIdx.x / WARPSZ;

    timing_start();

	float res[tm];
    int gold_row_id[tm];
	#pragma unroll
	for (int i=0; i<tm; ++i){
		res[i] = 0;
	}

	int computeWidth = 1; // # of C entries to be computed by a thread
	int tileRows_perBlk = 1; // # row tiles per block
	for (int row_idx=blockIdx.x*tileRows_perBlk; row_idx<(md.m+tm-1)/tm; row_idx += (gridDim.x*tileRows_perBlk)){ // over C rows
	   
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = row_idx * tm + i < md.m ? md.voMp_dev[row_idx*tm+i] : 0;
        }

        int tile_curR_id = 0, tile_nxtR_id = 0;
        tile_curR_id = md.tileRowPtr_dev[row_idx]; 
        tile_nxtR_id = md.tileRowPtr_dev[row_idx+1]; 

        for (int col_idx=warp_id*(32*computeWidth); col_idx<md.k; col_idx += warps*(32*computeWidth)){  // over C tile columns
            
            int tiles = 0;

            for (int tile_id=tile_curR_id; tile_id<tile_nxtR_id; tile_id+=tiles){

                uint32_t mask_tiles = __ballot_sync(FULL_MASK, tile_id+lane_id<tile_nxtR_id);
                tiles = __popc(mask_tiles); // maximum # tiles can be loaded in cur row 
                
                int start_of_tile = 0, nnz_of_tile = 0, col_of_tile = 0;
                if (tile_curR_id+lane_id<tile_nxtR_id){
                    // load as many as as tile info of cur tile-row
                    start_of_tile = md.tileNnz_dev[tile_id+lane_id];
                    nnz_of_tile = md.nnzTile_dev[tile_id+lane_id];
                    //bitmap_of_tile = md.bitMap_dev[tile_id+lane_id];
                    col_of_tile = md.tileColIdx_dev[tile_id+lane_id];
                }

                // use all loaded tiles
                for(int tile_cnt = 0; tile_cnt<tiles; ++tile_cnt){
                    int start_cur_tile = __shfl_sync(FULL_MASK, start_of_tile, tile_cnt);
                    int nnz_cur_tile = __shfl_sync(FULL_MASK, nnz_of_tile, tile_cnt);
                    //int bitmap_cur_tile = __shfl_sync(FULL_MASK, bitmap_of_tile, tile_cnt);
                    int col_cur_tile = __shfl_sync(FULL_MASK, col_of_tile, tile_cnt);
                    
                    const int n_rounds = nnz_cur_tile / WARPSZ;
                    for ( int rnd = 0;  rnd < n_rounds;  rnd++ )
                      {
                        // load sparse nnz from glb mem
                        const int vidx = start_cur_tile + rnd*WARPSZ + lane_id;
                        const float val = md.vals_dev[vidx];
                        const int rcidx = md.rcOffset_dev[vidx];

                        // exchange nnz within a warp && perfom FMA
                        for (int it=0; it<WARPSZ; ++it){
                          float v = __shfl_sync(FULL_MASK, val, it);
                          RC16 rc( __shfl_sync(FULL_MASK, rcidx, it) );
                          res[rc.c] +=
                            v * md.shadow_b_dev[(col_cur_tile+rc.r)*md.k + col_idx + lane_id];
                        }
                      }

                    const uint nnz_remaining = nnz_cur_tile % WARPSZ;
                    const int vidx_base = start_cur_tile + n_rounds * WARPSZ;

                    auto do_n = [&](int n)
                     {
                       for ( int z=0; z<n; z++ )
                         {
                           float val = md.vals_dev[ vidx_base + z ];
                           RC16 rc( md.rcOffset_dev[ vidx_base + z ] );
                           int x_row = col_cur_tile + rc.r;
                           res[ rc.c ] +=
                             val * md.shadow_b_dev[x_row*md.k + col_idx + lane_id];
                         }
                     };

                    if ( nnz_remaining ){
                      #define C5(n) C4(n) C4(n+16)
                      #define C4(n) C3(n) C3(n+8)
                      #define C3(n) C2(n) C2(n+4)
                      #define C2(n) C1(n) C1(n+2)
                      #define C1(n) C0(n) C0(n+1)
                      #define C0(n) case n: do_n(n); break;
                      switch ( nnz_remaining ) {
                        C2(1);  // This generates do_n(1) to do_n(4)
                      default: do_n( nnz_remaining ); break;
                      }
                      #undef C5
                      #undef C4
                      #undef C3
                      #undef C2
                      #undef C1
                      #undef C0
                    }    
                }// end visiting all loaded sparse tiles
            }// end visiting all sparse tiles in cur tile-row
            
			// store C tiles back to global mem
            //#pragma unroll
            for (int c=0; c<tm; ++c){
                if (row_idx*tm+c<md.m){
                    // Not sure if gold_row_id here would incur "pointer chasing"
                    md.mat_c_dev[gold_row_id[c]*md.k+col_idx+lane_id] = res[c];
                }
                res[c] = 0;
            }
    
		} // end C column loops
	} // end C row loops

        timing_end();
}
template<int tm, int tn, int warps>
__global__
void flexspmm_cuda_wo_pre_v10(){
    // requires preprocess dense mat B

    const Mat_POD& md = mat_dev;
	const uint32_t WARPSZ = 32;
	const uint32_t lane_id = threadIdx.x % WARPSZ;

    timing_start();

    int gold_row_id[tm];
    
    for ( int seg_idx=blockIdx.x; seg_idx<md.n_segs; seg_idx += gridDim.x ){ // over  tile segments
        
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = md.segVoMap_dev[seg_idx*tm+i];
        }
        int seg_cur_id = 0, seg_nxt_id = 0;
        seg_cur_id = md.segPtr_dev[seg_idx]; 
        seg_nxt_id = md.segPtr_dev[seg_idx+1]; 
        int nnz_cur_seg = seg_nxt_id - seg_cur_id;
        const int n_rounds = nnz_cur_seg / WARPSZ; 
        
        for ( int c_col=threadIdx.x; c_col<md.k; c_col += blockDim.x ){ // over C columns
            
            float res[tm]{};

            for ( int rnd = 0; rnd < n_rounds; ++rnd ){

                // load sparse nz from glb mem
                const int vidx = seg_cur_id + rnd*WARPSZ + lane_id;
                const float val = md.vals_dev[vidx];
                const int ridx = md.segNzRowIdx_dev[vidx];
                const int cidx = md.segNzColIdx_dev[vidx];
                
                // exchange nnz within a warp && perfom FMA
                for (int it=0; it<WARPSZ; ++it){
                  float v = __shfl_sync(FULL_MASK, val, it);
                  int v_r = __shfl_sync(FULL_MASK, ridx, it);
                  int v_c = __shfl_sync(FULL_MASK, cidx, it);
                  res[v_r] +=
                    v * md.shadow_b_dev[ v_c*md.k + c_col ];
                }    
            }
            
            const uint nnz_remaining = nnz_cur_seg % WARPSZ;
            const int vidx_base = seg_cur_id + n_rounds * WARPSZ;
            
            auto do_n = [&](int n)
             {
               for ( int z=0; z<n; z++ )
                 {
                   float val = md.vals_dev[ vidx_base + z ];
                   int ridx = md.segNzRowIdx_dev[ vidx_base + z ];
                   int cidx = md.segNzColIdx_dev[ vidx_base + z ];
                   
                   res[ ridx ] +=
                     val * md.shadow_b_dev[ cidx*md.k + c_col];
                 }
             };

            if ( nnz_remaining ){
              #define C5(n) C4(n) C4(n+16)
              #define C4(n) C3(n) C3(n+8)
              #define C3(n) C2(n) C2(n+4)
              #define C2(n) C1(n) C1(n+2)
              #define C1(n) C0(n) C0(n+1)
              #define C0(n) case n: do_n(n); break;
              switch ( nnz_remaining ) {
                C2(1);  // This generates do_n(1) to do_n(4)
              default: do_n( nnz_remaining ); break;
              }
              #undef C5
              #undef C4
              #undef C3
              #undef C2
              #undef C1
              #undef C0
            } 
            
            // store C tiles back to global mem
            //#pragma unroll
            for ( int c=0; c<tm; ++c ){
                int actual_row = gold_row_id[ c ] & 0x7fffffff;
                 
                if ( actual_row<md.m ){
                    int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                    int addr = actual_row*md.k + c_col;
                    if ( atomicORnot>>31 ){
                        atomicAdd( &md.mat_c_dev[ addr ], res[c]);
                    }else{
                        md.mat_c_dev[ addr ] = res[c];
                    }
                }
            }
             
        }// end C colums
    } // end tile-segs loops
        timing_end();
}

template<int tm, int tn, int warps>
__global__
void flexspmm_cuda_w_vec4_v11(){
    // requires preprocess dense mat B

    const Mat_POD& md = mat_dev;
	const uint32_t WARPSZ = 32;
    //const uint32_t wps = blockDim.x / WARPSZ;
	const uint32_t lane_id = threadIdx.x % WARPSZ;
    //const uint32_t warp_id = threadIdx.x / WARPSZ;

    timing_start();

    int gold_row_id[tm];
	//#pragma unroll
	//for (int i=0; i<tm; ++i){
	//	res[i][0] = 0;
	//	res[i][1] = 0;
	//	res[i][2] = 0;
	//	res[i][3] = 0;
	//}
    
    for ( int seg_idx=blockIdx.x; seg_idx<md.n_segs; seg_idx += gridDim.x ){ // over  tile segments
        
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = md.segVoMap_dev[seg_idx*tm+i];
        }
        int seg_cur_id = 0, seg_nxt_id = 0;
        seg_cur_id = md.segPtr_dev[seg_idx]; 
        seg_nxt_id = md.segPtr_dev[seg_idx+1]; 
        int nnz_cur_seg = seg_nxt_id - seg_cur_id;
        const int n_rounds = nnz_cur_seg / WARPSZ; 
       
        for ( int c_col=threadIdx.x; c_col<md.k/4; c_col += blockDim.x ){ // over C columns
	        float res[tm][4]{};
               
            for ( int rnd = 0; rnd < n_rounds; ++rnd ){

                // load sparse nz from glb mem
                const int vidx = seg_cur_id + rnd*WARPSZ + lane_id;
                const float val = md.vals_dev[vidx];
                const int ridx = md.segNzRowIdx_dev[vidx];
                const int cidx = md.segNzColIdx_dev[vidx];
                
                // exchange nnz within a warp && perfom FMA
                for (int it=0; it<WARPSZ; ++it){
                  float v = __shfl_sync(FULL_MASK, val, it);
                  int v_r = __shfl_sync(FULL_MASK, ridx, it);
                  int v_c = __shfl_sync(FULL_MASK, cidx, it);
                  float *shadow_b_addr = &md.shadow_b_dev[ v_c*md.k ];
                  float4 b_vec = reinterpret_cast<float4*>(shadow_b_addr)[ c_col ];
                  res[v_r][0] += v * b_vec.x;
                  res[v_r][1] += v * b_vec.y;
                  res[v_r][2] += v * b_vec.z;
                  res[v_r][3] += v * b_vec.w;
                }    
            }
            
            const uint nnz_remaining = nnz_cur_seg % WARPSZ;
            const int vidx_base = seg_cur_id + n_rounds * WARPSZ;
            
            auto do_n = [&](int n)
             {
               for ( int z=0; z<n; z++ )
                 {
                   float val = md.vals_dev[ vidx_base + z ];
                   int ridx = md.segNzRowIdx_dev[ vidx_base + z ];
                   int cidx = md.segNzColIdx_dev[ vidx_base + z ];
                   
                   float *shadow_b_addr = &md.shadow_b_dev[ cidx*md.k ];
                   float4 b_vec = reinterpret_cast<float4*>(shadow_b_addr)[ c_col ];
                   res[ridx][0] += val * b_vec.x;
                   res[ridx][1] += val * b_vec.y;
                   res[ridx][2] += val * b_vec.z;
                   res[ridx][3] += val * b_vec.w;
                 }
             };

            if ( nnz_remaining ){
              #define C5(n) C4(n) C4(n+16)
              #define C4(n) C3(n) C3(n+8)
              #define C3(n) C2(n) C2(n+4)
              #define C2(n) C1(n) C1(n+2)
              #define C1(n) C0(n) C0(n+1)
              #define C0(n) case n: do_n(n); break;
              switch ( nnz_remaining ) {
                C2(1);  // This generates do_n(1) to do_n(4)
              default: do_n( nnz_remaining ); break;
              }
              #undef C5
              #undef C4
              #undef C3
              #undef C2
              #undef C1
              #undef C0
            } 
            
            // store C tiles back to global mem
            //#pragma unroll
            for ( int c=0; c<tm; ++c ){
                int actual_row = gold_row_id[ c ] & 0x7fffffff;
                 
                if ( actual_row<md.m ){
                    int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                    int addr = actual_row*md.k;
                    if ( atomicORnot>>31 ){
                        atomicAdd( &md.mat_c_dev[ addr + c_col*4 + 0 ], res[c][0]);
                        atomicAdd( &md.mat_c_dev[ addr + c_col*4 + 1 ], res[c][1]);
                        atomicAdd( &md.mat_c_dev[ addr + c_col*4 + 2 ], res[c][2]);
                        atomicAdd( &md.mat_c_dev[ addr + c_col*4 + 3 ], res[c][3]);
                    }else{
                        float* mat_c = &md.mat_c_dev[ addr ];
                        float4 vect4_c = {res[c][0], res[c][1], res[c][2], res[c][3]}; 
                        reinterpret_cast<float4*>(mat_c)[ c_col ] = vect4_c; 
                        //md.mat_c_dev[ addr + c_col*4 + 0 ] = res[c][0];
                        //md.mat_c_dev[ addr + c_col*4 + 1 ] = res[c][1];
                        //md.mat_c_dev[ addr + c_col*4 + 2 ] = res[c][2];
                        //md.mat_c_dev[ addr + c_col*4 + 3 ] = res[c][3];
                    }
                }
                //res[c][0] = 0;
                //res[c][1] = 0;
                //res[c][2] = 0;
                //res[c][3] = 0;
            }
             
        }// end C colums
    } // end tile-segs loops
        timing_end();
}
template<int tm, int nnz_limit, int warps>
__global__
void flexspmm_cuda_w_pre_v12(){ 
    // requires preprocess dense mat B

    const Mat_POD& md = mat_dev;
    
    timing_start(); 
    
    int gold_row_id[tm];
    __shared__ int smem[2][3*(nnz_limit)+3*tm];
    int *rsm1 = reinterpret_cast<int*>(smem[0]);
    int *csm1 = reinterpret_cast<int*>(&smem[0][1*(nnz_limit+tm)]);
    float *vsm1 = reinterpret_cast<float*>(&smem[0][2*(nnz_limit+tm)]);
    
    int *rsm2 = reinterpret_cast<int*>(smem[1]);
    int *csm2 = reinterpret_cast<int*>(&smem[1][1*(nnz_limit+tm)]);
    float *vsm2 = reinterpret_cast<float*>(&smem[1][2*(nnz_limit+tm)]);
   
    int seg_cur_id = 0, seg_nxt_id = 0, nnz_cur_seg = 0;
    // preload the 1st tile-seg
    if ( blockIdx.x < md.n_segs ) {    
        seg_cur_id = md.segPtr_dev[ blockIdx.x ]; 
        seg_nxt_id = md.segPtr_dev[ blockIdx.x+1 ]; 

        for ( int i=seg_cur_id+threadIdx.x; i<seg_nxt_id; i += blockDim.x ){
            rsm1[ i-seg_cur_id ] = md.segNzRowIdx_dev[ i ];
            csm1[ i-seg_cur_id ] = md.segNzColIdx_dev[ i ];
            vsm1[ i-seg_cur_id ] = md.vals_dev[ i ];
            
        }
    }
    __syncthreads();
    
    for ( int seg_idx=blockIdx.x; seg_idx<md.n_segs; seg_idx += gridDim.x ){ // over  tile segments
                   
        nnz_cur_seg = seg_nxt_id - seg_cur_id;
        
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = md.segVoMap_dev[seg_idx*tm+i];
        }
        
        // preload next tile-seg
        if ( seg_idx + gridDim.x < md.n_segs ) {    
            seg_cur_id = md.segPtr_dev[ seg_idx + gridDim.x ]; 
            seg_nxt_id = md.segPtr_dev[ seg_idx + gridDim.x + 1 ]; 
            
            for ( int i=seg_cur_id+threadIdx.x; i<seg_nxt_id; i += blockDim.x ){
                rsm2[ i-seg_cur_id ] = md.segNzRowIdx_dev[ i ];
                csm2[ i-seg_cur_id ] = md.segNzColIdx_dev[ i ];
                vsm2[ i-seg_cur_id ] = md.vals_dev[ i ]; 
            }
        }

        for ( int c_col=threadIdx.x; c_col<md.k; c_col += blockDim.x ){ // over C columns
	        //float res[tm][4]{};
            float res[tm]{};    
            auto do_n = [&](int n)
             {
               for ( int z=0; z<n; z++ )
                 {
                   float val = vsm1[ z ];
                   int ridx = rsm1[ z ];
                   int cidx = csm1[ z ];
                   res[ridx] += md.shadow_b_dev[ cidx * md.k + c_col ] * val;
                   //float *shadow_b_addr = &md.shadow_b_dev[ cidx*md.k ];
                   //float4 b_vec = reinterpret_cast<float4*>(shadow_b_addr)[ c_col ];
                   //res[ridx][0] += val * b_vec.x;
                   //res[ridx][1] += val * b_vec.y;
                   //res[ridx][2] += val * b_vec.z;
                   //res[ridx][3] += val * b_vec.w;
                 }
             };
            do_n( nnz_cur_seg );
            
            // store C tiles back to global mem
            //#pragma unroll
            for ( int c=0; c<tm; ++c ){
                int actual_row = gold_row_id[ c ] & 0x7fffffff;
                 
                if ( actual_row<md.m ){
                    int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                    int addr = actual_row*md.k;
                    if ( atomicORnot>>31 ){
                        atomicAdd( &md.mat_c_dev[ addr + c_col], res[c] );
                        //atomicAdd( &md.mat_c_dev[ addr + c_col*4 + 0 ], res[c][0]);
                        //atomicAdd( &md.mat_c_dev[ addr + c_col*4 + 1 ], res[c][1]);
                        //atomicAdd( &md.mat_c_dev[ addr + c_col*4 + 2 ], res[c][2]);
                        //atomicAdd( &md.mat_c_dev[ addr + c_col*4 + 3 ], res[c][3]);
                    }else{
                        md.mat_c_dev[ addr + c_col ] = res[ c ];
                        //float* mat_c = &md.mat_c_dev[ addr ];
                        //float4 vect4_c = {res[c][0], res[c][1], res[c][2], res[c][3]}; 
                        //reinterpret_cast<float4*>(mat_c)[ c_col ] = vect4_c; 
                    }
                }
            }
         
        }// end C colums

        // switch buffer
        int *r_temp = rsm1;
        int *c_temp = csm1;
        float *v_temp = vsm1;


        __syncthreads();
        rsm1 = rsm2;
        csm1 = csm2;
        vsm1 = vsm2;

        
        rsm2 = r_temp;
        csm2 = c_temp;
        vsm2 = v_temp;
        
        // failed, have't figured out why 
        //dbl ^= 1;
        //rsm2 = reinterpret_cast<int*>(&smem[dbl][0]);
        //csm2 = reinterpret_cast<int*>(&smem[dbl][1*(nnz_limit+12)*4]);
        //vsm2 = reinterpret_cast<float*>(&smem[dbl][2*(nnz_limit+12)*4]);
         
    } // end tile-segs loops
    
        timing_end();
}

template<int tm, int nnz_limit, int warps>
__global__
void flexspmm_cuda_w_pre_w_vec_v13(){ 
    // requires preprocess dense mat B

    const Mat_POD& md = mat_dev;
	const uint32_t WARPSZ = 32;
    
    timing_start(); 
    
    int gold_row_id[tm];
    __shared__ int smem[2][3*(nnz_limit)+3*tm];
    int dbl = 0;
    int *rsm1 = reinterpret_cast<int*>(smem[0]);
    int *csm1 = reinterpret_cast<int*>(&smem[0][1*(nnz_limit+tm)]);
    float *vsm1 = reinterpret_cast<float*>(&smem[0][2*(nnz_limit+tm)]);
    
    int *rsm2 = reinterpret_cast<int*>(smem[1]);
    int *csm2 = reinterpret_cast<int*>(&smem[1][1*(nnz_limit+tm)]);
    float *vsm2 = reinterpret_cast<float*>(&smem[1][2*(nnz_limit+tm)]);
   
    int seg_cur_id = 0, seg_nxt_id = 0, nnz_cur_seg = 0;
    
    // preload the 1st tile-seg
    if ( blockIdx.x < md.n_segs ) {    
        seg_cur_id = md.segPtr_dev[ blockIdx.x ]; 
        seg_nxt_id = md.segPtr_dev[ blockIdx.x+1 ]; 

        for ( int i=seg_cur_id+threadIdx.x; i<seg_nxt_id; i += blockDim.x ){
            rsm1[ i-seg_cur_id ] = md.segNzRowIdx_dev[ i ];
            csm1[ i-seg_cur_id ] = md.segNzColIdx_dev[ i ];
            vsm1[ i-seg_cur_id ] = md.vals_dev[ i ];
            
        }
    }
    __syncthreads();
    
    for ( int seg_idx=blockIdx.x; seg_idx<md.n_segs; seg_idx += gridDim.x ){ // over  tile segments
                   
        nnz_cur_seg = seg_nxt_id - seg_cur_id;
        
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = md.segVoMap_dev[seg_idx*tm+i];
        }
        
        // preload next tile-seg
        if ( seg_idx + gridDim.x < md.n_segs ) {    
            seg_cur_id = md.segPtr_dev[ seg_idx + gridDim.x ]; 
            seg_nxt_id = md.segPtr_dev[ seg_idx + gridDim.x + 1 ]; 
            
            for ( int i=seg_cur_id+threadIdx.x; i<seg_nxt_id; i += blockDim.x ){
                rsm2[ i-seg_cur_id ] = md.segNzRowIdx_dev[ i ];
                csm2[ i-seg_cur_id ] = md.segNzColIdx_dev[ i ];
                vsm2[ i-seg_cur_id ] = md.vals_dev[ i ]; 
            }
        }

        for ( int c_col=threadIdx.x; c_col<md.k/2; c_col += blockDim.x ){ // over C columns
	        float res[tm][2]{};
            
            auto do_n = [&](int n)
             {
               for ( int z=0; z<n; z++ )
                 {
                   float val = vsm1[ z ];
                   int ridx = rsm1[ z ];
                   int cidx = csm1[ z ];
                   //res[ridx] += md.shadow_b_dev[ cidx * md.k + c_col ] * val;
                   float *shadow_b_addr = &md.shadow_b_dev[ cidx*md.k ];
                   float2 b_vec = reinterpret_cast<float2*>(shadow_b_addr)[ c_col ];
                   res[ridx][0] += val * b_vec.x;
                   res[ridx][1] += val * b_vec.y;
                 }
             };
            do_n( nnz_cur_seg );
            
            // store C tiles back to global mem
            //#pragma unroll
            for ( int c=0; c<tm; ++c ){
                int actual_row = gold_row_id[ c ] & 0x7fffffff;
                 
                if ( actual_row<md.m ){
                    int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                    int addr = actual_row*md.k;
                    if ( atomicORnot>>31 ){
                        //atomicAdd( &md.mat_c_dev[ addr + c_col], res[c] );
                        atomicAdd( &md.mat_c_dev[ addr + c_col*2 + 0 ], res[c][0]);
                        atomicAdd( &md.mat_c_dev[ addr + c_col*2 + 1 ], res[c][1]);
                    }else{
                        //md.mat_c_dev[ addr + c_col ] = res[ c ];
                        float* mat_c = &md.mat_c_dev[ addr ];
                        float2 vect2_c = {res[c][0], res[c][1]}; 
                        reinterpret_cast<float2*>(mat_c)[ c_col ] = vect2_c; 
                    }
                }
            }
         
        }// end C colums

        // switch buffer
        int *r_temp = rsm1;
        int *c_temp = csm1;
        float *v_temp = vsm1;


        __syncthreads();
        rsm1 = rsm2;
        csm1 = csm2;
        vsm1 = vsm2;

        
        rsm2 = r_temp;
        csm2 = c_temp;
        vsm2 = v_temp;
        
        // failed, have't figured out why 
        //dbl ^= 1;
        //rsm2 = reinterpret_cast<int*>(&smem[dbl][0]);
        //csm2 = reinterpret_cast<int*>(&smem[dbl][1*(nnz_limit+12)*4]);
        //vsm2 = reinterpret_cast<float*>(&smem[dbl][2*(nnz_limit+12)*4]);
         
    } // end tile-segs loops
    
        timing_end();
}

template<int tm, int nnz_limit, int warps>
__global__
void flexspmm_cuda_w_pre_w_vec_v14(){ 
    // requires preprocess dense mat B

    const Mat_POD& md = mat_dev;
    
    timing_start(); 
    
    int gold_row_id[tm];
    __shared__ int smem[2][3*(nnz_limit)+3*tm];
    //int dbl = 0;
    int *rsm1 = reinterpret_cast<int*>(smem[0]);
    int *csm1 = reinterpret_cast<int*>(&smem[0][1*(nnz_limit+tm)]);
    float *vsm1 = reinterpret_cast<float*>(&smem[0][2*(nnz_limit+tm)]);
    
    int *rsm2 = reinterpret_cast<int*>(smem[1]);
    int *csm2 = reinterpret_cast<int*>(&smem[1][1*(nnz_limit+tm)]);
    float *vsm2 = reinterpret_cast<float*>(&smem[1][2*(nnz_limit+tm)]);
   
    int seg_cur_id = 0, seg_nxt_id = 0, nnz_cur_seg = 0;
    
    // preload the 1st tile-seg
    if ( blockIdx.x < md.n_segs ) {    
        //int2* seg_ids_ld = (int2*)(&md.segPtr_dev[ blockIdx.x ]);
        //int2 seg_ids = seg_ids_ld[0];
        seg_cur_id = md.segPtr_dev[ blockIdx.x ]; 
        seg_nxt_id = md.segPtr_dev[ blockIdx.x+1 ]; 

        for ( int i=seg_cur_id+threadIdx.x; i<seg_nxt_id; i += blockDim.x ){
            int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[i];
            rsm1[ i-seg_cur_id ] = rc.x;
            csm1[ i-seg_cur_id ] = rc.y;
            vsm1[ i-seg_cur_id ] = md.vals_dev[ i ];
            
        }
    }
    __syncthreads();
    
    for ( int seg_idx=blockIdx.x; seg_idx<md.n_segs; seg_idx += gridDim.x ){ // over  tile segments
                   
        nnz_cur_seg = seg_nxt_id - seg_cur_id;
        
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = md.segVoMap_dev[seg_idx*tm+i];
        }
        
        // preload next tile-seg
        if ( seg_idx + gridDim.x < md.n_segs ) {    
            //int2* seg_ids_ld = (int2*)(&md.segPtr_dev[ seg_idx+blockIdx.x ]);
            //int2 seg_ids = seg_ids_ld[0];
            seg_cur_id = md.segPtr_dev[ seg_idx+gridDim.x ]; 
            seg_nxt_id = md.segPtr_dev[ seg_idx+gridDim.x+1 ]; 
            
            for ( int i=seg_cur_id+threadIdx.x; i<seg_nxt_id; i += blockDim.x ){
                int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[i];
                rsm2[ i-seg_cur_id ] = rc.x;
                csm2[ i-seg_cur_id ] = rc.y;
                vsm2[ i-seg_cur_id ] = md.vals_dev[ i ]; 
            }
        }

        for ( int c_col=threadIdx.x; c_col<md.k; c_col += blockDim.x ){ // over C columns
	        float res[tm]{};
            
            auto do_n = [&](int n)
             {
               for ( int z=0; z<n; z++ )
                 {
                   float val = vsm1[ z ];
                   int ridx = rsm1[ z ];
                   int cidx = csm1[ z ];
                   res[ridx] += md.shadow_b_dev[ cidx * md.k + c_col ] * val;
                   //float *shadow_b_addr = &md.shadow_b_dev[ cidx*md.k ];
                   //float2 b_vec = reinterpret_cast<float2*>(shadow_b_addr)[ c_col ];
                   //res[ridx][0] += val * b_vec.x;
                   //res[ridx][1] += val * b_vec.y;
                 }
             };
            do_n( nnz_cur_seg );
            
            // store C tiles back to global mem
            #pragma unroll
            for ( int c=0; c<tm; ++c ){
                int actual_row = gold_row_id[ c ] & 0x7fffffff;
                 
                if ( actual_row<md.m ){
                    int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                    int addr = actual_row*md.k;
                    if ( atomicORnot>>31 ){
                        atomicAdd( &md.mat_c_dev[ addr + c_col], res[c] );
                        //atomicAdd( &md.mat_c_dev[ addr + c_col*2 + 0 ], res[c][0]);
                        //atomicAdd( &md.mat_c_dev[ addr + c_col*2 + 1 ], res[c][1]);
                    }else{
                        md.mat_c_dev[ addr + c_col ] = res[ c ];
                        //float* mat_c = &md.mat_c_dev[ addr ];
                        //float2 vect2_c = {res[c][0], res[c][1]}; 
                        //reinterpret_cast<float2*>(mat_c)[ c_col ] = vect2_c; 
                    }
                }
            }
         
        }// end C colums

        // switch buffer
        int *r_temp = rsm1;
        int *c_temp = csm1;
        float *v_temp = vsm1;


        __syncthreads();
        rsm1 = rsm2;
        csm1 = csm2;
        vsm1 = vsm2;

        
        rsm2 = r_temp;
        csm2 = c_temp;
        vsm2 = v_temp;
        
        // failed, have't figured out why 
        //dbl ^= 1;
        //rsm2 = reinterpret_cast<int*>(&smem[dbl][0]);
        //csm2 = reinterpret_cast<int*>(&smem[dbl][1*(nnz_limit+12)*4]);
        //vsm2 = reinterpret_cast<float*>(&smem[dbl][2*(nnz_limit+12)*4]);
         
    } // end tile-segs loops
    
        timing_end();
}

template<int tm, int nnz_limit, int warps>
__global__
void flexspmm_cuda_wo_pre_w_vec_v15(){ 
    // requires preprocess dense mat B

    const Mat_POD& md = mat_dev;
    
    timing_start(); 
    
    int gold_row_id[tm];
    __shared__ int smem[3*(nnz_limit)+3*tm];
    //int dbl = 0;
    int *rsm1 = reinterpret_cast<int*>(smem);
    int *csm1 = reinterpret_cast<int*>(&smem[1*(nnz_limit+tm)]);
    float *vsm1 = reinterpret_cast<float*>(&smem[2*(nnz_limit+tm)]);
    
    int seg_cur_id = 0, seg_nxt_id = 0, nnz_cur_seg = 0;
    
    for ( int seg_idx=blockIdx.x; seg_idx<md.n_segs; seg_idx += gridDim.x ){ // over  tile segments
                   
        
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = md.segVoMap_dev[seg_idx*tm+i];
        }
        
        seg_cur_id = md.segPtr_dev[ seg_idx ]; 
        seg_nxt_id = md.segPtr_dev[ seg_idx+1 ]; 
        nnz_cur_seg = seg_nxt_id - seg_cur_id;
        
        for ( int i=seg_cur_id+threadIdx.x; i<seg_nxt_id; i += blockDim.x ){
            int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[i];
            rsm1[ i-seg_cur_id ] = rc.x;
            csm1[ i-seg_cur_id ] = rc.y;
            vsm1[ i-seg_cur_id ] = md.vals_dev[ i ]; 
        }
        __syncthreads();

        for ( int c_col=threadIdx.x; c_col<md.k; c_col += blockDim.x ){ // over C columns
	        float res[tm]{};
            
            auto do_n = [&](int n)
             {
               for ( int z=0; z<n; z++ )
                 {
                   float val = vsm1[ z ];
                   int ridx = rsm1[ z ];
                   int cidx = csm1[ z ];
                   res[ridx] += md.shadow_b_dev[ cidx * md.k + c_col ] * val;
                   //float *shadow_b_addr = &md.shadow_b_dev[ cidx*md.k ];
                   //float2 b_vec = reinterpret_cast<float2*>(shadow_b_addr)[ c_col ];
                   //res[ridx][0] += val * b_vec.x;
                   //res[ridx][1] += val * b_vec.y;
                 }
             };
            do_n( nnz_cur_seg );
            
            // store C tiles back to global mem
            #pragma unroll
            for ( int c=0; c<tm; ++c ){
                int actual_row = gold_row_id[ c ] & 0x7fffffff;
                 
                if ( actual_row<md.m ){
                    int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                    int addr = actual_row*md.k;
                    if ( atomicORnot>>31 ){
                        atomicAdd( &md.mat_c_dev[ addr + c_col], res[c] );
                        //atomicAdd( &md.mat_c_dev[ addr + c_col*2 + 0 ], res[c][0]);
                        //atomicAdd( &md.mat_c_dev[ addr + c_col*2 + 1 ], res[c][1]);
                    }else{
                        md.mat_c_dev[ addr + c_col ] = res[ c ];
                        //float* mat_c = &md.mat_c_dev[ addr ];
                        //float2 vect2_c = {res[c][0], res[c][1]}; 
                        //reinterpret_cast<float2*>(mat_c)[ c_col ] = vect2_c; 
                    }
                }
            }
         
        }// end C colums
        __syncthreads();

    } // end tile-segs loops
    
        timing_end();
}

template<int tm>
__device__
void less_quarter_lim_ker(int seg_idx, int seg_cur_idx, int seg_nxt_idx, int nnz_cur_seg, int* gold_row_id){ 

    // Case1: nnz falls in ( 0 , limit/4 ) e.g. nnz_limit==128, nnz of cur seg falss in (0, 32)
    const Mat_POD& md = mat_dev;

    __shared__ int smem[3*32];
    int *rsm1 = reinterpret_cast<int*>(smem);
    int *csm1 = reinterpret_cast<int*>(&smem[32]);
    float *vsm1 = reinterpret_cast<float*>(&smem[2*32]);
    
    int i = seg_cur_idx + threadIdx.x;
    if ( i<seg_nxt_idx ){
        int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[i];
        rsm1[ i-seg_cur_idx ] = rc.x;
        csm1[ i-seg_cur_idx ] = rc.y;
        vsm1[ i-seg_cur_idx ] = md.vals_dev[ i ]; 
    }
    __syncthreads();

    for ( int c_col=threadIdx.x; c_col<md.k; c_col += blockDim.x ){ // over C columns
        float res[tm]{};
    
        auto do_n = [&](int n)
         {
           for ( int z=0; z<n; z++ )
             {
               //int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[seg_cur_idx+z];
               //float val = md.vals_dev[seg_cur_idx+z];
               float val = vsm1[z];
               int ridx = rsm1[z];
               int cidx = csm1[z];
               res[ridx] += md.shadow_b_dev[ cidx * md.k + c_col ] * val;
             }
         };
        do_n( nnz_cur_seg );
        
        // store C tiles back to global mem
        #pragma unroll
        for ( int c=0; c<tm; ++c ){
            int actual_row = gold_row_id[ c ] & 0x7fffffff;
             
            if ( actual_row<md.m ){
                int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                int addr = actual_row*md.k;
                if ( atomicORnot>>31 ){
                    atomicAdd( &md.mat_c_dev[ addr + c_col], res[c] );
                }else{
                    md.mat_c_dev[ addr + c_col ] = res[ c ];
                }
            }
        } 
        
    }// end C colums    
    
    __syncthreads();
}

template<int tm>
__device__
void less_half_lim_ker(int seg_idx, int seg_cur_idx, int seg_nxt_idx, int nnz_cur_seg, int* gold_row_id){ 

    // Case2: nnz falls in [ limit/4 , limit/2 ) e.g. nnz_limit==128, nnz of cur seg falss in [32, 64)
    const Mat_POD& md = mat_dev;
    
    
    __shared__ int smem[3*64];
    int *rsm1 = reinterpret_cast<int*>(smem);
    int *csm1 = reinterpret_cast<int*>(&smem[64]);
    float *vsm1 = reinterpret_cast<float*>(&smem[2*64]);
    
    int i = seg_cur_idx + threadIdx.x;
    if ( i<seg_nxt_idx ){
        int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[i];
        rsm1[ i-seg_cur_idx ] = rc.x;
        csm1[ i-seg_cur_idx ] = rc.y;
        vsm1[ i-seg_cur_idx ] = md.vals_dev[ i ]; 
    }
    __syncthreads();
    
    for ( int c_col=threadIdx.x; c_col<md.k; c_col += blockDim.x ){ // over C columns
        float res[tm]{};
        
        auto do_n = [&](int n)
         {
           for ( int z=0; z<n; z++ )
             {
               //int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[seg_cur_idx+z];
               //float val = md.vals_dev[seg_cur_idx+z];
               float val = vsm1[z];
               int ridx = rsm1[z];
               int cidx = csm1[z];
               res[ridx] += md.shadow_b_dev[ cidx * md.k + c_col ] * val;
             }
         };
        do_n( nnz_cur_seg );
        
        // store C tiles back to global mem
        #pragma unroll
        for ( int c=0; c<tm; ++c ){
            int actual_row = gold_row_id[ c ] & 0x7fffffff;
             
            if ( actual_row<md.m ){
                int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                int addr = actual_row*md.k;
                if ( atomicORnot>>31 ){
                    atomicAdd( &md.mat_c_dev[ addr + c_col], res[c] );
                }else{
                    md.mat_c_dev[ addr + c_col ] = res[ c ];
                }
            }
        } 
        
    }// end C colums    
    __syncthreads(); 
}
template<int tm>
__device__
void less_lim_ker(int seg_idx, int seg_cur_idx, int seg_nxt_idx, int nnz_cur_seg, int* gold_row_id){ 

    // Case3: nnz falls in [ limit/2 , limit ) e.g. nnz_limit==128, nnz of cur seg falss in [64, 128)
    const Mat_POD& md = mat_dev;
    
    
    __shared__ int smem[3*128];
    int *rsm1 = reinterpret_cast<int*>(smem);
    int *csm1 = reinterpret_cast<int*>(&smem[128]);
    float *vsm1 = reinterpret_cast<float*>(&smem[2*128]);
    
    int i = seg_cur_idx + threadIdx.x;
    if ( i<seg_nxt_idx ){
        int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[i];
        rsm1[ i-seg_cur_idx ] = rc.x;
        csm1[ i-seg_cur_idx ] = rc.y;
        vsm1[ i-seg_cur_idx ] = md.vals_dev[ i ]; 
    }
    __syncthreads();
    for ( int c_col=threadIdx.x; c_col<md.k; c_col += blockDim.x ){ // over C columns
        float res[tm]{};
        
        auto do_n = [&](int n)
         {
           for ( int z=0; z<n; z++ )
             {
               //int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[seg_cur_idx+z];
               //float val = md.vals_dev[seg_cur_idx+z];
               float val = vsm1[z];
               int ridx = rsm1[z];
               int cidx = csm1[z];
               res[ridx] += md.shadow_b_dev[ cidx * md.k + c_col ] * val;
             }
         };
        do_n( nnz_cur_seg );
        
        // store C tiles back to global mem
        #pragma unroll
        for ( int c=0; c<tm; ++c ){
            int actual_row = gold_row_id[ c ] & 0x7fffffff;
             
            if ( actual_row<md.m ){
                int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                int addr = actual_row*md.k;
                if ( atomicORnot>>31 ){
                    atomicAdd( &md.mat_c_dev[ addr + c_col], res[c] );
                }else{
                    md.mat_c_dev[ addr + c_col ] = res[ c ];
                }
            }
        } 
        
    }// end C colums    
    __syncthreads();
}

template<int tm, int nnz_limit, int warps>
__global__
void flexspmm_cuda_wo_pre_w_vec_v16(){ 
    // requires preprocess dense mat B

    const Mat_POD& md = mat_dev;
    
    timing_start(); 
    
    int gold_row_id[tm];
    
    
    for ( int seg_idx=blockIdx.x; seg_idx<md.n_segs; seg_idx += gridDim.x ){ // over  tile segments
                   
        
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = md.segVoMap_dev[seg_idx*tm+i];
        }
        
        int seg_cur_id = md.segPtr_dev[ seg_idx ]; 
        int seg_nxt_id = md.segPtr_dev[ seg_idx+1 ]; 
        int nnz_cur_seg = seg_nxt_id - seg_cur_id;
        
        if ( true && nnz_cur_seg<nnz_limit/4 ){
            // Case1: nnz falls in ( 0 , limit/4 ) 
            less_quarter_lim_ker<tm>(seg_idx, seg_cur_id, seg_nxt_id, nnz_cur_seg, gold_row_id);
        }else if ( true && nnz_cur_seg<=nnz_limit/2 ){
            // Case2: nnz falls in [ limit/4 , limit/2 ) 
            less_half_lim_ker<tm>(seg_idx, seg_cur_id, seg_nxt_id, nnz_cur_seg, gold_row_id);
        }else if ( true && nnz_cur_seg<=nnz_limit ){
            // Case3: nnz falls in [ limit/2 , limit ) 
            less_lim_ker<tm>(seg_idx, seg_cur_id, seg_nxt_id, nnz_cur_seg, gold_row_id);
        }else{
            // Case4: nnz falls in [ limit , limit+tm ) 
            __shared__ int smem[3*(nnz_limit)+3*tm];
            int *rsm1 = reinterpret_cast<int*>(smem);
            int *csm1 = reinterpret_cast<int*>(&smem[1*(nnz_limit+tm)]);
            float *vsm1 = reinterpret_cast<float*>(&smem[2*(nnz_limit+tm)]);
            for ( int i=seg_cur_id+threadIdx.x; i<seg_nxt_id; i += blockDim.x ){
                int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[i];
                rsm1[ i-seg_cur_id ] = rc.x;
                csm1[ i-seg_cur_id ] = rc.y;
                vsm1[ i-seg_cur_id ] = md.vals_dev[ i ]; 
            }
            __syncthreads();

            for ( int c_col=threadIdx.x; c_col<md.k; c_col += blockDim.x ){ // over C columns
                float res[tm]{};
                
                auto do_n = [&](int n)
                 {
                   for ( int z=0; z<n; z++ )
                     {
                       float val = vsm1[ z ];
                       int ridx = rsm1[ z ];
                       int cidx = csm1[ z ];
                       res[ridx] += md.shadow_b_dev[ cidx * md.k + c_col ] * val;
                     }
                 };
                do_n( nnz_cur_seg );
                
                // store C tiles back to global mem
                #pragma unroll
                for ( int c=0; c<tm; ++c ){
                    int actual_row = gold_row_id[ c ] & 0x7fffffff;
                     
                    if ( actual_row<md.m ){
                        int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                        int addr = actual_row*md.k;
                        if ( atomicORnot>>31 ){
                            atomicAdd( &md.mat_c_dev[ addr + c_col], res[c] );
                        }else{
                            md.mat_c_dev[ addr + c_col ] = res[ c ];
                        }
                    }
                }
             
            }// end C colums
        }
        //__syncthreads();
    } // end tile-segs loops
    
        timing_end();
}


template<int tm, int nnz_limit, int warps>
__global__
void flexspmm_cuda_wo_pre_w_vec_v17(){ 
    // requires preprocess dense mat B

    const Mat_POD& md = mat_dev;
    
    timing_start(); 
    
    int gold_row_id[tm];
    __shared__ int smem[3*(nnz_limit)+3*tm];
    //int dbl = 0;
    int *rsm1 = reinterpret_cast<int*>(smem);
    int *csm1 = reinterpret_cast<int*>(&smem[1*(nnz_limit+tm)]);
    float *vsm1 = reinterpret_cast<float*>(&smem[2*(nnz_limit+tm)]);
    
    int seg_cur_id = 0, seg_nxt_id = 0, nnz_cur_seg = 0;
    int segs_per_blk = (md.n_segs<gridDim.x) ? 3 : ( md.n_segs + gridDim.x - 1 ) / gridDim.x; 
    int bd = min((blockIdx.x+1)*segs_per_blk, md.n_segs);
    for ( int seg_idx=blockIdx.x*segs_per_blk; seg_idx<bd; seg_idx += 1 ){ // over  tile segments
                   
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = md.segVoMap_dev[seg_idx*tm+i];
        }
        
        seg_cur_id = md.segPtr_dev[ seg_idx ]; 
        seg_nxt_id = md.segPtr_dev[ seg_idx+1 ]; 
        nnz_cur_seg = seg_nxt_id - seg_cur_id;
        
        for ( int i=seg_cur_id+threadIdx.x; i<seg_nxt_id; i += blockDim.x ){
            int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[i];
            rsm1[ i-seg_cur_id ] = rc.x;
            csm1[ i-seg_cur_id ] = rc.y;
            vsm1[ i-seg_cur_id ] = md.vals_dev[ i ]; 
        }
        __syncthreads();

        for ( int c_col=threadIdx.x; c_col<md.k; c_col += blockDim.x ){ // over C columns
	        float res[tm]{};
            
            auto do_n = [&](int n)
             {
               for ( int z=0; z<n; z++ )
                 {
                   float val = vsm1[ z ];
                   int ridx = rsm1[ z ];
                   int cidx = csm1[ z ];
                   res[ridx] += md.shadow_b_dev[ cidx * md.k + c_col ] * val;
                   //float *shadow_b_addr = &md.shadow_b_dev[ cidx*md.k ];
                   //float2 b_vec = reinterpret_cast<float2*>(shadow_b_addr)[ c_col ];
                   //res[ridx][0] += val * b_vec.x;
                   //res[ridx][1] += val * b_vec.y;
                 }
             };
            do_n( nnz_cur_seg );
            
            // store C tiles back to global mem
            #pragma unroll
            for ( int c=0; c<tm; ++c ){
                int actual_row = gold_row_id[ c ] & 0x7fffffff;
                 
                if ( actual_row<md.m ){
                    int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                    int addr = actual_row*md.k;
                    if ( atomicORnot>>31 ){
                        atomicAdd( &md.mat_c_dev[ addr + c_col], res[c] );
                        //atomicAdd( &md.mat_c_dev[ addr + c_col*2 + 0 ], res[c][0]);
                        //atomicAdd( &md.mat_c_dev[ addr + c_col*2 + 1 ], res[c][1]);
                    }else{
                        md.mat_c_dev[ addr + c_col ] = res[ c ];
                        //float* mat_c = &md.mat_c_dev[ addr ];
                        //float2 vect2_c = {res[c][0], res[c][1]}; 
                        //reinterpret_cast<float2*>(mat_c)[ c_col ] = vect2_c; 
                    }
                }
            }
         
        }// end C colums
        __syncthreads();

    } // end tile-segs loops
    
        timing_end();
}


template<int tm, int nnz_limit, int warps>
__global__
void flexspmm_cuda_wo_pre_w_vec_v18(){ 
    // requires preprocess dense mat B

    const Mat_POD& md = mat_dev;
    
    timing_start(); 
    
    int gold_row_id[tm];
    
    int seg_cur_id = 0, seg_nxt_id = 0, nnz_cur_seg = 0;
    int segs_per_blk = (md.n_segs<gridDim.x) ? 3 : ( md.n_segs + gridDim.x - 1 ) / gridDim.x; 
    int bd = min((blockIdx.x+1)*segs_per_blk, md.n_segs);
    for ( int seg_idx=blockIdx.x*segs_per_blk; seg_idx<bd; seg_idx += 1 ){ // over  tile segments
                   
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = md.segVoMap_dev[seg_idx*tm+i];
        }
        
        seg_cur_id = md.segPtr_dev[ seg_idx ]; 
        seg_nxt_id = md.segPtr_dev[ seg_idx+1 ]; 
        nnz_cur_seg = seg_nxt_id - seg_cur_id;
        

        for ( int c_col=threadIdx.x; c_col<md.k; c_col += blockDim.x ){ // over C columns
	        float res[tm]{};
            
            auto do_n = [&](int n)
             {
               for ( int z=0; z<n; z++ )
                 {
                   int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[seg_cur_id+z];
                   float val = md.vals_dev[ seg_cur_id+z ];
                   int ridx = rc.x;
                   int cidx = rc.y;
                   res[ridx] += md.shadow_b_dev[ cidx * md.k + c_col ] * val;
                   //float *shadow_b_addr = &md.shadow_b_dev[ cidx*md.k ];
                   //float2 b_vec = reinterpret_cast<float2*>(shadow_b_addr)[ c_col ];
                   //res[ridx][0] += val * b_vec.x;
                   //res[ridx][1] += val * b_vec.y;
                 }
             };
            do_n( nnz_cur_seg );
            
            // store C tiles back to global mem
            #pragma unroll
            for ( int c=0; c<tm; ++c ){
                int actual_row = gold_row_id[ c ] & 0x7fffffff;
                 
                if ( actual_row<md.m ){
                    int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                    int addr = actual_row*md.k;
                    if ( atomicORnot>>31 ){
                        atomicAdd( &md.mat_c_dev[ addr + c_col], res[c] );
                        //atomicAdd( &md.mat_c_dev[ addr + c_col*2 + 0 ], res[c][0]);
                        //atomicAdd( &md.mat_c_dev[ addr + c_col*2 + 1 ], res[c][1]);
                    }else{
                        md.mat_c_dev[ addr + c_col ] = res[ c ];
                        //float* mat_c = &md.mat_c_dev[ addr ];
                        //float2 vect2_c = {res[c][0], res[c][1]}; 
                        //reinterpret_cast<float2*>(mat_c)[ c_col ] = vect2_c; 
                    }
                }
            }
         
        }// end C colums

    } // end tile-segs loops
    
        timing_end();
}


template<int tm, int nnz_limit, int warps>
__global__
void flexspmm_cuda_w_pre_w_vec_v19(){ 
    // requires preprocess dense mat B

    const Mat_POD& md = mat_dev;
    
    timing_start(); 
    
    int gold_row_id[tm];
    __shared__ int smem[2][3*(nnz_limit)+3*tm];
    //int dbl = 0;
    int *rsm1 = reinterpret_cast<int*>(smem[0]);
    int *csm1 = reinterpret_cast<int*>(&smem[0][1*(nnz_limit+tm)]);
    float *vsm1 = reinterpret_cast<float*>(&smem[0][2*(nnz_limit+tm)]);
    
    int *rsm2 = reinterpret_cast<int*>(smem[1]);
    int *csm2 = reinterpret_cast<int*>(&smem[1][1*(nnz_limit+tm)]);
    float *vsm2 = reinterpret_cast<float*>(&smem[1][2*(nnz_limit+tm)]);
   
    int seg_cur_id = 0, seg_nxt_id = 0, nnz_cur_seg = 0;
    
    int segs_per_blk = (md.n_segs<gridDim.x) ? 3 : ( md.n_segs + gridDim.x - 1 ) / gridDim.x; 
    int bd = min((blockIdx.x+1)*segs_per_blk, md.n_segs);
    // preload the 1st tile-seg
    if ( blockIdx.x*segs_per_blk < bd ) {    
        //int2* seg_ids_ld = (int2*)(&md.segPtr_dev[ blockIdx.x ]);
        //int2 seg_ids = seg_ids_ld[0];
        seg_cur_id = md.segPtr_dev[ blockIdx.x*segs_per_blk ]; 
        seg_nxt_id = md.segPtr_dev[ blockIdx.x*segs_per_blk+1 ]; 

        for ( int i=seg_cur_id+threadIdx.x; i<seg_nxt_id; i += blockDim.x ){
            int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[i];
            rsm1[ i-seg_cur_id ] = rc.x;
            csm1[ i-seg_cur_id ] = rc.y;
            vsm1[ i-seg_cur_id ] = md.vals_dev[ i ];
            
        }
    }
    __syncthreads();
    
    for ( int seg_idx=blockIdx.x*segs_per_blk; seg_idx<bd; seg_idx += 1 ){ // over  tile segments
                   
        nnz_cur_seg = seg_nxt_id - seg_cur_id;
        
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = md.segVoMap_dev[seg_idx*tm+i];
        }
        
        // preload next tile-seg
        if ( seg_idx + 1 < bd ) {    
            //int2* seg_ids_ld = (int2*)(&md.segPtr_dev[ seg_idx+blockIdx.x ]);
            //int2 seg_ids = seg_ids_ld[0];
            seg_cur_id = md.segPtr_dev[ seg_idx+1 ]; 
            seg_nxt_id = md.segPtr_dev[ seg_idx+1+1 ]; 
            
            for ( int i=seg_cur_id+threadIdx.x; i<seg_nxt_id; i += blockDim.x ){
                int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[i];
                rsm2[ i-seg_cur_id ] = rc.x;
                csm2[ i-seg_cur_id ] = rc.y;
                vsm2[ i-seg_cur_id ] = md.vals_dev[ i ]; 
            }
        }

        for ( int c_col=threadIdx.x; c_col<md.k; c_col += blockDim.x ){ // over C columns
	        float res[tm]{};
            
            auto do_n = [&](int n)
             {
               for ( int z=0; z<n; z++ )
                 {
                   float val = vsm1[ z ];
                   int ridx = rsm1[ z ];
                   int cidx = csm1[ z ];
                   res[ridx] += md.shadow_b_dev[ cidx * md.k + c_col ] * val;
                   //float *shadow_b_addr = &md.shadow_b_dev[ cidx*md.k ];
                   //float2 b_vec = reinterpret_cast<float2*>(shadow_b_addr)[ c_col ];
                   //res[ridx][0] += val * b_vec.x;
                   //res[ridx][1] += val * b_vec.y;
                 }
             };
            do_n( nnz_cur_seg );
            
            // store C tiles back to global mem
            #pragma unroll
            for ( int c=0; c<tm; ++c ){
                int actual_row = gold_row_id[ c ] & 0x7fffffff;
                 
                if ( actual_row<md.m ){
                    int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                    int addr = actual_row*md.k;
                    if ( atomicORnot>>31 ){
                        atomicAdd( &md.mat_c_dev[ addr + c_col], res[c] );
                        //atomicAdd( &md.mat_c_dev[ addr + c_col*2 + 0 ], res[c][0]);
                        //atomicAdd( &md.mat_c_dev[ addr + c_col*2 + 1 ], res[c][1]);
                    }else{
                        md.mat_c_dev[ addr + c_col ] = res[ c ];
                        //float* mat_c = &md.mat_c_dev[ addr ];
                        //float2 vect2_c = {res[c][0], res[c][1]}; 
                        //reinterpret_cast<float2*>(mat_c)[ c_col ] = vect2_c; 
                    }
                }
            }
         
        }// end C colums

        // switch buffer
        int *r_temp = rsm1;
        int *c_temp = csm1;
        float *v_temp = vsm1;


        __syncthreads();
        rsm1 = rsm2;
        csm1 = csm2;
        vsm1 = vsm2;

        
        rsm2 = r_temp;
        csm2 = c_temp;
        vsm2 = v_temp;
        
        // failed, have't figured out why 
        //dbl ^= 1;
        //rsm2 = reinterpret_cast<int*>(&smem[dbl][0]);
        //csm2 = reinterpret_cast<int*>(&smem[dbl][1*(nnz_limit+12)*4]);
        //vsm2 = reinterpret_cast<float*>(&smem[dbl][2*(nnz_limit+12)*4]);
         
    } // end tile-segs loops
    
        timing_end();
}


template<int tm, int nnz_limit, int warps>
__global__
void flexspmm_cuda_wo_pre_w_vec_v20(){ 
    // requires preprocess dense mat B

    const Mat_POD& md = mat_dev;
    
    timing_start(); 
    
    int gold_row_id[tm];
    
    int seg_cur_id = 0, seg_nxt_id = 0, nnz_cur_seg = 0;
    
    for ( int seg_idx=blockIdx.x; seg_idx<md.n_segs; seg_idx += gridDim.x ){ // over  tile segments
                   
        
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = md.segVoMap_dev[seg_idx*tm+i];
        }
        
        seg_cur_id = md.segPtr_dev[ seg_idx ]; 
        seg_nxt_id = md.segPtr_dev[ seg_idx+1 ]; 
        nnz_cur_seg = seg_nxt_id - seg_cur_id;
        
        for ( int c_col=threadIdx.x; c_col<md.k; c_col += blockDim.x ){ // over C columns
	        float res[tm]{};
            
            auto do_n = [&](int n)
             {
               for ( int z=0; z<n; z++ )
                 {
                   int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[seg_cur_id+z];
                   float val = md.vals_dev[ seg_cur_id+z ];
                   int ridx = rc.x;
                   int cidx = rc.y;
                   res[ridx] += md.shadow_b_dev[ cidx * md.k + c_col ] * val;
                 }
             };
            do_n( nnz_cur_seg );
            
            // store C tiles back to global mem
            #pragma unroll
            for ( int c=0; c<tm; ++c ){
                int actual_row = gold_row_id[ c ] & 0x7fffffff;
                 
                if ( actual_row<md.m ){
                    int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                    int addr = actual_row*md.k;
                    if ( atomicORnot>>31 ){
                        atomicAdd( &md.mat_c_dev[ addr + c_col], res[c] );
                    }else{
                        md.mat_c_dev[ addr + c_col ] = res[ c ];
                    }
                }
            }
         
        }// end C colums

    } // end tile-segs loops
    
        timing_end();
}

template<int tm, int nnz_limit, int warps>
__global__
void flexspmm_cuda_wo_pre_w_vec_v21(){ 
    // requires preprocess dense mat B

    const Mat_POD& md = mat_dev;
    uint32_t sm_id = smid_get();    
    timing_start(); 
    
    int gold_row_id[tm];
    
    int nsi = sm_id;
    const int tail_seg_idx = md.grouped_tailSeg_dev[nsi];
    while ( true ) {

      int seg_idx_0 = threadIdx.x ? 0 : atomicAdd( &md.next_seg_dev[ nsi ], 1 );

      __shared__ int seg_idx;

      __syncthreads();
      if ( threadIdx.x == 0 ) seg_idx = seg_idx_0;
      __syncthreads();

      if ( nsi < md.sms && seg_idx >= tail_seg_idx ) { nsi = md.sms; continue; }
      if ( nsi == md.sms && seg_idx >= md.n_segs ) break;

        int seg_cur_id = md.segPtr_dev[ seg_idx ]; 
        int nnz_cur_seg = md.segPtr_dev[ seg_idx+1 ] - seg_cur_id;
        
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = md.segVoMap_dev[seg_idx*tm+i];
        }
        
        for ( int c_col=threadIdx.x; c_col<md.k; c_col += blockDim.x ){ // over C columns
	        float res[tm]{};
            
            auto do_n = [&](int n)
             {
               for ( int z=0; z<n; z++ )
                 {
                   int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[seg_cur_id+z];
                   float val = md.vals_dev[ seg_cur_id+z ];
                   int ridx = rc.x;
                   int cidx = rc.y;
                   res[ridx] += md.shadow_b_dev[ cidx * md.k + c_col ] * val;  
                 }
             };
            do_n( nnz_cur_seg );
            
            // store C tiles back to global mem
            #pragma unroll
            for ( int c=0; c<tm; ++c ){
                int actual_row = gold_row_id[ c ] & 0x7fffffff;
                 
                if ( actual_row<md.m ){
                    int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                    int addr = actual_row*md.k;
                    if ( atomicORnot>>31 ){
                        atomicAdd( &md.mat_c_dev[ addr + c_col], res[c] );
                    }else{
                        md.mat_c_dev[ addr + c_col ] = res[ c ];
                    }
                }
            }
         
        }// end C colums

    } // end tile-segs loops
    
        timing_end();
}


template<int tm, int nnz_limit, int warps>
__global__
void flexspmm_cuda_wo_pre_v22(){ 
    // requires preprocess dense mat B

    const Mat_POD& md = mat_dev;
    uint32_t sm_id = smid_get();    
    uint32_t lane_id = threadIdx.x%32;
    timing_start(); 
    
    int gold_row_id[tm];
   
    int nsi = sm_id; 
    const int tail_seg_idx = md.grouped_tailSeg_dev[ nsi ];     
    while ( true ){ // over tile segments in a bucket
                   
        int seg_idx_0 = threadIdx.x? 0 : atomicAdd(&md.next_seg_dev[ nsi ],1);
        __shared__ int seg_idx;
        
        __syncthreads();
        if ( threadIdx.x==0 ) seg_idx = seg_idx_0;
        __syncthreads();

        if ( nsi < md.sms && seg_idx >= tail_seg_idx ){ nsi = md.sms; continue; }
        if ( nsi == md.sms && seg_idx >= md.n_segs ) break;

        int seg_cur_id = md.segPtr_dev[ seg_idx ]; 
        int nnz_cur_seg = md.segPtr_dev[ seg_idx+1 ] - seg_cur_id;
        
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = md.segVoMap_dev[seg_idx*tm+i];
        }
        
        const int n_rounds = nnz_cur_seg / 32; 
        const uint nnz_remaining = nnz_cur_seg % 32;
        const int vidx_base = seg_cur_id + n_rounds * 32;
        
        for ( int c_col=threadIdx.x; c_col<md.k; c_col += blockDim.x ){ // over C columns
	        float res[tm]{};
                
            for ( int rnd = 0; rnd < n_rounds; ++rnd ){

                // load sparse nz from glb mem
                const int vidx = seg_cur_id + rnd*32 + lane_id;
                const float val = md.vals_dev[vidx];
                const int ridx = md.segNzRowIdx_dev[vidx];
                const int cidx = md.segNzColIdx_dev[vidx];
                
                // exchange nnz within a warp && perfom FMA
                for (int it=0; it<32; ++it){
                  float v = __shfl_sync(FULL_MASK, val, it);
                  int v_r = __shfl_sync(FULL_MASK, ridx, it);
                  int v_c = __shfl_sync(FULL_MASK, cidx, it);
                  res[v_r] +=
                    v * md.shadow_b_dev[ v_c*md.k + c_col ];
                }    
            }


            auto do_n = [&](int n)
             {
               for ( int z=0; z<n; z++ )
                 {
                   float val = md.vals_dev[ vidx_base+z ];
                   int ridx = md.segNzRowIdx_dev[ vidx_base+z ];
                   int cidx = md.segNzColIdx_dev[ vidx_base+z ];
                   res[ridx] += md.shadow_b_dev[ cidx * md.k + c_col ] * val;  
                 }
             };
            
            if ( nnz_remaining ){
              #define C5(n) C4(n) C4(n+16)
              #define C4(n) C3(n) C3(n+8)
              #define C3(n) C2(n) C2(n+4)
              #define C2(n) C1(n) C1(n+2)
              #define C1(n) C0(n) C0(n+1)
              #define C0(n) case n: do_n(n); break;
              switch ( nnz_remaining ) {
                C2(1);  // This generates do_n(1) to do_n(4)
              default: do_n( nnz_remaining ); break;
              }
              #undef C5
              #undef C4
              #undef C3
              #undef C2
              #undef C1
              #undef C0
            } 
            
            // store C tiles back to global mem
            #pragma unroll
            for ( int c=0; c<tm; ++c ){
                int actual_row = gold_row_id[ c ] & 0x7fffffff;
                 
                if ( actual_row<md.m ){
                    int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                    int addr = actual_row*md.k;
                    if ( atomicORnot>>31 ){
                        atomicAdd( &md.mat_c_dev[ addr + c_col], res[c] );
                    }else{
                        md.mat_c_dev[ addr + c_col ] = res[ c ];
                    }
                }
            }
         
        }// end C colums
        
    } // end tile-segs loops
    
        timing_end();
}

template<int tm, int nnz_limit, int warps>
__global__
void flexspmm_cuda_w_pre_v23(){ 
    // requires preprocess dense mat B
    cg::thread_block tb = cg::this_thread_block();
    int use_memcpy_asy = 1;

    const Mat_POD& md = mat_dev;
    uint32_t sm_id = smid_get();    
    int nsi = sm_id;
    timing_start(); 
    
    int gold_row_id[tm];
    
    __align__(4) __shared__ int smem[2][3*(nnz_limit)+3*tm+1];
    int* seg_idx = &smem[0][3*(nnz_limit+tm)];

    int *rsm1 = reinterpret_cast<int*>(smem[0]);
    int *csm1 = reinterpret_cast<int*>(&smem[0][1*(nnz_limit+tm)]);
    float *vsm1 = reinterpret_cast<float*>(&smem[0][2*(nnz_limit+tm)]);
    
    int *rsm2 = reinterpret_cast<int*>(smem[1]);
    int *csm2 = reinterpret_cast<int*>(&smem[1][1*(nnz_limit+tm)]);
    float *vsm2 = reinterpret_cast<float*>(&smem[1][2*(nnz_limit+tm)]);

    //if (threadIdx.x==0){
    if (tb.thread_rank()==0){
        seg_idx[0] = atomicAdd(&md.next_seg_dev[ nsi ],1);
    }
    int tail_seg_idx = md.grouped_tailSeg_dev[ nsi ];     
    //tb.sync();
    cg::wait(tb);
    //__syncthreads();

    if ( nsi<md.sms && seg_idx[0] >= tail_seg_idx ){ 
        nsi = md.sms; 
        if ( tb.thread_rank()==0 ){
            seg_idx[0] = atomicAdd(&md.next_seg_dev[ nsi ],1);
        }
        tail_seg_idx = md.grouped_tailSeg_dev[ nsi ];     
    }
    cg::wait(tb);
    if ( nsi == md.sms && seg_idx[0] >= md.n_segs ) { timing_end(); return; }
    
    int seg_cur_id = md.segPtr_dev[ seg_idx[0] ]; 
    int nnz_cur_seg = md.segPtr_dev[ seg_idx[0]+1 ] - seg_cur_id;
    int nnz_nxt_seg = 0;
    int seg_nxt_id = -1;
    
    if ( use_memcpy_asy ){
        cg::memcpy_async(tb, rsm1, md.segNzRowIdx_dev + seg_cur_id, nnz_cur_seg*sizeof(int));
        cg::memcpy_async(tb, csm1, md.segNzColIdx_dev + seg_cur_id, nnz_cur_seg*sizeof(int));
        cg::memcpy_async(tb, vsm1, md.vals_dev + seg_cur_id, nnz_cur_seg*sizeof(float));
    }else{
        for ( int i=seg_cur_id+threadIdx.x; i<seg_cur_id+nnz_cur_seg; i += blockDim.x ){
            rsm1[ i-seg_cur_id ] = md.segNzRowIdx_dev[ i ];
            csm1[ i-seg_cur_id ] = md.segNzColIdx_dev[ i ];
            vsm1[ i-seg_cur_id ] = md.vals_dev[ i ];
        }
    }
    if ( !use_memcpy_asy ) cg::wait(tb);
    
    int cur_seg_id = -1;        
    while ( true  ){ // over tile segments in a bucket
        
        cur_seg_id = seg_idx[0];        
/***********  fetch next seg   ***********/
        int seg_idx_0 = tb.thread_rank()? 0 : atomicAdd(&md.next_seg_dev[ nsi ],1);
        
        cg::wait(tb);
        if ( tb.thread_rank()==0 ) seg_idx[0] = seg_idx_0;
        cg::wait(tb);

        tail_seg_idx = md.grouped_tailSeg_dev[ nsi ];     
        if ( nsi < md.sms && seg_idx[0] >= tail_seg_idx ){ 
            nsi = md.sms;
            if ( tb.thread_rank()==0 ){
                seg_idx[0] = atomicAdd(&md.next_seg_dev[ nsi ],1);
            }
            tail_seg_idx = md.grouped_tailSeg_dev[ nsi ];     
            cg::wait(tb);
        }
        if ( nsi == md.sms && seg_idx[0] >= md.n_segs ) break;
         
        seg_nxt_id = md.segPtr_dev[ seg_idx[0] ]; 
        nnz_nxt_seg = md.segPtr_dev[ seg_idx[0]+1 ] - seg_nxt_id; 
        if ( use_memcpy_asy ){
            cg::memcpy_async(tb, rsm2, md.segNzRowIdx_dev + seg_nxt_id, nnz_nxt_seg*sizeof(int));
            cg::memcpy_async(tb, csm2, md.segNzColIdx_dev + seg_nxt_id, nnz_nxt_seg*sizeof(int));
            cg::memcpy_async(tb, vsm2, md.vals_dev + seg_nxt_id, nnz_nxt_seg*sizeof(float));
        }else{
            for ( int i=seg_nxt_id+threadIdx.x; i<seg_nxt_id+nnz_nxt_seg; i += blockDim.x ){
                rsm2[ i-seg_nxt_id ] = md.segNzRowIdx_dev[ i ];
                csm2[ i-seg_nxt_id ] = md.segNzColIdx_dev[ i ];
                vsm2[ i-seg_nxt_id ] = md.vals_dev[ i ];
            }
        }
/****************************************/
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = md.segVoMap_dev[ cur_seg_id*tm+i ];
        }
        
        if ( use_memcpy_asy ){
            cg::wait_prior<3>(tb); 
        } 
        //for ( int c_col=threadIdx.x; c_col<md.k; c_col += blockDim.x ){ // over C columns
        for ( int c_col=tb.thread_rank(); c_col<md.k; c_col += tb.size() ){ // over C columns
	        float res[tm]{};
                
            auto do_n = [&](int n)
             {
               for ( int z=0; z<n; z++ )
                 {
                   float val = vsm1[ z ];
                   int ridx = rsm1[ z ];
                   int cidx = csm1[ z ];

                   res[ridx] += md.shadow_b_dev[ cidx * md.k + c_col ] * val;  

                 }
             };
            do_n(nnz_cur_seg); 
            
            // store C tiles back to global mem
            //#pragma unroll
            for ( int c=0; c<tm; ++c ){
                int actual_row = gold_row_id[ c ] & 0x7fffffff;
                 
                if ( actual_row<md.m ){
                    int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                    int addr = actual_row*md.k;
                    if ( atomicORnot>>31 ){
                        atomicAdd( &md.mat_c_dev[ addr + c_col], res[c] );
                    }else{
                        md.mat_c_dev[ addr + c_col ] = res[ c ];
                    }
                }
                
            }
             
        }// end C colums
       
        // switch buffer
        int *r_temp = rsm1;
        int *c_temp = csm1;
        float *v_temp = vsm1;

        cg::wait(tb);
        //tb.sync();
        //__syncthreads();
        rsm1 = rsm2;
        csm1 = csm2;
        vsm1 = vsm2;


        rsm2 = r_temp;
        csm2 = c_temp;
        vsm2 = v_temp;

        nnz_cur_seg = nnz_nxt_seg;
    } // end tile-segs loops
    
    // handle the last tile-seg 
    #pragma unroll
    for (int i=0; i<tm; ++i){
        gold_row_id[i] = md.segVoMap_dev[ cur_seg_id*tm+i ];
    }
    for ( int c_col=tb.thread_rank(); c_col<md.k; c_col += tb.size() ){ // over C columns
        float res[tm]{};
            
        auto do_n = [&](int n)
         {
           for ( int z=0; z<n; z++ )
             {
               float val = vsm1[ z ];
               int ridx = rsm1[ z ];
               int cidx = csm1[ z ];
               res[ridx] += md.shadow_b_dev[ cidx * md.k + c_col ] * val;  

             }
         };
        do_n(nnz_cur_seg); 
        
        // store C tiles back to global mem
        //#pragma unroll
        for ( int c=0; c<tm; ++c ){
            int actual_row = gold_row_id[ c ] & 0x7fffffff;
             
            if ( actual_row<md.m ){
                int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                int addr = actual_row*md.k;
                if ( atomicORnot>>31 ){
                    atomicAdd( &md.mat_c_dev[ addr + c_col], res[c] );
                }else{
                    md.mat_c_dev[ addr + c_col ] = res[ c ];
                }
            }
            
        }
         
    }// end C colums
        timing_end();
}

template<int tm, int CF, int warps>
__global__
void flexspmm_cuda_wo_pre_w_vec_v24(){ 
    // requires preprocess dense mat B

    const Mat_POD& md = mat_dev;
    uint32_t sm_id = smid_get();    
    int warp_id = threadIdx.y;
    timing_start(); 
    
    extern __shared__ int sh[];
    // |--------|-------|-------|-------|-------|-------|--|--|
    // |  32,r  |  32,r |  32,c |  32,c |  32,v |  32,v |   
    // |  wp0   |  wp1  |  wp0  |  wp1  |  wp0  |  wp1  |sidx0|sidx1|
      
    int     *rIdx = sh;
    int     *cIdx = &sh[ (blockDim.y<<5) ]; // imply blockDim.x==32
    float    *val = (float *)&sh[ (blockDim.y<<6) ];
    int  *seg_idx = &sh[ (blockDim.y<<5)*3 + warp_id ];
    
    // since blockDim.x==32, shmem_offset is the offset of a warp
    int shmem_offset = ( threadIdx.y<<5 ); 

    int gold_row_id[tm];
    int nsi = sm_id;
    const int tail_seg_idx = md.grouped_tailSeg_dev[nsi];     
    
    while ( true ) { // over tile segments in a bucket
                
        int seg_idx_0 = threadIdx.x ? 0 : atomicAdd(&md.next_seg_dev[nsi],1);
        
        __syncwarp();
        if ( threadIdx.x == 0) seg_idx[0] = seg_idx_0;
        __syncwarp();
        
        if ( nsi < md.sms && seg_idx[0] >= tail_seg_idx ){ nsi = md.sms; continue; }
        if ( nsi == md.sms && seg_idx[0] >= md.n_segs ) break;


        int seg_cur_id = md.segPtr_dev[ seg_idx[0] ]; 
        int seg_nxt_id = md.segPtr_dev[ seg_idx[0]+1 ];
        
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = md.segVoMap_dev[ seg_idx[0]*tm+i ];
        }
     
        // note: blockDim.x = 32
        for ( int cid = 0; cid<md.k; cid+=32 ){
            float res[tm]{};
            for ( int nz_i=seg_cur_id; nz_i<seg_nxt_id; nz_i += 32 ){    
                // load non-zeros from glb to sh
                if ( nz_i+threadIdx.x < seg_nxt_id ){
                    int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[ nz_i+threadIdx.x ];
                    rIdx[ shmem_offset+threadIdx.x ] = rc.x;
                    cIdx[ shmem_offset+threadIdx.x ] = rc.y * md.k;
                    val[ shmem_offset+threadIdx.x ] = md.vals_dev[ nz_i+threadIdx.x ];
                }
                __syncwarp();

                if ( cid+threadIdx.x<md.k ){
                    // use loaded non-zeros
                    for ( int z=0; z<32 && (nz_i+z)<seg_nxt_id; z++ ){  
                        int ridx = rIdx[ shmem_offset + z ];
                        float nz_val = val[ shmem_offset + z ];
                        int b_col = cIdx[ shmem_offset + z ] + cid + threadIdx.x;

                        res[ridx] += md.shadow_b_dev[ b_col ] * nz_val;  
                    }
                }
                __syncwarp();
            }
            
            if ( cid+threadIdx.x<md.k ){
                // store C tiles back to global mem
                #pragma unroll
                for ( int c=0; c<tm; ++c ){
                    int actual_row = gold_row_id[ c ] & 0x7fffffff;
                     
                    if ( actual_row<md.m ){
                        int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                        int addr = actual_row*md.k;
                        if ( atomicORnot>>31 ){
                            atomicAdd( &md.mat_c_dev[ addr + cid + threadIdx.x ], res[c] );
                        }else{
                            md.mat_c_dev[ addr + cid + threadIdx.x ] = res[c];
                        }
                    }
                }
            }
        } // end C column 
    } // end tile-segs loops
    
        timing_end();
}

template<int tm, int CF>
__global__
void flexspmm_cuda_wo_pre_w_vec_v25_0(){ 
    // requires preprocess dense mat B

    const Mat_POD& md = mat_dev;
    uint32_t sm_id = smid_get();    
    int warp_id = threadIdx.y;
    
    extern __shared__ int sh[];
    // |--------|-------|-------|-------|-------|-------|--|--|
    // |  32,r  |  32,r |  32,c |  32,c |  32,v |  32,v |   
    // |  wp0   |  wp1  |  wp0  |  wp1  |  wp0  |  wp1  |sidx0|sidx1|
      
    int     *rIdx = sh;
    int     *cIdx = &sh[ (blockDim.y<<5) ]; // imply blockDim.x==32
    float    *val = (float *)&sh[ (blockDim.y<<6) ];
    int  *seg_idx = &sh[ (blockDim.y<<5)*3 + warp_id ];
    
    // since blockDim.x==32, shmem_offset is the offset of a warp
    int shmem_offset = ( threadIdx.y<<5 ); 

    int gold_row_id[tm];
    int nsi = sm_id;
    const int tail_seg_idx = md.grouped_tailSeg_dev[nsi];     
    
    while ( true ) { // over tile segments in a bucket
                
        int seg_idx_0 = threadIdx.x ? 0 : atomicAdd(&md.next_seg_dev[nsi],1);
        
        __syncwarp();
        if ( threadIdx.x == 0 ) seg_idx[0] = seg_idx_0;
        __syncwarp();
        
        if ( nsi < md.sms && seg_idx[0] >= tail_seg_idx ){ break; }
        //if ( nsi == md.sms && seg_idx[0] >= md.n_segs ) { break; }


        int seg_cur_id = md.segPtr_dev[ seg_idx[0] ]; 
        int seg_nxt_id = md.segPtr_dev[ seg_idx[0]+1 ];
        
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = md.segVoMap_dev[ seg_idx[0]*tm+i ];
        }
     
        // note: blockDim.x = 32
        for ( int cid = 0; cid<md.k; cid+=32 ){
            float res[tm]{};
            for ( int nz_i=seg_cur_id; nz_i<seg_nxt_id; nz_i += 32 ){    
                // load non-zeros from glb to sh
                if ( nz_i+threadIdx.x < seg_nxt_id ){
                    int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[ nz_i+threadIdx.x ];
                    rIdx[ shmem_offset+threadIdx.x ] = rc.x;
                    cIdx[ shmem_offset+threadIdx.x ] = rc.y * md.k;
                    val[ shmem_offset+threadIdx.x ] = md.vals_dev[ nz_i+threadIdx.x ];
                }
                __syncwarp();

                if ( cid+threadIdx.x<md.k ){
                    // use loaded non-zeros
                    for ( int z=0; z<32 && (nz_i+z)<seg_nxt_id; z++ ){  
                        int ridx = rIdx[ shmem_offset + z ];
                        float nz_val = val[ shmem_offset + z ];
                        int b_col = cIdx[ shmem_offset + z ] + cid + threadIdx.x;

                        res[ridx] += md.shadow_b_dev[ b_col ] * nz_val;  

                    }
                }
                //__syncwarp();
            }
            
            if ( cid+threadIdx.x<md.k ){
                // store C tiles back to global mem
                #pragma unroll
                for ( int c=0; c<tm; ++c ){
                    int actual_row = gold_row_id[ c ] & 0x7fffffff;
                     
                    if ( actual_row<md.m ){
                        int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                        int addr = actual_row*md.k;
                        if ( atomicORnot>>31 ){
                            atomicAdd( &md.mat_c_dev[ addr + cid + threadIdx.x ], res[c] );
                        }else{
                            md.mat_c_dev[ addr + cid + threadIdx.x ] = res[c];
                        }
                    }
                }
            }
        } // end C column 
    } // end tile-segs loops

}


template<int tm>
__global__
void flexspmm_cuda_wo_pre_w_vec_v25_1(){ 
    // requires preprocess dense mat B

    const Mat_POD& md = mat_dev;
    int nsi = smid_get(); 
    __shared__ int sh[96+tm];
    int *rIdx = sh;
    int *cIdx = &sh[32];
    float *vals = (float*)&sh[64];
    int *gold_row_id = &sh[96];
    
    for ( int seg_idx=md.next_seg_dev[md.sms]+blockIdx.x; seg_idx<md.n_segs; seg_idx += gridDim.x ) { // over tile segments in a bucket
                 
        int seg_cur_id = md.segPtr_dev[ seg_idx ]; 
        int seg_nxt_id = md.segPtr_dev[ seg_idx+1 ];
        
        if ( threadIdx.x<tm ){
            gold_row_id[threadIdx.x] = md.segVoMap_dev[ seg_idx*tm+threadIdx.x ];
        }
        __syncthreads();
     
        for ( int cid = 0; cid<md.k; cid += blockDim.x ){
            float res[tm]{};
            for ( int nz_i=seg_cur_id; nz_i<seg_nxt_id; nz_i+=32 ){    
                // load non-zeros from glb to sh
                if ( threadIdx.x<32 && (nz_i + threadIdx.x) < seg_nxt_id ){
                    int2 rc = reinterpret_cast<int2*>(md.segNzRCIdx_dev)[ nz_i+threadIdx.x ];
                    rIdx[ threadIdx.x ] = rc.x;
                    cIdx[ threadIdx.x ] = rc.y * md.k;
                    vals[ threadIdx.x ] = md.vals_dev[ nz_i+threadIdx.x ];
                }
                __syncthreads();

                if ( cid+threadIdx.x<md.k ){
                    // use loaded non-zeros
                    for ( int z=0; z<32 && (nz_i+z)<seg_nxt_id; z++ ){  
                        int ridx = rIdx[ z ];
                        float nz_val = vals[ z ];
                        int b_col = cIdx[ z ] + cid + threadIdx.x;

                        res[ridx] += md.shadow_b_dev[ b_col ] * nz_val;  

                    }
                }
                //__syncthreads();
            }
            
            if ( cid+threadIdx.x<md.k ){
                // store C tiles back to global mem
                #pragma unroll
                for ( int c=0; c<tm; ++c ){
                    int actual_row = gold_row_id[ c ] & 0x7fffffff;
                     
                    if ( actual_row<md.m ){
                        int atomicORnot = gold_row_id[c] & (1<<31); // get MSB
                        int addr = actual_row*md.k;
                        if ( atomicORnot>>31 ){
                            atomicAdd( &md.mat_c_dev[ addr + cid + threadIdx.x ], res[c] );
                        }else{
                            md.mat_c_dev[ addr + cid + threadIdx.x ] = res[c];
                        }
                    }
                }
            }
        } // end C column 
    } // end tile-segs loops
    
}

GPU_Info
print_gpu_and_kernel_info()
{
   GPU_Info info;

   gpu_info_print();

   // Choose GPU 0 because it's usually the better choice.
   //
   int dev = gpu_choose_index();
   CE(cudaSetDevice(dev));
   printf("Using GPU %d\n",dev);
   info.get_gpu_info(dev);

   return info;
}   
struct tileConf {
    const int tm, tn;
};
constexpr tileConf tileConfs[] = 
    { {4,4},{8,4},{16,4},{32,4},{64,4},{128,4},{256,4}
     ,{4,8},{8,8},{16,8},{32,8},{64,8},{128,8},{256,8}
     ,{4,16},{8,16},{16,16},{32,16},{64,16},{128,16},{256,16}
     ,{4,32},{8,32},{16,32},{32,32},{64,32},{128,32},{256,32}
    };

void
resCheck(float* h_gold, float* h_res, const Mat& mat, Perfs& perfRes)
{
    const int tm = mat.tm;
    const int tn = mat.tn;
    const int m = mat.m;
    const int n = mat.k;
    // verify results
    int count = 0, err_show_remaining = 20;
    int nz = 0;
    double max_err = 0;
    int me_nnz = 0;
    int last_err_row = -1;

    for (int r=0; r<m; ++r){

        const auto& rp = mat.dl.dl_original->rowPtr;
        const int row_nnz = rp[r+1] - rp[r];
        const double tol = numeric_limits<float>::epsilon() * row_nnz * 4;
        //
        // The tolerance above is for partial products that are about
        // 1. The tolerance will miss errors when partial products are
        // < 1 and it will result in false positives when partial
        // products are larger than one.

        for (int c=0; c<n; ++c){
          const int idx = r*n+c;
          if ( h_gold[idx] == 0 ) nz++;
          const double err =
            fabs(h_gold[idx]) < 1 ? fabs( double(h_gold[idx]) - h_res[idx] )
            : fabs( 1.0 - double(h_res[idx])/h_gold[idx] );
          if ( set_max(max_err,err) ) me_nnz = row_nnz;
          if ( err > tol ) {
            count++;
            if ( r != last_err_row && err_show_remaining-- > 0 )
              {
                last_err_row = r;
                printf(" ref[%d][%d]:  %f!=%f (correct)"
                       "  %d nnzs, dif %g, tol %g\n",
                       r, c, h_res[idx], h_gold[idx],
                       row_nnz, err, tol);
              }
          }
        }
    }

    perfRes.flex_spmm_errors.push_back(count);
    if ( count )
      {
        cout <<"Kernel ("<< to_string(tm) << "X" << to_string(tn)
             << ") errs: " << count;
        printf(" Max err %g at nnz=%d.\n", max_err, me_nnz);
        assert( !count );
      }

    // If correct result has too many zeros it will be hard to catch errors.
    assert( nz < n/2 );

    memset(h_res, 0, n*m*sizeof(float));
}
void resCheck2(float* h_gold, float* h_res, int len){
    int errors = 0;
    for (int i=0; i<len; ++i){
        if (fabs(h_gold[i]-h_res[i])>0.01){
            errors++;
        }
        if (errors>0 && errors<10){ 
            printf(" ref[%d][%d]:  %f!=%f (correct)\n",
                   i/32, i%32, h_res[i], h_gold[i]);
        }
    }    
    if (errors){
        //printf("ge-spmm errors: %d\n",errors);
        printf("stream-spmm errors: %d\n",errors);
    }else{
        printf("Congrats\n");
    }

}
void run_split_k(DataLoader& input_vo){

    Perfs perfRes;
    input_vo.c_cuSpmm_run(perfRes);
    //cudaMemset(input_vo.gpuC, 0, input_vo.gpuC_bytes);

    printf("cuSpMM setup/µs: %.2f , prosessing/µs: %.2f, total/µs: %.2f\n",
                   perfRes.cuSpmmSetup,perfRes.cuSpmmProcessing,perfRes.cuSpmm_time); 

    float elap_t = 0.0;
    //float flex_k0 = 0.0;
    //float flex_k1 = 0.0;
    cudaEvent_t k_start, k_stop;
    //cudaEvent_t k0_start, k0_stop;
    //cudaEvent_t k1_start, k1_stop;
	cudaEventCreate(&k_start);
	cudaEventCreate(&k_stop); 
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

   
    Mat mat(input_vo, 4, 4);
    mat.csr2tile();
    mat.transfer2();
    mat.launch_prep();
    CE( cudaMemcpy(mat.next_seg_dev, mat.next_seg.data(), mat.next_seg.size()*sizeof(int), cudaMemcpyHostToDevice) );
    dim3 grid(48*mat.sms, 1, 1); 
    int wps = 4;
    dim3 block(32, wps, 1); 
    cudaEventRecord(k_start,0); 
    
    flexspmm_cuda_wo_pre_w_vec_v25_0<4, 4><<<grid, block, wps*32*3*4+wps*4, stream0>>>();
    flexspmm_cuda_wo_pre_w_vec_v25_1<4><<<mat.sms*8, wps*32, (96+4)*sizeof(int), stream1>>>(); 

    cudaEventRecord(k_stop,0);
    cudaEventSynchronize(k_start);
    cudaEventSynchronize(k_stop);
    cudaEventElapsedTime(&elap_t, k_start, k_stop);
    cudaDeviceSynchronize();
    
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    printf("flex time: %f\n", elap_t*1e3);
    
    float* const h_res_c = (float*) malloc( input_vo.gpuC_bytes );
    cudaMemcpy( h_res_c, mat.mat_c_dev, input_vo.gpuC_bytes, cudaMemcpyDeviceToHost );
    resCheck2( input_vo.h_ref_c.data(), h_res_c, input_vo.m*input_vo.dim);
        
    free( h_res_c );
    mat.freeMatGPU2();
}
void run_ge_spmm(DataLoader& input_vo){
     
    Perfs perfRes;
    input_vo.c_cuSpmm_run(perfRes);
    Mat mat(input_vo, 0, 0); // the last two args are useless
    mat.launch_prep();

    NPerf_init();
    GPU_Info info = print_gpu_and_kernel_info();
    const cudaDeviceProp cuda_prop = info.cuda_prop;
    // Get number of SMs (aka MPs).
    //
    const int num_sm = cuda_prop.multiProcessorCount;

    // Compute number of FP32 per chip.
    //
    const int fp32_per_chip = info.get_fp32_per_sm() * num_sm;
    
    NPerf_metric_collect("sm__cycles_elapsed.max");                                                                            
    NPerf_metric_collect("sm__inst_executed.sum");
    NPerf_metric_collect("l1tex__m_xbar2l1tex_read_bytes.sum");
    NPerf_metric_collect("l1tex__m_l1tex2xbar_write_bytes.sum");
    NPerf_metric_collect("sm__sass_inst_executed_op_ld.sum");
    NPerf_metric_collect("sm__sass_inst_executed_op_global_ld.sum");
    NPerf_metric_collect("sm__sass_inst_executed_op_st.sum");
    NPerf_metric_collect("sm__sass_inst_executed_op_global_st.sum");
    NPerf_metric_collect
        ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed");
    NPerf_metric_collect
        ("l1tex__m_l1tex2xbar_throughput.avg.pct_of_peak_sustained_elapsed");
    NPerf_metric_collect("dram__bytes.sum");
    
    struct App_Kernel_Info {
         App_Kernel_Info  
         (Kernel_Info& k, 
          const char *name_b): k_ptr(k.func_ptr),name_base{name_b}{}
        GPU_Info_Func k_ptr;
        const char *name_base;
    };  
    vector<App_Kernel_Info> kernels;
    
    #define PUSH_KERNEL(kb,k) \
    {  kernels.emplace_back(info.GET_INFO((k)),#kb); }

#define GRE64
#ifdef LESS32
    PUSH_KERNEL(spmm_test0,spmm_test0);
#endif
#ifdef LESS64
    PUSH_KERNEL(spmm_test1,spmm_test1);
#endif
#ifdef GRE64
    PUSH_KERNEL(spmm_test2,spmm_test2);
#endif
    
    vector<Timing_Item> timing_items;
    Timing timing_dh{nullptr};
    size_t timing_items_bytes = 0;
    int grid_n_wps;
    
    pTable table(stdout);
    Kernel_Info* const ki = &info.get_info(kernels[0].k_ptr);
    typedef void (*KPtr)();
    pTable_Row row(table);
    
    //float* gespmm_c;
    //cudaMalloc(&gespmm_c, input_vo.gpuC_bytes);
    //cudaMemset(gespmm_c, 0, input_vo.gpuC_bytes);
    if (input_vo.dim<32){
        const int row_per_block = 128/input_vo.dim;
        const int n_block = (input_vo.m+row_per_block-1)/row_per_block;
        {   
            const int max_wps = n_block * (input_vo.dim * row_per_block) / 32; 
            grid_n_wps = max_wps;
            timing_items.resize( max_wps );
            timing_items_bytes = timing_items.size() * sizeof(timing_items[0]);
            CE( cudaFree( timing_dh.timing_items ) );
            CE( cudaMalloc( &timing_dh.timing_items, timing_items_bytes ) );
            CE( cudaMemset( timing_dh.timing_items, 0, timing_items_bytes ) );
            CE( cudaDeviceSynchronize() );
            CE( cudaMemcpyToSymbol
                ( timing_dev, &timing_dh, sizeof(timing_dh),
                  0, cudaMemcpyHostToDevice ) );
        }
        for ( NPerf_data_reset(); NPerf_need_run_get(); ){
           KPtr(ki->func_ptr)<<<dim3(n_block,1,1),dim3(input_vo.dim, row_per_block, 1)>>>();
        }

    }else if (input_vo.dim<64){
        const int tile_k = (input_vo.dim+31)/32;
        const int n_block = (input_vo.m+4-1)/4;

        {   
            const int max_wps = n_block * tile_k * 4; 
            grid_n_wps = max_wps;
            timing_items.resize( max_wps );
            timing_items_bytes = timing_items.size() * sizeof(timing_items[0]);
            CE( cudaFree( timing_dh.timing_items ) );
            CE( cudaMalloc( &timing_dh.timing_items, timing_items_bytes ) );
            CE( cudaMemset( timing_dh.timing_items, 0, timing_items_bytes ) );
            CE( cudaDeviceSynchronize() );
            CE( cudaMemcpyToSymbol
                ( timing_dev, &timing_dh, sizeof(timing_dh),
                  0, cudaMemcpyHostToDevice ) );
        }
        for ( NPerf_data_reset(); NPerf_need_run_get(); ){
           KPtr(ki->func_ptr)<<<dim3(n_block,tile_k,1),dim3(32, 4, 1),32*4*(sizeof(int)+sizeof(float))>>>();
        }

    }else{
        const int tile_k = (input_vo.dim+63)/64;
        const int n_block = (input_vo.m+8-1)/8;

        {   
            const int max_wps = n_block * tile_k * 8; 
            grid_n_wps = max_wps;
            timing_items.resize( max_wps );
            timing_items_bytes = timing_items.size() * sizeof(timing_items[0]);
            CE( cudaFree( timing_dh.timing_items ) );
            CE( cudaMalloc( &timing_dh.timing_items, timing_items_bytes ) );
            CE( cudaMemset( timing_dh.timing_items, 0, timing_items_bytes ) );
            CE( cudaDeviceSynchronize() );
            CE( cudaMemcpyToSymbol
                ( timing_dev, &timing_dh, sizeof(timing_dh),
                  0, cudaMemcpyHostToDevice ) );
        }
        for ( NPerf_data_reset(); NPerf_need_run_get(); ){
           KPtr(ki->func_ptr)<<<dim3(n_block,tile_k,1),dim3(32, 8, 1),32*8*(sizeof(int)+sizeof(float))>>>();
        }
    }

    // Copy per-warp timing data back to host.
    //
    CE( cudaMemcpy( timing_items.data(), timing_dh.timing_items,
                    timing_items_bytes, cudaMemcpyDeviceToHost ) );
                // Compute per-sm minimum-start and maximum-finish (end) times.
    //
    map<int32_t,Timing_Item> sm_start_end;
    int n_migs = 0; // Number of migrations.
    for ( auto& ti: views::take(timing_items,grid_n_wps+1) )
      if ( ti.smid_start != ti.smid_end )
        {
          n_migs++;
        }
      else
        {
          auto& tis = sm_start_end[ti.smid_start];
          if ( tis.time_start == tis.time_end ) tis = ti;
          set_min( tis.time_start, ti.time_start );
          set_max( tis.time_end, ti.time_end );
        }

    if ( n_migs ) printf("-- Number of migrations: %d\n",n_migs);
    //
    // Note: The per-sm data collection won't work if a block
    // migrates from one sm to another.

    // Compute average sm execution time and maximum sm execution time.
    //
    int64_t et_sum = 0;
    vector<int64_t> et;
    for ( auto& [smid,tis]: sm_start_end )
      {
        const int64_t elapsed = tis.time_end - tis.time_start;
        et_sum += elapsed;
        et.push_back( elapsed );
      }

    ranges::sort( et, ranges::greater() );

    const double clock_period_us = 1e6 / info.clock_freq_hz;
    const double et_clock_max_us = et[0] * clock_period_us;
    const double et_clock_avg_us = et_sum * clock_period_us / num_sm;
    const double imbalance_penalty =
      et_clock_avg_us ? et_clock_max_us / et_clock_avg_us - 1 : 0.0;

    const double et_seconds = NPerf_kernel_et_get();
    table.entry( "t/µs", "%7.1f", et_seconds * 1e6 );

    table.entry( "Imb", "%3.0f", 100 * imbalance_penalty );

    float* const h_res_c = (float*) malloc( input_vo.gpuC_bytes );
    cudaMemcpy( h_res_c, mat.mat_c_dev, input_vo.gpuC_bytes, cudaMemcpyDeviceToHost );
    resCheck2( input_vo.h_ref_c.data(), h_res_c, input_vo.m*input_vo.dim);
        
    CE( cudaFree( timing_dh.timing_items ) );
    free( h_res_c );
    mat.freeMatGPU2();
}
void run(DataLoader& input_vo){

    if ( false ){
        //run_ge_spmm(input_vo);
        run_split_k(input_vo);
        return ;
    }

    Perfs perfRes;
    input_vo.c_cuSpmm_run(perfRes);
    
    // Prepare a DFS-ordered matrix.
    DataLoaderDFS input_dfs(input_vo);
    DataLoaderRabbit input_rabbit(input_vo);
    //DataLoaderDeg input_deg(input_vo);
    //DataLoaderRcm input_rcm(input_vo);
    DataLoaderGorder input_gorder(input_vo);

    const bool opt_vary_grid_size = true;
    constexpr bool show_insn_shared = false;
    constexpr bool show_insn_local = false;

    //input.print_data();
    NPerf_init();
    GPU_Info info = print_gpu_and_kernel_info();
    const cudaDeviceProp cuda_prop = info.cuda_prop;
    // Get number of SMs (aka MPs).
    //
    const int num_sm = cuda_prop.multiProcessorCount;

    // Compute number of FP32 per chip.
    //
    const int fp32_per_chip = info.get_fp32_per_sm() * num_sm;
    
    NPerf_metric_collect("sm__cycles_elapsed.max");                                                                            
    NPerf_metric_collect("sm__inst_executed.sum");
    NPerf_metric_collect("l1tex__m_xbar2l1tex_read_bytes.sum");
    NPerf_metric_collect("l1tex__m_l1tex2xbar_write_bytes.sum");
    NPerf_metric_collect("sm__sass_inst_executed_op_ld.sum");
    NPerf_metric_collect("sm__sass_inst_executed_op_global_ld.sum");
    NPerf_metric_collect("sm__sass_inst_executed_op_st.sum");
    if ( show_insn_shared )
      {
        NPerf_metric_collect("sm__sass_inst_executed_op_shared_ld.sum");
        NPerf_metric_collect("sm__sass_inst_executed_op_shared_st.sum");
      }
    if ( show_insn_local )
      {
        NPerf_metric_collect("sm__sass_inst_executed_op_local_ld.sum");
        NPerf_metric_collect("sm__sass_inst_executed_op_local_st.sum");
      }
    NPerf_metric_collect("sm__sass_inst_executed_op_global_st.sum");
    NPerf_metric_collect
        ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed");
    NPerf_metric_collect
        ("l1tex__m_l1tex2xbar_throughput.avg.pct_of_peak_sustained_elapsed");
    NPerf_metric_collect("dram__bytes.sum");
    
    
    // ------------ run baseline cuSpmm ----------------
    //input_dfs.c_cuSpmm_run(perfRes);
    //input_deg.c_cuSpmm_run(perfRes);
    //input_rcm.c_cuSpmm_run(perfRes);
    //input_gorder.c_cuSpmm_run(perfRes);
   // input_vo.c_cuSpmm_run(perfRes);
    // ---------------------------------------------------
/*    
    cudaEventRecord(cuspmm_stop);
	cudaEventSynchronize(cuspmm_stop);
	cudaEventElapsedTime(&cuspmm_duration, cuspmm_start, cuspmm_stop);
    float t = cuspmm_duration*(1e-3)/10;
    std::cout<<"cuSpmm time: "<<t<<" s "<<std::endl;
    float gflops = (2*input.cpuA->nnz*input.dim)/(1e+9);
    std::cout<<"cuSpmm Throughput: "<<gflops/t<<" gflops/s "<<std::endl;
    float gb = (float)((input.n+1 + 2*input.cpuA->nnz + 2*input.n*input.dim)*4)/(1e+9);
    std::cout<<"cuSpmm Bandwidth: "<<gb/t<<" GB/s "<<std::endl;
*/

    // --------- run Flex titling spmm ---------------

    float* const h_res_c = (float*) malloc( input_vo.gpuC_bytes );
    struct App_Kernel_Info {
         App_Kernel_Info  
         (Kernel_Info& k,
          const char *name_b, const char *name_tmpl,
          int i, int nbx, int nby, int nt):
           k_ptr(k.func_ptr),name_base{name_b},name_tmpl{name_tmpl},
           shape_idx{i}, n_threads{nt},n_blocks_x{nbx},n_blocks_y{nby}{}
        GPU_Info_Func k_ptr;
        const char *name_base;
        const char *name_tmpl;
        const int shape_idx;
        const int n_blocks_x, n_blocks_y, n_threads;
    };  
    vector<App_Kernel_Info> kernels;
    vector<Mat> spMats;
    
    #define EXAMINE_KERNEL1(kb,k,sidx,graph,nbx,nby,nt) \
    {  spMats.emplace_back(graph, tileConfs[sidx].tm, tileConfs[sidx].tn); \
       kernels.emplace_back(info.GET_INFO((k)),#kb,#k,sidx,nbx,nby,nt); }

    #define EXAMINE_KERNEL(kb,k,sidx,nbx,nby,nt) \
        EXAMINE_KERNEL1(kb,k,sidx,input_vo,nbx,nby,nt);\
        EXAMINE_KERNEL1(kb,k,sidx,input_rabbit,nbx,nby,nt);\
        EXAMINE_KERNEL1(kb,k,sidx,input_gorder,nbx,nby,nt);\
        EXAMINE_KERNEL1(kb,k,sidx,input_dfs,nbx,nby,nt);
    //EXAMINE_KERNEL1(k,sidx,input_deg);EXAMINE_KERNEL1(k,sidx,input_rcm);
    
    #define SPECIFY_KERNEL(k,sidx,nbx,nby,nt)\
    {const int idx = kernels.size(); \
        EXAMINE_KERNEL(k,(k<tileConfs[sidx].tm,NNZ_LIMIT,4>), sidx, nbx, nby, nt); }
// NBX,NBY,NT are useless currently
#define NBX 1
#define NBY 1
#define NT 1
   
// v7-v8 need to activate macro "COL_MAJ_TILE" in DataLoader.cuh. 
// v4-v6 need to deactivate macro "COL_MAJ_TILE" in DataLoader.cuh.   
// v9 need to deactivate macro "VO_RECOVER" in DataLoader.cuh.   

// v10: w/o buffering, rows-based seg allocation, w/o vec, broadcast within a warp 
// v22: w/o buffering, sm-based seg allocation, w/o vec, broadcast within a warp 
//#define flex_kernel flexspmm_cuda_wo_pre_v22

// v11: w/o buffering, rows-based seg allocation, vec4 dense input, broadcast within a warp 
//#define flex_kernel flexspmm_cuda_w_vec4_v11

// v12: double buffering, rows-based seg allocation, w/o vec 
// v23: double buffering, sm-based seg allocation, w/o vec 
//#define flex_kernel flexspmm_cuda_w_pre_v23


// v15: single buffering, rows-based seg allocation, vec r and c of sparse input  
// v16: single buffering, varying kernels with seg size, vec r and c of sparse input 
// v17: single buffering, a block process contiguous layout segs, vec r and c of sparse input 
// v24: single buffering, sm-based seg allocation, vec r and c of sparse input, warp-seg maping 
//#define flex_kernel flexspmm_cuda_wo_pre_w_vec_v24

// v18: w/o buffering, a block process contiguous layout segs, vec r and c of sparse input 
// v20: w/o buffering, rows-based seg allocation, vec r and c of sparse input  
// v21: w/o buffering, sm-based seg allocation, vec r and c of sparse input  
#define flex_kernel flexspmm_cuda_wo_pre_w_vec_v21

// v13: double buffering, rows-based seg allocation, vec2 dense input 
// v14: double buffering, rows-based seg allocation, vec r and c of sparse input 
// v19: double buffering, a block process contiguous layout segs, vec r and c of sparse input 
//#define flex_kernel flexspmm_cuda_w_pre_w_vec_v19

    // Vector size of instructions that load B matrix elements.
    map<string,int> k_prop_vec_b
      { { "flexspmm_cuda_w_vec4_v11", 4 },
        { "flexspmm_cuda_w_pre_w_vec_v13", 2}
      };

    // Kernels that use a vec2 to load row/column pairs.
    set<string> k_prop_vec2_rc { "flexspmm_cuda_w_pre_w_vec_v14" };

    // RC Direct means that all threads in a warp load the same RC and
    // wht elements from global memory. Otherwise, Different threads
    // load different elements and shared memory or swizzles are used
    // to distribute them.
    set<string> k_prop_rc_direct
      { "flexspmm_cuda_wo_pre_v10", "flexspmm_cuda_w_vec4_v11" };

#ifdef CUBE4X4
        SPECIFY_KERNEL(flex_kernel, 0, NBX, NBY, NT);
#endif
#ifdef RECT8X4
        SPECIFY_KERNEL(flex_kernel, 1, NBX, NBY, NT);
#endif
#ifdef RECT16X4
        SPECIFY_KERNEL(flex_kernel, 2, NBX, NBY, NT);
#endif
#ifdef RECT32X4
        SPECIFY_KERNEL(flex_kernel, 3, NBX, NBY, NT);
#endif
#ifdef RECT64X4
        SPECIFY_KERNEL(flex_kernel, 4, NBX, NBY, NT);
#endif
#ifdef RECT128X4
        SPECIFY_KERNEL(flex_kernel, 5, NBX, NBY, NT);
#endif
#ifdef RECT256X4
        SPECIFY_KERNEL(flex_kernel, 6, NBX, NBY, NT);
#endif
#ifdef RECT4X8
        SPECIFY_KERNEL(flex_kernel, 7, NBX, NBY, NT);
#endif
#ifdef CUBE8X8
        SPECIFY_KERNEL(flex_kernel, 8, NBX, NBY, NT);
#endif
#ifdef RECT16X8
        SPECIFY_KERNEL(flex_kernel, 9, NBX, NBY, NT);
#endif
#ifdef RECT32X8
        SPECIFY_KERNEL(flex_kernel, 10, NBX, NBY, NT);
#endif
#ifdef RECT64X8
        SPECIFY_KERNEL(flex_kernel, 11, NBX, NBY, NT);
#endif
#ifdef RECT128X8
        SPECIFY_KERNEL(flex_kernel, 12, NBX, NBY, NT);
#endif
#ifdef RECT256X8
        SPECIFY_KERNEL(flex_kernel, 13, NBX, NBY, NT);
#endif
#ifdef RECT4X16
        SPECIFY_KERNEL(flex_kernel, 14, NBX, NBY, NT);
#endif
#ifdef RECT8X16
        SPECIFY_KERNEL(flex_kernel, 15, NBX, NBY, NT);
#endif
#ifdef CUBE16X16
        SPECIFY_KERNEL(flex_kernel, 16, NBX, NBY, NT);
#endif
#ifdef RECT32X16
        SPECIFY_KERNEL(flex_kernel, 17, NBX, NBY, NT);
#endif
#ifdef RECT64X16
        SPECIFY_KERNEL(flex_kernel, 18, NBX, NBY, NT);
#endif
#ifdef RECT128X16
        SPECIFY_KERNEL(flex_kernel, 19, NBX, NBY, NT);
#endif
#ifdef RECT256X16
        SPECIFY_KERNEL(flex_kernel, 20, NBX, NBY, NT);
#endif
#ifdef RECT4X32
        SPECIFY_KERNEL(flex_kernel, 21, NBX, NBY, NT);
#endif
#ifdef RECT8X32
        SPECIFY_KERNEL(flex_kernel, 22, NBX, NBY, NT);
#endif
#ifdef RECT16X32
        SPECIFY_KERNEL(flex_kernel, 23, NBX, NBY, NT);
#endif
#ifdef CUBE32X32
        SPECIFY_KERNEL(flex_kernel, 24, NBX, NBY, NT);
#endif
#ifdef RECT64X32
        SPECIFY_KERNEL(flex_kernel, 25, NBX, NBY, NT);
#endif
#ifdef RECT128X32
        SPECIFY_KERNEL(flex_kernel, 26, NBX, NBY, NT);
#endif
#ifdef RECT256X32
        SPECIFY_KERNEL(flex_kernel, 27, NBX, NBY, NT);
#endif

    cudaEvent_t spgemm_start, spgemm_stop;
    cudaEventCreate(&spgemm_start);
    cudaEventCreate(&spgemm_stop);

    constexpr int wp_sz = 32;
    constexpr int threads = 128;  // Block Size
    constexpr int block_n_wps = threads / wp_sz;
    assert( block_n_wps * wp_sz == threads );

    vector<Timing_Item> timing_items;
    Timing timing_dh{nullptr};
    size_t timing_items_bytes = 0;

    const vector<string> table_order_1( { "OVO", "DEG", "DFS", "GOR", "RBT" } );
    map<string,int> table_order;
    for ( auto& s: table_order_1 ) table_order[s] = table_order.size();
    for ( auto& mat: spMats )
      if ( !table_order.contains(mat.dl.vertex_order_abbr) )
        table_order[mat.dl.vertex_order_abbr] = table_order.size();

    // Sort kernels so that results are grouped by vertex ordering.
    //
    vector<int> torder;
    for ( int i: views::iota(0,int(kernels.size())) ) torder.push_back(i);
    ranges::stable_sort
      ( torder,
        [&](int ai, int bi){
          Mat& a = spMats[ai];
          Mat& b = spMats[bi];
          return a.tm == b.tm
            ? table_order[a.dl.vertex_order_abbr]
            < table_order[b.dl.vertex_order_abbr]
            : a.tm < b.tm;
        });

    map<string,int> kernels_seen;
    for ( auto& i: torder )
      if ( App_Kernel_Info& aki = kernels[i]; !kernels_seen[aki.name_tmpl]++ )
        {
          Kernel_Info& ki = info.get_info(aki.k_ptr);
          printf("Kernel %s:  %d regs,  %zd local,  %zd B shared.\n",
                 aki.name_tmpl,ki.cfa.numRegs,
                 ki.cfa.localSizeBytes, ki.cfa.sharedSizeBytes );
        }

    const char* stats_file_name = "flex-tile-stats.log";
    FILE *tile_stats = fopen(stats_file_name,"w");
    const char* nperf_file_name = "flex-tile-nperf.csv";
    FILE *tile_nperf = fopen(nperf_file_name,"aw");
    fprintf(tile_nperf,"%s\n",input_vo.graph_name.c_str());
    printf("Writing detailed statistics to file %s\n",stats_file_name);

    pTable table(stdout);
    for ( int id: torder ){
         
        Mat& mat = spMats[id];
        DataLoader& input = mat.dl;
        App_Kernel_Info& aki = kernels[id];

        spMats[id].csr2tile();
        fprintf(tile_stats,"** Data for kernel %s\n",aki.name_tmpl);
        mat.stats_collect2(tile_stats);
        //continue;

        fprintf(tile_stats,"\n");
        if ( table.num_lines == 0 ){
           printf("cuSpmm setup/µs: %.2f , prosessing/µs: %.2f, total/µs: %.2f\n",
                   perfRes.cuSpmmSetup,perfRes.cuSpmmProcessing,perfRes.cuSpmm_time); 
        }
        mat.transfer2();
        spMats[id].dataVolume_est2();
        spMats[id].launch_prep();

        if (input.vertex_order_abbr != "OVO"){
            const int blocks = num_sm * 8;
            flexspmm_v9_permuteX<<<blocks, 128>>>();        
        }
    
        // Compute the expected number of multiply/add instructions.
        //
        const int64_t n_madd = spMats[id].newVals.size()*spMats[id].k; // #FMA
        const double n_madd_p_wp = double(n_madd) / wp_sz;
        const double n_b_elt_p_thd = double(mat.k) / threads;

        vector<uint> grid_sizes;

        if ( opt_vary_grid_size )
          for ( int n_blks = num_sm; n_blks < mat.n_segs; n_blks <<= 1 )
          {
              grid_sizes.push_back( n_blks );
          }
        grid_sizes.push_back( mat.n_segs );

        // Allocate storage for timing data.
        //
        if ( const int max_wps = grid_sizes.back() * block_n_wps;
             timing_items.size() < max_wps )
          {
            timing_items.resize( max_wps );
            timing_items_bytes = timing_items.size() * sizeof(timing_items[0]);
            CE( cudaFree( timing_dh.timing_items ) );
            CE( cudaMalloc( &timing_dh.timing_items, timing_items_bytes ) );
            CE( cudaMemcpyToSymbol
                ( timing_dev, &timing_dh, sizeof(timing_dh),
                  0, cudaMemcpyHostToDevice ) );
          }

        for ( const uint gridx: grid_sizes )
          {
            const dim3 grid_sz = { gridx, 1, 1 };
            const dim3 block_sz = { 128, 1, 1 };
            //uint warps = mat.k<128 ? 2 : 4; // v24 
            //const dim3 block_sz = { 32, warps, 1 }; // v24
            
            const int num_blocks = grid_sz.x * grid_sz.y * grid_sz.z;
            const int grid_n_wps = num_blocks * block_n_wps;


            pTable_Row row(table);
            Kernel_Info* const ki = &info.get_info(kernels[id].k_ptr);
            typedef void (*KPtr)();
            
            CE( cudaMemset( timing_dh.timing_items, 0, timing_items_bytes ) );
            CE( cudaDeviceSynchronize() );
            NPerf_metrics_off();

            float spgemm_duration;
            CE( cudaEventRecord(spgemm_start,0) );

            /// Launch Kernel -- Without Performance Counter Sampling
            //
            CE( cudaMemcpy(mat.next_seg_dev, mat.next_seg.data(), mat.next_seg.size()*sizeof(int), cudaMemcpyHostToDevice) );
            CE( cudaMemset( mat.mat_c_dev, 0.0, mat.m*mat.k*sizeof(float) ) );
            KPtr(ki->func_ptr)<<<grid_sz,block_sz>>>();
            //KPtr(ki->func_ptr)<<<grid_sz,block_sz, 32*warps*(4+4+4)+warps>>>(); //v24
            
            //printf("%d of %s------\n",__LINE__,__FILE__);
            //
            // Until NPerf_metrics_off is fixed event timing won't work.

            CE( cudaEventRecord(spgemm_stop,0) );
            CE( cudaEventSynchronize(spgemm_stop) );
            CE( cudaEventElapsedTime(&spgemm_duration, spgemm_start, spgemm_stop) );
            const float elap_t = spgemm_duration;

            // Copy per-warp timing data back to host.
            //
            CE( cudaMemcpy( timing_items.data(), timing_dh.timing_items,
                            timing_items_bytes, cudaMemcpyDeviceToHost ) );

            NPerf_metrics_on();
            
            /// Launch Kernels -- With Performance Counter Sampling
            //
            CE( cudaMemcpy(mat.next_seg_dev, mat.next_seg.data(), mat.next_seg.size()*sizeof(int), cudaMemcpyHostToDevice) );
            CE( cudaMemset( mat.mat_c_dev, 0.0, mat.m*mat.k*sizeof(float) ) ); 
            for ( NPerf_data_reset(); NPerf_need_run_get(); ){
                KPtr(ki->func_ptr)<<<grid_sz,block_sz>>>();
                //KPtr(ki->func_ptr)<<<grid_sz,block_sz, 32*warps*(4+4+4)+warps>>>(); // v24
            }
            

            // Compute per-sm minimum-start and maximum-finish (end) times.
            //
            map<int32_t,Timing_Item> sm_start_end;
            int n_migs = 0; // Number of migrations.
            for ( auto& ti: views::take(timing_items,grid_n_wps+1) )
              if ( ti.smid_start != ti.smid_end )
                {
                  n_migs++;
                }
              else
                {
                  auto& tis = sm_start_end[ti.smid_start];
                  if ( tis.time_start == tis.time_end ) tis = ti;
                  set_min( tis.time_start, ti.time_start );
                  set_max( tis.time_end, ti.time_end );
                }

            if ( n_migs ) printf("-- Number of migrations: %d\n",n_migs);
            //
            // Note: The per-sm data collection won't work if a block
            // migrates from one sm to another.

            // Compute average sm execution time and maximum sm execution time.
            //
            int64_t et_sum = 0;
            vector<int64_t> et;
            for ( auto& [smid,tis]: sm_start_end )
              {
                const int64_t elapsed = tis.time_end - tis.time_start;
                et_sum += elapsed;
                et.push_back( elapsed );
              }

            ranges::sort( et, ranges::greater() );

            const double clock_period_us = 1e6 / info.clock_freq_hz;
            const double et_clock_max_us = et[0] * clock_period_us;
            const double et_clock_avg_us = et_sum * clock_period_us / num_sm;
            const double imbalance_penalty =
              et_clock_avg_us ? et_clock_max_us / et_clock_avg_us - 1 : 0.0;
            //
            // A value of 0 is ideal. A value of 1 means that execution
            // time is twice as long as a perfectly balanced workload, a
            // value of 1.2 means that execution time was 1+1.2 = 2.2
            // times longer than it would be if the workload were
            // perfectly balanced.

              table.entry("Ord", "%3s", input.vertex_order_abbr);
              fprintf(tile_nperf,"%3s,", input.vertex_order_abbr.c_str());
              //table.entry
              //  ("Tile", "%-6s",
              //   to_string(spMats[id].tm)+"x"+to_string(spMats[id].tn));
              table.entry("tm", "%3d", spMats[id].tm);
              fprintf(tile_nperf,"%3d,", spMats[id].tm);

              // The maximum number of active blocks per SM for this
              // kernel when launched with a block size of thd_per_block.
              //
              const int max_bl_per_sm =
                ki->get_max_active_blocks_per_mp(threads);

              // Compute number of blocks available per SM based only on
              // the number of blocks.  This may be larger than the
              // number of blocks that can run.
              //
              const int bl_per_sm_available =
                0.999 + double(num_blocks) / num_sm;

              // The number of active blocks is the minimum of what
              // can fit and how many are available.
              //
              const int bl_per_sm =
                min( bl_per_sm_available, max_bl_per_sm );

              // Based on the number of blocks, compute the number of warps.
              //
              const int act_wps = block_n_wps * bl_per_sm;

              table.entry("b/s", "%3d", bl_per_sm_available);
              fprintf(tile_nperf,"%3d,", bl_per_sm_available);
              
              table.entry("aw", "%2d", act_wps);
              fprintf(tile_nperf,"%2d,", act_wps);

              table.entry("atm/r", "%5.2f", double(mat.atomic_op)/mat.m );
              fprintf(tile_nperf,"%5.2f,",  double(mat.atomic_op)/mat.m );
              
              const int64_t n_tiles = mat.nnzTile.size();
              const int64_t n_segs = mat.segPtr.size()-1;

              //table.header_span( "T1", 1);
              if (false){
                table.entry
                    ("%", "%2.0f", double(mat.tile_nnz_histo[1])*100.0/n_tiles);
              }
              const double nz_p_t = double(mat.nnz) / n_tiles;
              if (false) table.entry("nz/t", "%5.2f", nz_p_t);
              const double nz_p_seg = double(mat.nnz) / n_segs;
              
              table.entry("nz/seg", "%7.2f", nz_p_seg);
              fprintf(tile_nperf,"%7.2f,", nz_p_seg);

              const double n_t_rows = double(mat.m) / mat.tm;

              const double nz_p_tr = double(mat.nnz) / n_t_rows;

              if ( false ) table.entry("nz/tr", "%5.0f", nz_p_tr);

              const double t_p_tr = n_tiles / n_t_rows;

              if ( false ) table.entry("t/tr", "%4.0f", t_p_tr);

              // Number of nz per occupied tile column.
              const double nz_p_toc = double(mat.nnz) / mat.n_col_sum;
              // Worst-case: 1.  Ideal: Average degree.

              table.entry("B-Re", "%4.2f", nz_p_toc);
              fprintf(tile_nperf,"%4.2f,", nz_p_toc);

              // Get and print elapsed time.
              //
              const double et_seconds = NPerf_kernel_et_get();
              table.entry( "t/µs", "%7.1f", et_seconds * 1e6 );
              fprintf(tile_nperf,"%7.1f,", et_seconds * 1e6 );
              const bool more_timing = false;
              if ( more_timing )
                {
                  table.entry( "M t/µs", "%8.2f", et_clock_max_us );
                  table.entry( "A t/µs", "%8.2f", et_clock_avg_us );
                }
              table.entry( "Imb", "%3.0f", 100 * imbalance_penalty );
              fprintf(tile_nperf,"%3.0f,", 100 * imbalance_penalty );

              // Write a heading that will span multiple columns.
              //
              table.header_span_start("Per Mult");

              table.header_span_start("Num Insns");

              table.header_span_start("Load");

              table.entry
                ( "All", "%4.1f",
                  NPerf_metric_value_get("sm__sass_inst_executed_op_ld.sum")
                  / n_madd_p_wp );
              fprintf(tile_nperf,"%4.1f,", NPerf_metric_value_get("sm__sass_inst_executed_op_ld.sum") / n_madd_p_wp  );
              
              if (false){
                const int n_ld_trow = 2;  // Loads per tile row. tileRowPtr (two)
                const int n_ld_tile = 4;  // Loads per tile.
                const int n_ld_nz = 2;    // Loads per nz. ( rc, edge weight)
                const int n_ld_b_elt = 1; // Loads per B matrix element.
                const double n_ld_p_nz =
                    n_ld_trow / nz_p_tr
                    + n_ld_tile / nz_p_tr * ceil(t_p_tr/32)
                    + n_ld_nz / nz_p_t
                    + n_ld_b_elt / nz_p_toc;
              }

              const bool v_vec2_rc = k_prop_vec2_rc.contains(aki.name_base);
              const int vec_b_sz = max(1,k_prop_vec_b[aki.name_base]);

              const int n_ld_seg = mat.tm+2;  // Loads per tile-segment. (segVoMap + segPtr)

              // Loads per nz. ( r , c, edge weight)
              const int n_ld_nz = 3;
              const int n_ld_insn_nz = 1 + ( v_vec2_rc ? 1 : 2 );

              const int n_ld_b_elt = 1; // Loads per B matrix element.
              const double n_ld_insn_b_elt = double(n_ld_b_elt) / vec_b_sz;

              const double n_ld_p_madd =
                (   n_ld_seg / nz_p_seg
                  + n_ld_insn_nz /
                    ( k_prop_rc_direct.contains(aki.name_base)
                      ? 1.0 : nz_p_seg * ceil(nz_p_seg/32) ) )
                / n_b_elt_p_thd
                + n_ld_insn_b_elt / nz_p_toc;

              table.entry
                ( "G", "%4.1f",
                  NPerf_metric_value_get("sm__sass_inst_executed_op_global_ld.sum")
                  / n_madd_p_wp );
              fprintf(tile_nperf, "%4.1f,",
                  NPerf_metric_value_get("sm__sass_inst_executed_op_global_ld.sum")
                  / n_madd_p_wp );

              table.entry( "GC", "%4.1f", n_ld_p_madd);
              fprintf(tile_nperf, "%4.1f,", n_ld_p_madd);

              if ( show_insn_local )
                table.entry
                  ( "L", "%4.1f",
                    NPerf_metric_value_get("sm__sass_inst_executed_op_local_ld.sum")
                    / n_madd_p_wp );
              if ( show_insn_shared )
                table.entry
                  ( "S", "%4.1f",
                    NPerf_metric_value_get
                    ("sm__sass_inst_executed_op_shared_ld.sum")
                    / n_madd_p_wp );
              table.header_span_end();

              table.header_span_start("Store");
              table.entry
                ( "All", "%5.2f",
                  NPerf_metric_value_get("sm__sass_inst_executed_op_st.sum")
                  / n_madd_p_wp );
              fprintf(tile_nperf, "%5.2f,",
                  NPerf_metric_value_get("sm__sass_inst_executed_op_st.sum")
                  / n_madd_p_wp );
              
              table.entry
                ( "G", "%4.2f",
                  NPerf_metric_value_get("sm__sass_inst_executed_op_global_st.sum")
                  / n_madd_p_wp );
              fprintf(tile_nperf, "%4.2f,",
                  NPerf_metric_value_get("sm__sass_inst_executed_op_global_st.sum")
                  / n_madd_p_wp );
              
              if ( show_insn_local )
                table.entry
                  ( "L", "%3.1f",
                    NPerf_metric_value_get("sm__sass_inst_executed_op_local_st.sum")
                    / n_madd_p_wp );
              if ( show_insn_shared )
                table.entry
                  ( "S", "%3.1f",
                    NPerf_metric_value_get
                    ("sm__sass_inst_executed_op_shared_st.sum")
                    / n_madd_p_wp );
              table.header_span_end();

              table.entry
                ( "All", "%5.1f",
                  NPerf_metric_value_get("sm__inst_executed.sum")
                  / n_madd_p_wp );
              fprintf(tile_nperf, "%5.1f,",
                  NPerf_metric_value_get("sm__inst_executed.sum")
                  / n_madd_p_wp );

              table.header_span_end();

              // Write an extra header line over the next entry.
              table.header_span("Time",1);
              table.entry
                ( "Cyc", "%4.0f",
                  NPerf_metric_value_get("sm__cycles_elapsed.max") * fp32_per_chip
                  / n_madd );
              fprintf(tile_nperf, "%4.0f,",
                  NPerf_metric_value_get("sm__cycles_elapsed.max") * fp32_per_chip
                  / n_madd );

              table.header_span_start("L1←L2");

              // nD = 4/u + 12/k + (2+tm) * #segs * 4 / (k * #nz)
              // nD = 4/u + 12/k + (2+tm) * 4 / (#nz * k / #segs )
              double nD = NPerf_metric_value_get("l1tex__m_xbar2l1tex_read_bytes.sum") / n_madd;
              double u = 4.0 / ( nD - 12.0/mat.k - (2+mat.tm)*mat.n_segs*4.0 / (mat.nnz*mat.k) );
              table.entry( "expBs", "%5.2f",u );
              fprintf(tile_nperf, "%5.2f,",u);

              table.entry
                ( "Bytes", "%5.2f",
                  NPerf_metric_value_get("l1tex__m_xbar2l1tex_read_bytes.sum")
                    / n_madd );
              fprintf(tile_nperf, "%5.2f,",
                  NPerf_metric_value_get("l1tex__m_xbar2l1tex_read_bytes.sum")
                    / n_madd );
              if ( false ){
                  //const double n_bytes =
                  //  4 * ( n_t_rows * n_ld_trow
                  //        + n_tiles * n_ld_tile
                  //        + mat.nnz * n_ld_nz
                  //        + mat.nnz * n_ld_b_elt * mat.k / nz_p_toc );
              }
              const double n_bytes =
                4 * ( n_segs * n_ld_seg
                      + mat.nnz * n_ld_nz
                      + mat.nnz * n_ld_b_elt * mat.k / nz_p_toc );
              table.entry( "BC", "%5.2f", n_bytes / n_madd );
              fprintf(tile_nperf, "%5.2f,", n_bytes / n_madd );
              table.header_span_end();

              table.header_span_end();

              table.header_span_start("Entire GPU");

              table.header_span_start("L1 ⇆ L2");
              table.entry
                ( "GB/s", "%4.0f",
                  ( NPerf_metric_value_get("l1tex__m_l1tex2xbar_write_bytes.sum")
                    + NPerf_metric_value_get("l1tex__m_xbar2l1tex_read_bytes.sum") )
                  / et_seconds * 1e-9 );

              fprintf(tile_nperf, "%4.0f,",
                  ( NPerf_metric_value_get("l1tex__m_l1tex2xbar_write_bytes.sum")
                    + NPerf_metric_value_get("l1tex__m_xbar2l1tex_read_bytes.sum") )
                  / et_seconds * 1e-9 );
              table.entry
                ("% Pk", "%4.1f",
                 NPerf_metric_value_get
                 ("l1tex__m_l1tex2xbar_throughput"
                  ".avg.pct_of_peak_sustained_elapsed"));
              fprintf(tile_nperf, "%4.1f,",
                 NPerf_metric_value_get
                 ("l1tex__m_l1tex2xbar_throughput"
                  ".avg.pct_of_peak_sustained_elapsed"));
              table.header_span_end();

              table.header_span_start("L2 ⇆ DRAM");
              table.entry
                ( "Bytes", "%5.2f",
                  NPerf_metric_value_get("dram__bytes.sum") / n_madd );
              fprintf(tile_nperf, "%5.2f,",
                  NPerf_metric_value_get("dram__bytes.sum") / n_madd );
              table.entry
                ( "GB/s", "%4.0f",
                  NPerf_metric_value_get("dram__bytes.sum") / et_seconds * 1e-9 );
              fprintf(tile_nperf, "%4.0f,",
                  NPerf_metric_value_get("dram__bytes.sum") / et_seconds * 1e-9 );

              table.entry
                ("% Pk", "%5.1f",
                 NPerf_metric_value_get
                 ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"));
              fprintf(tile_nperf, "%5.1f,",
                 NPerf_metric_value_get
                 ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"));

              table.header_span_end();

              if ( true ) {
                  table.header_span( "FP Thpt", 1);
                  table.entry( "GFLOP/s", "%9.1f", 1e-9 * n_madd / et_seconds );
                  fprintf(tile_nperf, "%9.1f\n", 2 * 1e-9 * n_madd / et_seconds );
                  table.header_span_end();
              }
            // transfer data to host
            cudaMemcpy
              ( h_res_c, spMats[id].mat_c_dev, input.gpuC_bytes,
                cudaMemcpyDeviceToHost);
            resCheck( input_vo.h_ref_c.data(), h_res_c, spMats[id], perfRes );
            //fprintf(tile_nperf, "%4f\n", 100.0*perfRes.flex_spmm_errors.back()/(mat.m*mat.k));

            float t = elap_t*(1e-3);
            perfRes.flex_spmm_time.push_back(t);
        } // warps confiuration
        //spMats[id].freeMatGPU();
        spMats[id].freeMatGPU2();
    }

    printf("Key:  b/s,   Blocks per SM\n"
           "      aw,    Active (Resident) Warps per SM\n"
           "      atm/r, Atomic Operations per Row\n"
           "      B-Re,  B Matrix Element Reuse per Segment\n");

    fclose(tile_stats);
    free(h_res_c);
    cuda_freez( timing_dh.timing_items );

}
void cuSpmm(DataLoader& input, Perfs& perfRes){
    float elap_t = 0.0;
    float cuSpmmSetup_duration = 0.0;
    float cuSpmmProcessing_duration = 0.0;
    cudaEvent_t cuSpmmSetup_start, cuSpmmSetup_stop;
    cudaEvent_t cuSpmmProcessing_start, cuSpmmProcessing_stop;
	cudaEventCreate(&cuSpmmSetup_start);
	cudaEventCreate(&cuSpmmSetup_stop); 
	cudaEventCreate(&cuSpmmProcessing_start);
	cudaEventCreate(&cuSpmmProcessing_stop); 
    
    const float alpha = 1.0;
    const float beta = 0.0;

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    
    cudaEventRecord(cuSpmmSetup_start,0);
    
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, input.m, input.n, input.nnz,
                                      input.rowPtr_dev, input.col_dev, input.vals_dev,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, input.n, input.dim, input.dim, input.gpuX,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, input.m, input.dim, input.dim, input.gpuC,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG3, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    
    cudaEventRecord(cuSpmmSetup_stop,0);
    cudaEventSynchronize(cuSpmmSetup_start);
    cudaEventSynchronize(cuSpmmSetup_stop);
    cudaEventElapsedTime(&cuSpmmSetup_duration, cuSpmmSetup_start, cuSpmmSetup_stop);
    perfRes.cuSpmmSetup = cuSpmmSetup_duration * (1e3);  // in microsecond

    // warm-up
    for (int i=0; i<5; ++i){
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG3, dBuffer))
    }
    // execute SpMM
    cudaEventRecord(cuSpmmProcessing_start,0);
    for (int i=0; i<10; ++i){
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG3, dBuffer))
    }
    cudaEventRecord(cuSpmmProcessing_stop,0);
    cudaEventSynchronize(cuSpmmProcessing_start);
    cudaEventSynchronize(cuSpmmProcessing_stop);
    cudaEventElapsedTime(&cuSpmmProcessing_duration, cuSpmmProcessing_start, cuSpmmProcessing_stop);
    elap_t += cuSpmmProcessing_duration;
    float t = elap_t*(1e3)/10; // microsecond
    perfRes.cuSpmmProcessing = t;
    perfRes.cuSpmm_time = cuSpmmSetup_duration*(1e3) + t;
    
    //float gflops = (2*input.nnz*input.dim)/(1e+9);
    //perfRes.cuspmm_throughput = gflops/t/(1e-6);
    //std::cout<<"cuSpmm Throughput: "<<gflops/t<<" gflops/s "<<std::endl;
    //float gb = (float)((input.n+1 + 2*input.nnz + 2*input.n*input.dim)*4)/(1e+9);
    //perfRes.cuspmm_bandwidth = gb/t/(1e-6);
    //std::cout<<"cuSpmm Bandwidth: "<<gb/t<<" GB/s "<<std::endl;
    
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    CHECK_CUDA( cudaFree(dBuffer) )
}
