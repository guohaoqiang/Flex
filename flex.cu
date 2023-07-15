#include "flex.cuh"
#include <ranges>
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
void resCheck(float* h_gold, float* h_res, int m, int n, Perfs& perfRes, const int tm, const int tn){
    // verify results
    int count = 0;
    int nz = 0;
    //std::cout<<"Verify result accuracy ("<< to_string(tm) << "X" << to_string(tn) << ") ... " <<std::endl; 
    for (int i=0; i<m; ++i){
        for (int j=0; j<n; ++j){
          if ( h_gold[i*n+j] == 0 ) nz++;
            if (abs(h_gold[i*n+j]-h_res[i*n+j])>=0.1){
                count++;
                if (j==0) 
                    std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_gold[i*n+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res[i*n+j]<<std::endl;
            }
        }
    }
    perfRes.flex_spmm_errors.push_back(count);
    if (count>0)
        std::cout<<"Kernel ("<< to_string(tm) << "X" << to_string(tn) << ") errs: " << count<<std::endl;

    // If correct result has too many zeros it will be hard to catch errors.
    //assert( nz < n/2 );

    memset(h_res, 0, n*m*sizeof(float));
}
void run(DataLoader& input_vo){

    // Prepare a DFS-ordered matrix.
    DataLoaderDFS input_dfs(input_vo);
    //DataLoaderDeg input_deg(input_vo);
    //DataLoaderRcm input_rcm(input_vo);
    DataLoaderGorder input_gorder(input_vo);

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
    
    Perfs perfRes;
    
    // ------------ run baseline cuSpmm ----------------
    //input_dfs.c_cuSpmm_run(perfRes);
    //input_deg.c_cuSpmm_run(perfRes);
    //input_rcm.c_cuSpmm_run(perfRes);
    //input_gorder.c_cuSpmm_run(perfRes);
    input_vo.c_cuSpmm_run(perfRes);
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
         (Kernel_Info& k,const char *name, int i):
           k_ptr(k.func_ptr),name_base(name),shape_idx{i}{}
        GPU_Info_Func k_ptr;
        const char *name_base;
        const int shape_idx;
    };  
    vector<App_Kernel_Info> kernels;
    vector<Mat> spMats;
    
    #define EXAMINE_KERNEL1(k,sidx,graph) \
    {  spMats.emplace_back(graph, tileConfs[sidx].tm, tileConfs[sidx].tn); \
       kernels.emplace_back(info.GET_INFO((k)),#k,sidx); }

    #define EXAMINE_KERNEL(k,sidx,nbx,nby,nt) \
      EXAMINE_KERNEL1(k,sidx,input_dfs);\
      EXAMINE_KERNEL1(k,sidx,input_gorder);\
      EXAMINE_KERNEL1(k,sidx,input_vo);
    //EXAMINE_KERNEL1(k,sidx,input_deg);EXAMINE_KERNEL1(k,sidx,input_rcm);
    
    #define SPECIFY_KERNEL(k,sidx,nbx,nby,nt)\
    {const int idx = kernels.size(); \
        EXAMINE_KERNEL((k<tileConfs[sidx].tm,tileConfs[sidx].tn,4>), sidx, nbx, nby, nt); }
// NBX,NBY,NT are useless currently
#define NBX 1
#define NBY 1
#define NT 1
   
// v7-v8 need to activate macro "COL_MAJ_TILE" in DataLoader.cuh. 
// v4-v6 need to deactivate macro "COL_MAJ_TILE" in DataLoader.cuh.   
// v9 need to deactivate macro "VO_RECOVER" in DataLoader.cuh.   
#define flex_kernel flexspmm_cuda_wo_pre_v9
//#define flex_kernel flexspmm_cuda_wo_pre_v8

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

    int n_blocks_max = 0;
    for ( auto& m: spMats ) set_max(n_blocks_max,(m.m+m.tm-1)/m.tm);
    const int n_warps_max = threads / wp_sz * n_blocks_max;
    vector<Timing_Item> timing_items( n_warps_max );
    Timing timing_dh;
    const size_t timing_items_bytes =
      timing_items.size() * sizeof( timing_items[0] );
    CUDA_CHECK( cudaMalloc( &timing_dh.timing_items, timing_items_bytes ) );
    CUDA_CHECK( cudaMemcpyToSymbol( timing_dev, &timing_dh, sizeof(timing_dh),
                                    0, cudaMemcpyHostToDevice ) );

    // Sort kernels so that results are grouped by vertex ordering.
    //
    vector<int> torder;
    for ( int i: views::iota(0,int(kernels.size())) ) torder.push_back(i);
    ranges::stable_sort
      ( torder,
        [&](int ai, int bi){
          Mat& a = spMats[ai];
          Mat& b = spMats[bi];
          return a.tn == b.tn
            ? a.dl.vertex_order_abbr < b.dl.vertex_order_abbr
            : a.tn < b.tn;
        });

    map<string,int> kernels_seen;
    for ( auto& i: torder )
      if ( App_Kernel_Info& aki = kernels[i]; !kernels_seen[aki.name_base]++ )
        {
          Kernel_Info& ki = info.get_info(aki.k_ptr);
          printf("Kernel %s:  %d regs,  %zd local\n",
                 aki.name_base,ki.cfa.numRegs, ki.cfa.localSizeBytes);
        }

    const char* stats_file_name = "flex-tile-stats.log";
    FILE *tile_stats = fopen(stats_file_name,"w");
    printf("Writing detailed statistics to file %s\n",stats_file_name);

    pTable table(stdout);
    for ( int id: torder ){
        
        Mat& mat = spMats[id];
        DataLoader& input = mat.dl;
        spMats[id].csr2tile();
        
        fprintf(tile_stats,"** Data for kernel %s\n",kernels[id].name_base);
        mat.stats_collect(tile_stats);
        fprintf(tile_stats,"\n");
        if ( table.num_lines == 0 ){
           printf("cuSpmm setup/µs: %.2f , prosessing/µs: %.2f, total/µs: %.2f\n",
                   perfRes.cuSpmmSetup,perfRes.cuSpmmProcessing,perfRes.cuSpmm_time); 
        }
        mat.transfer();
        spMats[id].dataVolume_est();
        spMats[id].launch_prep();
        
        // V9 requires to activate the following if statement
        if (input.vertex_order_abbr != "OVO"){
            const int blocks = 1024;
            flexspmm_v9_permuteX<<<blocks, 128>>>();        
        }
    
        pTable_Row row(table);
        // Compute the expected number of multiply/add instructions.
        //
        const int64_t n_madd = spMats[id].newVals.size()*spMats[id].k; // #FMA
        const double n_madd_p_wp = double(n_madd) / wp_sz;
                

        Kernel_Info* const ki = &info.get_info(kernels[id].k_ptr);
        typedef void (*KPtr)();
        
        int gridx = (spMats[id].m+spMats[id].tm-1)/spMats[id].tm;
        const int grid_n_wps = threads / wp_sz * gridx;

        CE( cudaMemset( timing_dh.timing_items, 0, timing_items_bytes ) );
        CE( cudaDeviceSynchronize() );
        NPerf_metrics_off();

        float spgemm_duration;
        CE( cudaEventRecord(spgemm_start,0) );

        /// Launch Kernel -- Without Performance Counter Sampling
        //
        KPtr(ki->func_ptr)<<<gridx,threads,0,0>>>();
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
        for ( NPerf_data_reset(); NPerf_need_run_get(); ){
            KPtr(ki->func_ptr)<<<gridx,threads>>>();
        }

        // Compute per-sm minimum-start and maximum-finish (end) times.
        //
        map<int32_t,Timing_Item> sm_start_end;
        int n_migs = 0; // Number of migrations.
        for ( auto& ti: views::take(timing_items,grid_n_wps) )
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
          table.entry
            ("Tile", "%-6s",
             to_string(spMats[id].tm)+"x"+to_string(spMats[id].tn));

          const int64_t n_tiles = mat.nnzTile.size();

          table.entry( "Event/µs", "%8.2f", elap_t * 1e3 );

          table.header_span( "T1", 1);
          table.entry
            ("%", "%2.0f", double(mat.tile_nnz_histo[1])*100.0/n_tiles);

          const double nz_p_t = double(mat.nnz) / n_tiles;

          table.entry("nz/t", "%5.2f", nz_p_t);

          const double n_t_rows = double(mat.m) / mat.tm;

          const double nz_p_tr = double(mat.nnz) / n_t_rows;

          if ( false ) table.entry("nz/tr", "%5.0f", nz_p_tr);

          const double t_p_tr = n_tiles / n_t_rows;

          table.entry("t/tr", "%4.0f", t_p_tr);

          // Number of nz per occupied tile column.
          const double nz_p_toc = double(mat.nnz) / mat.n_col_sum;

          table.entry("B-Re", "%4.2f", nz_p_toc);

          // Get and print elapsed time.
          //
          const double et_seconds = NPerf_kernel_et_get();
          table.entry( "t/µs", "%7.1f", et_seconds * 1e6 );
          const bool more_timing = false;
          if ( more_timing )
            {
              table.entry( "M t/µs", "%8.2f", et_clock_max_us );
              table.entry( "A t/µs", "%8.2f", et_clock_avg_us );
            }
          table.entry( "Imb", "%3.0f", 100 * imbalance_penalty );

          // Write a heading that will span multiple columns.
          //
          table.header_span_start("Per Mult");

          table.header_span_start("Num Insns");

          table.header_span_start("Load");

          table.entry
            ( "All", "%4.1f",
              NPerf_metric_value_get("sm__sass_inst_executed_op_ld.sum")
              / n_madd_p_wp );

          const int n_ld_trow = 2;  // Loads per tile row. tileRowPtr (two)
          const int n_ld_tile = 4;  // Loads per tile.
          const int n_ld_nz = 2;    // Loads per nz. ( rc, edge weight)
          const int n_ld_b_elt = 1; // Loads per B matrix element.
          const double n_ld_p_nz =
            n_ld_trow / nz_p_tr
            + n_ld_tile / nz_p_tr * ceil(t_p_tr/32)
            + n_ld_nz / nz_p_t
            + n_ld_b_elt / nz_p_toc;

          table.entry
            ( "G", "%4.1f",
              NPerf_metric_value_get("sm__sass_inst_executed_op_global_ld.sum")
              / n_madd_p_wp );

          table.entry( "GC", "%4.1f", n_ld_p_nz);

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
          table.entry
            ( "G", "%4.2f",
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

          table.header_span_end();

          // Write an extra header line over the next entry.
          table.header_span("Time",1);
          table.entry
            ( "Cyc", "%4.0f",
              NPerf_metric_value_get("sm__cycles_elapsed.max") * fp32_per_chip
              / n_madd );

          table.header_span_start("L1←L2");
          table.entry
            ( "Bytes", "%5.2f",
              NPerf_metric_value_get("l1tex__m_xbar2l1tex_read_bytes.sum")
                / n_madd );

          const double n_bytes =
            4 * ( n_t_rows * n_ld_trow
                  + n_tiles * n_ld_tile
                  + mat.nnz * n_ld_nz
                  + mat.nnz * n_ld_b_elt * mat.k / nz_p_toc );
          table.entry( "BC", "%5.2f", n_bytes / n_madd );
          table.header_span_end();

          table.header_span_end();

          table.header_span_start("Entire GPU");

          table.header_span_start("L1 ⇆ L2");
          table.entry
            ( "GB/s", "%4.0f",
              ( NPerf_metric_value_get("l1tex__m_l1tex2xbar_write_bytes.sum")
                + NPerf_metric_value_get("l1tex__m_xbar2l1tex_read_bytes.sum") )
              / et_seconds * 1e-9 );

          table.entry
            ("% Pk", "%4.1f",
             NPerf_metric_value_get
             ("l1tex__m_l1tex2xbar_throughput"
              ".avg.pct_of_peak_sustained_elapsed"));

          table.header_span_end();

          table.header_span_start("L2 ⇆ DRAM");
          table.entry
            ( "GB/s", "%4.0f",
              NPerf_metric_value_get("dram__bytes.sum") / et_seconds * 1e-9 );

          table.entry
            ("% Pk", "%5.1f",
             NPerf_metric_value_get
             ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"));

          table.header_span_end();

          if ( true ) {
          table.header_span( "FP Thpt", 1);
          table.entry( "GFLOP/s", "%9.1f", 1e-9 * n_madd / et_seconds );
          table.header_span_end();
          }
        //printf("%d of %s \n",__LINE__,__FILE__); 
        // transfer data to host
        cudaMemcpy
          ( h_res_c, spMats[id].mat_c_dev, input.gpuC_bytes,
            cudaMemcpyDeviceToHost);
        //resCheck
        //  ( input.h_ref_c.data(), h_res_c, spMats[id].m, spMats[id].k,
        //    perfRes, spMats[id].tm, spMats[id].tn);
        resCheck
          ( input_vo.h_ref_c.data(), h_res_c, spMats[id].m, spMats[id].k,
            perfRes, spMats[id].tm, spMats[id].tn);
        
        float t = elap_t*(1e-3);
        perfRes.flex_spmm_time.push_back(t);
        spMats[id].freeMatGPU();
    }

    fclose(tile_stats);
    free(h_res_c);

#ifdef OUTPUTCSV
    std::ofstream myfile(input.graph_name+"_time.csv");
    myfile << "cuSpmm," << "4X4,"<<"8X4,"<<"16X4,"<<"32X4,"<< "64X4,"<<"128X4,"<<"256X4,"
           << "4X8,"<<"8X8,"<<"16X8,"<<"32X8,"<< "64X8,"<<"128X8,"<<"256X8,"
           << "4X16,"<<"8X16,"<<"16X16,"<<"32X16,"<< "64X16,"<<"128X16,"<<"256X16,"
           << "4X32,"<<"8X32,"<<"16X32,"<<"32X32,"<< "64X32,"<<"128X32,"<<"256X32"<<"\n";
    myfile << perfRes.cuispmm_time << ",";
    for (int i=0; i<perfRes.flex_spmm_time.size(); ++i){
        myfile << perfRes.flex_spmm_time[i];
        if (i<perfRes.flex_spmm_time.size()-1)    myfile<<","; 
    }
    myfile << "\n";
    myfile.close(); 
#endif
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
