#include "flex.cuh"
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
// args:
//		tileRowPtr: tile ptr for the 1st tile in each row
//		nnzPtr: ptr for the 1st non zero entry of each tile
// 		nnz: #nnz of each tile
// 		bitMap: mark B rows required by the each tile
// 		tileLeftCol: column idx of each tile. // tba: MSB bit "1" indicates its the last tile in current row-tiles
//      rcOffset: row and column indexfor each non-zero entry
//		vals: non-zero entries
// 		spH: height of sparseMat
// 		mat_b: input dense mat
//		k: width of mat_b
//		mat_c: output dense mat
// A: sparse, m * n
// B: dense, n * k   (k << n)
template<int tm, int tn, int warps>
__global__
void flexspgemm_cuda_wo_pre_v4(int* tileRowPtr,
                int* nnzPtr,
                int* nnz,
                int* bitMap,
                int* tileLeftCol,
				int* rcOffset,
				float* vals,
				int spH,
				float* mat_b,
				int k,
                float* mat_c){
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
	for (int row_idx=blockIdx.x*tileRows_perBlk; row_idx<(spH+tm-1)/tm; row_idx += (gridDim.x*tileRows_perBlk)){ // over C rows
	   
        int tile_curR_id = 0, tile_nxtR_id = 0;
        //int temp_tile_id = 0;
        //if (lane_id<2){
        //    temp_tile_id = tileRowPtr[row_idx+lane_id]; 
        //}
        //__syncwarp();
        //tile_curR_id = __shfl_sync(FULL_MASK, temp_tile_id, 0);
        //tile_nxtR_id = __shfl_sync(FULL_MASK, temp_tile_id, 1);
        tile_curR_id = tileRowPtr[row_idx]; 
        tile_curR_id = tileRowPtr[row_idx+1]; 

        for (int col_idx=warp_id*(32*computeWidth); col_idx<k; col_idx += warps*(32*computeWidth)){  // over C tile columns
             
            int tiles = 0;

            for (int tile_id=tile_curR_id; tile_id<tile_nxtR_id; tile_id+=tiles){

                uint32_t mask_tiles = __ballot_sync(FULL_MASK, tile_id+lane_id<tile_nxtR_id);
                tiles = __popc(mask_tiles); // maximum # tiles can be loaded in cur row 
                
                int start_of_tile = 0, nnz_of_tile = 0, bitmap_of_tile = 0, col_of_tile = 0;
                if (tile_curR_id+lane_id<tile_nxtR_id){
                    // load as many as as tile info of cur tile-row
                    start_of_tile = nnzPtr[tile_id+lane_id];
                    nnz_of_tile = nnz[tile_id+lane_id];
                    bitmap_of_tile = bitMap[tile_id+lane_id];
                    col_of_tile = tileLeftCol[tile_id+lane_id];
                }

                // use all loaded tiles
                for(int tile_cnt = 0; tile_cnt<tiles; ++tile_cnt){
                    int start_cur_tile = __shfl_sync(FULL_MASK, start_of_tile, tile_cnt);
                    int nnz_cur_tile = __shfl_sync(FULL_MASK, nnz_of_tile, tile_cnt);
                    int bitmap_cur_tile = __shfl_sync(FULL_MASK, bitmap_of_tile, tile_cnt);
                    int col_cur_tile = __shfl_sync(FULL_MASK, col_of_tile, tile_cnt);
                    
					// load requiring B rows to smem
					for (int j=0; j<tn; ++j){
						if ((bitmap_cur_tile & (1<<j)) && col_idx+lane_id<k){
                            curB[warp_id][j*32+lane_id] = mat_b[(col_cur_tile+j)*k + col_idx + lane_id];
						}
					}
					//__syncwarp(); // I doubt if it is necessary besause warp is the minimum sheduling unit

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
                		    val = vals[kk+lane_id];
                		    rcidx = rcOffset[kk+lane_id];
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
                if (row_idx*tm+c<spH){
                    mat_c[(row_idx*tm+c)*k+col_idx+lane_id] = res[c];
                }
                res[c] = 0;
            }
    
		} // end C column loops
	} // end C row loops
}
/*
void run_test(float* h_res_c, DataLoader& input, 
                const float* mat_b, 
                int tilem,
                int tilen  
                int tilek, 
                int warmup, 
                int runs, 
                Perfs& perfRes){
    
    mat<tilem,tilek> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
    
    //std::cout<<data.vals[0]<<","<<data.vals[32]<<","<<data.vals[64]<<std::endl;
	data.csr2tile();
    //std::cout<<data.newVals[0]<<","<<data.newVals[32]<<","<<data.newVals[64]<<std::endl;
	//data.print1();
	//data.print2();
     
    flexspgemm(h_res_c, data, host_mat_b, perfRes);

}
*/
void run(DataLoader& input){
    Perfs perfRes;
    
    // ------------ run baseline cuspgemm ----------------
    float* host_mat_b = (float*)malloc(input.n*input.dim*sizeof(float)); 
    for (int i=0; i<input.n*input.dim; ++i){
        host_mat_b[i] = input.cpuX[i];
    }
    cuSpgemm(input, perfRes);
    float* h_ref_c = (float*)malloc(input.n*input.dim*sizeof(float)); 
    CUDA_CHECK(cudaMemcpy(h_ref_c, input.gpuRef1, sizeof(float)*input.n*input.dim, cudaMemcpyDeviceToHost));
    // ---------------------------------------------------
/*    
    cudaEventRecord(cuspgemm_stop);
	cudaEventSynchronize(cuspgemm_stop);
	cudaEventElapsedTime(&cuspgemm_duration, cuspgemm_start, cuspgemm_stop);
    float t = cuspgemm_duration*(1e-3)/10;
    std::cout<<"cuSpgemm time: "<<t<<" s "<<std::endl;
    float gflops = (2*input.cpuA->nnz*input.dim)/(1e+9);
    std::cout<<"cuSpgemm Throughput: "<<gflops/t<<" gflops/s "<<std::endl;
    float gb = (float)((input.n+1 + 2*input.cpuA->nnz + 2*input.n*input.dim)*4)/(1e+9);
    std::cout<<"cuSpgemm Bandwidth: "<<gb/t<<" GB/s "<<std::endl;
*/

    // --------- run flex titling spgemm ---------------
    //vector<vector<int>> mnk = {{16,16,16},{32,8,16},{8,32,16}};

    float* h_res_c = (float*)malloc(input.n*input.dim*sizeof(float)); 
    int warmup = 5;
    int runs = 10;
#ifdef CUBE4X4
    {
        mat<4,4> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
	    //data.print2();
        flexspgemm<mat<4,4>, 4, 4>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (4X4) ...";
        printf("%d of %s, Verify result accuracy (4X4) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.01){
                    count++;
                    //if (j==0) 
                    //    std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@246:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT8X4
    {
        mat<8,4> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<8,4>, 8, 4>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (8X4) ...";
        printf("%d of %s, Verify result accuracy (8X4) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.01){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@269:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT16X4
    {
        mat<16,4> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        //data.print2();
        flexspgemm<mat<16,4>, 16, 4>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (16X4) ...";
        printf("%d of %s, Verify result accuracy (16X4) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.01){
                    count++;
                    //if (j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@293:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT32X4
    {
        mat<32,4> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<32,4>, 32, 4>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (32X4) ...";
        printf("%d of %s, Verify result accuracy (32X4) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@316:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT64X4
    {
        mat<64,4> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<64,4>, 64, 4>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (64X4) ...";
        printf("%d of %s, Verify result accuracy (64X4) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.01){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@339:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT128X4
    {
        mat<128,4> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<128,4>, 128, 4>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (128X4) ...";
        printf("%d of %s, Verify result accuracy (128X4) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.01){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@362:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT256X4
    {
        mat<256,4> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<256,4>, 256, 4>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (256X4) ...";
        printf("%d of %s, Verify result accuracy (256X4) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@385:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT4X8
    {
        mat<4,8> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<4,8>, 4, 8>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (4X8) ...";
        printf("%d of %s, Verify result accuracy (4X8) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@408:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef CUBE8X8
    {
        mat<8,8> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<8,8>, 8, 8>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (8X8) ...";
        printf("%d of %s, Verify result accuracy (8X8) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@431:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT16X8
    {
        mat<16,8> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<16,8>, 16, 8>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (16X8) ...";
        printf("%d of %s, Verify result accuracy (16X8) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@454:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT32X8
    {
        mat<32,8> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<32,8>, 32, 8>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (32X8) ...";
        printf("%d of %s, Verify result accuracy (32X8) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@477:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT64X8
    {
        mat<64,8> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<64,8>, 64, 8>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (64X8) ...";
        printf("%d of %s, Verify result accuracy (64X8) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@500:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT128X8
    {
        mat<128,8> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<128,8>, 128, 8>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (128X8) ...";
        printf("%d of %s, Verify result accuracy (128X8) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@523:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT256X8
    {
        mat<256,8> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<256,8>, 256, 8>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (256X8) ...";
        printf("%d of %s, Verify result accuracy (256X8) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@546:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT4X16
    {
        mat<4,16> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<4,16>, 4, 16>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (4X16) ...";
        printf("%d of %s, Verify result accuracy (4X16) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@569:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT8X16
    {
        mat<8,16> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<8,16>, 8, 16>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (8X16) ...";
        printf("%d of %s, Verify result accuracy (8X16) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@592:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef CUBE16X16
    {
        mat<16,16> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<16,16>, 16, 16>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (16X16) ...";
        printf("%d of %s, Verify result accuracy (16X16) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@615:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT32X16
    {
        mat<32,16> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<32,16>, 32, 16>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (32X16) ...";
        printf("%d of %s, Verify result accuracy (32X16) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@638:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT64X16
    {
        mat<64,16> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<64,16>, 64, 16>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (64X16) ...";
        printf("%d of %s, Verify result accuracy (64X16) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@661:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT128X16
    {
        mat<128,16> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<128,16>, 128, 16>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (128X16) ...";
        printf("%d of %s, Verify result accuracy (128X16) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@684:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT256X16
    {
        mat<256,16> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<256,16>, 256, 16>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (256X16) ...";
        printf("%d of %s, Verify result accuracy (256X16) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@661:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT4X32
    {
        mat<4,32> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<4,32>, 4, 32>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (4X32) ...";
        printf("%d of %s, Verify result accuracy (4X32) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@730:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT8X32
    {
        mat<8,32> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<8,32>, 8, 32>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (8X32) ...";
        printf("%d of %s, Verify result accuracy (8X32) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@753:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT16X32
    {
        mat<16,32> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<16,32>, 16, 32>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (16X32) ...";
        printf("%d of %s, Verify result accuracy (16X32) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@776:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef CUBE32X32
    {
        mat<32,32> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<32,32>, 32, 32>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (32X32) ...";
        printf("%d of %s, Verify result accuracy (32X32) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@799:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT64X32
    {
        mat<64,32> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<64,32>, 64, 32>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (64X32) ...";
        printf("%d of %s, Verify result accuracy (64X32) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@822:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT128X32
    {
        mat<128,32> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<128,32>, 128, 32>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (128X32) ...";
        printf("%d of %s, Verify result accuracy (128X32) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@845:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT256X32
    {
        mat<256,32> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm<mat<256,32>, 256, 32>(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        //LOG(INFO) << "Verify result accuracy (256X32) ...";
        printf("%d of %s, Verify result accuracy (256X32) ...\n",__LINE__,__FILE__);
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    //std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        std::cout<<"@868:   Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
    free(h_res_c);
    free(h_ref_c);
    free(host_mat_b);

    int show_idx = 0;
    std::cout<<setw(20)<<left<<"sparse mat (M X N) "
        <<setw(18)<<left<<" tile size (tm X tn) "
        <<setw(20)<<left<<"  dense mat (N X K) "
        <<setw(15)<<left<<" cuspgemm t "
        <<setw(15)<<left<<" flex_spgemm t "
        <<setw(20)<<left<<" flex_spgemm errors "<<std::endl;
#ifdef CUBE4X4
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 4 X 4 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT8X4
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 8 X 4 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT16X4
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 16 X 4 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT32X4
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 32 X 4 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT64X4
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 64 X 4 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT128X4
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 128 X 4 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT256X4
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 256 X 4 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT4X8
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 4 X 8 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef CUBE8X8
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 8 X 8 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT16X8
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 16 X 8 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT32X8
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 32 X 8 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT64X8
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 64 X 8 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT128X8
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 128 X 8 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT256X8
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 256 X 8 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT4X16
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 4 X 16 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT8X16
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 8 X 16 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef CUBE16X16
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 16 X 16 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT32X16
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 32 X 16 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT64X16
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 64 X 16 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT128X16
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 128 X 16 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT256X16
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 256 X 16 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT4X32
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 4 X 32 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT8X32
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 8 X 32 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT16X32
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 16 X 32 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef CUBE32X32
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 32 X 32 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT64X32
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 64 X 32 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT128X32
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 128 X 32 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef RECT256X32
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 256 X 32 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[show_idx])
        <<setw(20)<<left<<to_string(perfRes.flex_spgemm_errors[show_idx++])<<std::endl;
#endif
#ifdef OUTPUTCSV
    std::ofstream myfile(input.graph_name+"_time.csv");
    myfile << "cuSpgemm," << "4X4,"<<"8X4,"<<"16X4,"<<"32X4,"<< "64X4,"<<"128X4,"<<"256X4,"
           << "4X8,"<<"8X8,"<<"16X8,"<<"32X8,"<< "64X8,"<<"128X8,"<<"256X8,"
           << "4X16,"<<"8X16,"<<"16X16,"<<"32X16,"<< "64X16,"<<"128X16,"<<"256X16,"
           << "4X32,"<<"8X32,"<<"16X32,"<<"32X32,"<< "64X32,"<<"128X32,"<<"256X32"<<"\n";
    myfile << perfRes.cuspgemm_time << ",";
    for (int i=0; i<perfRes.flex_spgemm_time.size(); ++i){
        myfile << perfRes.flex_spgemm_time[i];
        if (i<perfRes.flex_spgemm_time.size()-1)    myfile<<","; 
    }
    myfile << "\n";
    myfile.close(); 
#endif
}

template<typename MT, int tm, int tn>
void flexspgemm(float* h_res_c, MT& data, const float* mat_b, Perfs& perfRes){

	// allocate device memory
    // index of the first nz entry in each tile, length = #tiles+1
    int* d_tileNnz; 
	CHECK_CUDA(cudaMalloc(&d_tileNnz, data.nnzPtr.size()*sizeof(int)));
    
#ifdef V3_KERNEL
    // index of the first tile for each thread block, length = #blocks+1
    int* d_block_tileStart_idx; 
	CHECK_CUDA(cudaMalloc(&d_block_tileStart_idx, data.block_tileStart_idx.size()*sizeof(int)));
    
    // row index of tiles for each thread block, length = #blocks
    int* d_warp_tileRow_idx; 
	CHECK_CUDA(cudaMalloc(&d_warp_tileRow_idx, data.warp_tileRow_idx.size()*sizeof(int)));
	
    // row&col index of vals in sparse matrix, length = nnz
    char* d_r_c_Offset; 
	CHECK_CUDA(cudaMalloc(&d_r_c_Offset, data.rc_Offset.size()*sizeof(char)));
#endif
    // column index of tiles, length = #tiles
    int* d_tileColIdx; 
	CHECK_CUDA(cudaMalloc(&d_tileColIdx, data.tileLeftColIdx.size()*sizeof(int)));
      
    // non-zero vals of sparse matrix, length = nnz
    float* d_vals; 
	CHECK_CUDA(cudaMalloc(&d_vals, data.newVals.size()*sizeof(int)));
   

    // v4 kernel
    int* d_tileRowPtr; 
	CHECK_CUDA(cudaMalloc(&d_tileRowPtr, data.tileRowPtr.size()*sizeof(int)));
    //std::cout<<"@536: metaTile.size() = "<<data.metaTile.size()<<std::endl;
    int* d_nnzTile; 
	CHECK_CUDA(cudaMalloc(&d_nnzTile, data.nnzTile.size()*sizeof(int)));
    int* d_bitMap; 
	CHECK_CUDA(cudaMalloc(&d_bitMap, data.bitMap.size()*sizeof(int)));
    int* d_rcOffset; 
	CHECK_CUDA(cudaMalloc(&d_rcOffset, data.rcOffset.size()*sizeof(int)));
    //std::cout<<"@539: rcOffset.size() = "<<data.rcOffset.size()<<std::endl;

	//data.print2();

    /*
    // Matrix B
    float* mat_b = (float*)malloc(data.m*data.k*sizeof(float));
    for (size_t i=0; i<data.m; ++i){
        for (size_t j=0; j<data.k; ++j){
            mat_b[i*data.k+j] = 1.0;
        }
    }
    */
    float* d_mat_b; 
	CHECK_CUDA(cudaMalloc(&d_mat_b, data.m*data.k*sizeof(float)));
    
    // Matrix C
    float* d_mat_c; 
	CHECK_CUDA(cudaMalloc(&d_mat_c, data.m*data.k*sizeof(float)));
    cudaMemset(d_mat_c, 0.0, data.m*data.k*sizeof(float));
    cudaDeviceSynchronize(); 
    
    
    // transfer data to device
	cudaMemcpy(d_tileNnz, data.nnzPtr.data(), data.nnzPtr.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tileColIdx, data.tileLeftColIdx.data(), data.tileLeftColIdx.size()*sizeof(int), cudaMemcpyHostToDevice);
#ifdef V3_KERNEL
    cudaMemcpy(d_block_tileStart_idx, data.block_tileStart_idx.data(), data.block_tileStart_idx.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_warp_tileRow_idx, data.warp_tileRow_idx.data(), data.warp_tileRow_idx.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r_c_Offset, data.rc_Offset.data(), data.rc_Offset.size()*sizeof(char), cudaMemcpyHostToDevice);
#endif
    cudaMemcpy(d_vals, data.newVals.data(), data.newVals.size()*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat_b, mat_b, data.m*data.k*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_mat_c, mat_c, data.m*data.k*sizeof(float), cudaMemcpyHostToDevice);

    // v4 kernel
	cudaMemcpy(d_tileRowPtr, data.tileRowPtr.data(), data.tileRowPtr.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nnzTile, data.nnzTile.data(), data.nnzTile.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bitMap, data.bitMap.data(), data.bitMap.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rcOffset, data.rcOffset.data(), data.rcOffset.size()*sizeof(int), cudaMemcpyHostToDevice);
    

	// each thread block has 2 warps
	//dim3 grid(data.block_tileStart_idx.size()-1, (data.k+31)/32);
    //printf("@415:   data.block_tileStart_idx.size() = %d\n",data.block_tileStart_idx.size());
    //printf("@416:   data.k = %d\n",data.k);
    //LOG(INFO) << "Ahead the kernel ...";
    printf("%d of %s, Ahead the kernel ...\n",__LINE__,__FILE__);
    //std::cout<<"block_tileStart_idx:"<<std::endl;
    //print(block_tileStart_idx);
    //std::cout<<"warp_tileRow_idx:"<<std::endl;
    //print(warp_tileRow_idx);
	
    int gridx = (data.m+tm-1)/tm;
    //int gridx = 4096;
    int threads = 128;
    // warm up
    for (int i=0; i<5; ++i){
     /*
        //flexspgemm_cuda_reg_pre_v2<<<grid, 64>>>(d_tileNnz, 
        flexspgemm_cuda_wo_pre_v3<<<grid, 64>>>(d_tileNnz,
                                                d_block_tileStart_idx, 
                                                d_warp_tileRow_idx, 
                                                d_tileColIdx, 
                                                data.tileLeftColIdx.size(), 
                                                d_r_c_Offset, 
                                                d_vals, 
                                                data.k, 
                                                d_mat_b, 
                                                d_mat_c);
       */ 
        flexspgemm_cuda_wo_pre_v4<tm, tn, 4><<<gridx,threads>>>(d_tileRowPtr, 
                                                             d_tileNnz, 
                                                             d_nnzTile,
                                                             d_bitMap,
                                                             d_tileColIdx,
                                                             d_rcOffset, 
                                                             d_vals, 
                                                             data.m,
                                                             d_mat_b, 
                                                             data.k, 
                                                             d_mat_c);
    }
    cudaMemset(d_mat_c, 0.0, data.m*data.k*sizeof(float));
    // run test
    float spgemm_duration;
    float elap_t = 0; 
    cudaEvent_t spgemm_start, spgemm_stop;
	cudaEventCreate(&spgemm_start);
	cudaEventCreate(&spgemm_stop);
    cudaEventRecord(spgemm_start);
    for (int i=0; i<10; ++i){
        //std::cout<<"@618 -----------------------   i = "<<i<<" gridx = "<<gridx<<std::endl;
        /*
        //flexspgemm_cuda_reg_pre_v2<<<grid, 64>>>(d_tileNnz,
        flexspgemm_cuda_wo_pre_v3<<<grid, 64>>>(d_tileNnz,
                                        d_block_tileStart_idx,
                                        d_warp_tileRow_idx,
                                        d_tileColIdx,
                                        data.tileLeftColIdx.size(),
                                        d_r_c_Offset,
                                        d_vals,
                                        data.k,
                                        d_mat_b,
                                        d_mat_c);
	    */
        
        flexspgemm_cuda_wo_pre_v4<tm, tn, 4><<<gridx,threads>>>(d_tileRowPtr, 
                                                             d_tileNnz, 
                                                             d_nnzTile,
                                                             d_bitMap,
                                                             d_tileColIdx,
                                                             d_rcOffset, 
                                                             d_vals, 
                                                             data.m,
                                                             d_mat_b, 
                                                             data.k, 
                                                             d_mat_c);
        
        //if (i<9) cudaMemset(d_mat_c, 0.0, data.m*data.k*sizeof(float));
    }
    cudaEventRecord(spgemm_stop);
    cudaEventSynchronize(spgemm_stop);
    cudaEventElapsedTime(&spgemm_duration, spgemm_start, spgemm_stop);
    elap_t += spgemm_duration; 
    cudaDeviceSynchronize(); 
    //LOG(INFO) << "After the kernel ...";
    printf("%d of %s, After the kernel ...\n",__LINE__,__FILE__);

    // transfer data to host
    //LOG(INFO) << "Transfer results back ...";
    printf("%d of %s, Transfer results back ...\n",__LINE__,__FILE__);
	cudaMemcpy(h_res_c, d_mat_c, data.m*data.k*sizeof(float), cudaMemcpyDeviceToHost);
    
    float t = elap_t*(1e-3)/10;
    perfRes.flex_spgemm_time.push_back(t);
    //std::cout<<"Flexspgemm time: "<<t<<" s "<<std::endl;
    float gflops = (2*data.newVals.size()*data.k)/(1e+9);
    perfRes.flex_spgemm_throughput.push_back(gflops/t);
    //std::cout<<"Flexspgemm Throughput: "<<gflops/t<<" gflops/s "<<std::endl;
    float gb = (float)((data.nnzPtr.size()+data.block_tileStart_idx.size()
            +data.warp_tileRow_idx.size()+data.tileColIdx.size()
            +data.tileLeftColIdx.size()+data.newVals.size()+2*data.m*data.k)*4+data.rc_Offset.size())/(1e+9);
    perfRes.flex_spgemm_bandwidth.push_back(gb/t);
    //std::cout<<"Flexspgemm Bandwidth: "<<gb/t<<" GB/s "<<std::endl;
    
    CHECK_CUDA(cudaFree(d_tileNnz));
#ifdef V3_KERNEL
	CHECK_CUDA(cudaFree(d_block_tileStart_idx));
	CHECK_CUDA(cudaFree(d_warp_tileRow_idx));
	CHECK_CUDA(cudaFree(d_r_c_Offset));
#endif
    CHECK_CUDA(cudaFree(d_tileColIdx));
    CHECK_CUDA(cudaFree(d_vals));
	CHECK_CUDA(cudaFree(d_mat_b));
	CHECK_CUDA(cudaFree(d_mat_c));
    
    // v4 kernel
    CHECK_CUDA(cudaFree(d_tileRowPtr));
    CHECK_CUDA(cudaFree(d_nnzTile));
    CHECK_CUDA(cudaFree(d_bitMap));
    CHECK_CUDA(cudaFree(d_rcOffset));
}
/*
void flexspgemm(float* h_res_c, const mat& data, const float* mat_b, Perfs& perfRes){

	// allocate device memory
    // index of the first nz entry in each tile, length = #tiles+1
    int* d_tileNnz; 
	cudaMalloc(&d_tileNnz, data.nnzPtr.size()*sizeof(int));
    
    // index of the first tile for each thread block, length = #blocks+1
    int* d_block_tileStart_idx; 
	cudaMalloc(&d_block_tileStart_idx, data.block_tileStart_idx.size()*sizeof(int));
    
    // row index of tiles for each thread block, length = #blocks
    int* d_warp_tileRow_idx; 
	cudaMalloc(&d_warp_tileRow_idx, data.warp_tileRow_idx.size()*sizeof(int));
	
    // column index of tiles, length = #tiles
    int* d_tileColIdx; 
	cudaMalloc(&d_tileColIdx, data.tileLeftColIdx.size()*sizeof(int));
     
    // row&col index of vals in sparse matrix, length = nnz
    char* d_r_c_Offset; 
	cudaMalloc(&d_r_c_Offset, data.rc_Offset.size()*sizeof(char));
    
    // non-zero vals of sparse matrix, length = nnz
    float* d_vals; 
	cudaMalloc(&d_vals, data.newVals.size()*sizeof(int));
    
    
    // Matrix B
    //float* mat_b = (float*)malloc(data.m*data.k*sizeof(float));
    //for (size_t i=0; i<data.m; ++i){
    //    for (size_t j=0; j<data.k; ++j){
    //        mat_b[i*data.k+j] = 1.0;
    //    }
    //}
    
    float* d_mat_b; 
	cudaMalloc(&d_mat_b, data.m*data.k*sizeof(float));
    
    // Matrix C
    float* d_mat_c; 
	cudaMalloc(&d_mat_c, data.m*data.k*sizeof(float));
    cudaMemset(d_mat_c, 0.0, data.m*data.k*sizeof(float));
    cudaDeviceSynchronize(); 
    
    
    // transfer data to device
	cudaMemcpy(d_tileNnz, data.nnzPtr.data(), data.nnzPtr.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_block_tileStart_idx, data.block_tileStart_idx.data(), data.block_tileStart_idx.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_warp_tileRow_idx, data.warp_tileRow_idx.data(), data.warp_tileRow_idx.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tileColIdx, data.tileLeftColIdx.data(), data.tileLeftColIdx.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r_c_Offset, data.rc_Offset.data(), data.rc_Offset.size()*sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vals, data.newVals.data(), data.newVals.size()*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat_b, mat_b, data.m*data.k*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_mat_c, mat_c, data.m*data.k*sizeof(float), cudaMemcpyHostToDevice);

	// each thread block has 2 warps
	dim3 grid(data.block_tileStart_idx.size()-1, (data.k+31)/32);
    LOG(INFO) << "Ahead the kernel ...";
    //std::cout<<"block_tileStart_idx:"<<std::endl;
    //print(block_tileStart_idx);
    //std::cout<<"warp_tileRow_idx:"<<std::endl;
    //print(warp_tileRow_idx);
	
    // warm up
    for (size_t i=0; i<5; ++i){
        flexspgemm_cuda_reg_pre<<<grid, 64>>>(d_tileNnz,
                                        d_block_tileStart_idx,
                                        d_warp_tileRow_idx,
                                        d_tileColIdx,
                                        data.tileLeftColIdx.size(),
                                        d_r_c_Offset,
                                        d_vals,
                                        data.k,
                                        d_mat_b,
                                        d_mat_c);
        cudaMemset(d_mat_c, 0.0, data.m*data.k*sizeof(float));
    }
    // run test
    float spgemm_duration;
    cudaEvent_t spgemm_start, spgemm_stop;
	cudaEventCreate(&spgemm_start);
	cudaEventCreate(&spgemm_stop);
    float elap_t = 0; 
    for (size_t i=0; i<10; ++i){
        cudaEventRecord(spgemm_start);
        flexspgemm_cuda_reg_pre<<<grid, 64>>>(d_tileNnz,
                                        d_block_tileStart_idx,
                                        d_warp_tileRow_idx,
                                        d_tileColIdx,
                                        data.tileLeftColIdx.size(),
                                        d_r_c_Offset,
                                        d_vals,
                                        data.k,
                                        d_mat_b,
                                        d_mat_c);
	    cudaEventRecord(spgemm_stop);
	    cudaEventSynchronize(spgemm_stop);
	    cudaEventElapsedTime(&spgemm_duration, spgemm_start, spgemm_stop);
        elap_t += spgemm_duration;
        if (i<9) cudaMemset(d_mat_c, 0.0, data.m*data.k*sizeof(float));
    }
    
    cudaDeviceSynchronize(); 
    LOG(INFO) << "After the kernel ...";

    // transfer data to host
    LOG(INFO) << "Transfer results back ...";
	cudaMemcpy(h_res_c, d_mat_c, data.m*data.k*sizeof(float), cudaMemcpyDeviceToHost);
    
    float t = elap_t*(1e-3)/10;
    perfRes.flex_spgemm_time.push_back(t);
    //std::cout<<"Flexspgemm time: "<<t<<" s "<<std::endl;
    float gflops = (2*data.newVals.size()*data.k)/(1e+9);
    perfRes.flex_spgemm_throughput.push_back(gflops/t);
    //std::cout<<"Flexspgemm Throughput: "<<gflops/t<<" gflops/s "<<std::endl;
    float gb = (float)((data.nnzPtr.size()+data.block_tileStart_idx.size()
            +data.warp_tileRow_idx.size()+data.tileColIdx.size()
            +data.tileLeftColIdx.size()+data.newVals.size()+2*data.m*data.k)*4+data.rc_Offset.size())/(1e+9);
    perfRes.flex_spgemm_bandwidth.push_back(gb/t);
    //std::cout<<"Flexspgemm Bandwidth: "<<gb/t<<" GB/s "<<std::endl;
    
    CHECK_CUDA(cudaFree(d_tileNnz));
	CHECK_CUDA(cudaFree(d_block_tileStart_idx));
	CHECK_CUDA(cudaFree(d_warp_tileRow_idx));
	CHECK_CUDA(cudaFree(d_tileColIdx));
	CHECK_CUDA(cudaFree(d_r_c_Offset));
    CHECK_CUDA(cudaFree(d_vals));
	CHECK_CUDA(cudaFree(d_mat_b));
	CHECK_CUDA(cudaFree(d_mat_c)); 
}
*/

void cuSpgemm(DataLoader& input, Perfs& perfRes){
    float elap_t = 0.0;
    float cuspgemm_duration;
    cudaEvent_t cuspgemm_start, cuspgemm_stop;
	cudaEventCreate(&cuspgemm_start);
	cudaEventCreate(&cuspgemm_stop); 
    
    const float alpha = 1.0;
    const float beta = 0.0;

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    
    //cudaEventRecord(cuspgemm_start);
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, input.cpuA->r, input.cpuA->c, input.cpuA->nnz,
                                      input.gpuA->row, input.gpuA->col, input.gpuA->vals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, input.n, input.dim, input.dim, input.gpuX,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, input.n, input.dim, input.dim, input.gpuRef1,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG3, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    
    //cudaEventRecord(cuspgemm_stop);
    //cudaEventSynchronize(cuspgemm_stop);
    //cudaEventElapsedTime(&cuspgemm_duration, cuspgemm_start, cuspgemm_stop);
    //elap_t += cuspgemm_duration;

    // warm-up
    for (int i=0; i<5; ++i){
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG3, dBuffer))
    }
    // execute SpMM
    cudaEventRecord(cuspgemm_start);
    for (int i=0; i<10; ++i){
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG3, dBuffer))
    }
    cudaEventRecord(cuspgemm_stop);
    cudaEventSynchronize(cuspgemm_stop);
    cudaEventElapsedTime(&cuspgemm_duration, cuspgemm_start, cuspgemm_stop);
    elap_t += cuspgemm_duration;
    float t = elap_t*(1e-3)/10;
    perfRes.cuspgemm_time = t;
    
    float gflops = (2*input.cpuA->nnz*input.dim)/(1e+9);
    perfRes.cuspgemm_throughput = gflops/t;
    //std::cout<<"cuSpgemm Throughput: "<<gflops/t<<" gflops/s "<<std::endl;
    float gb = (float)((input.n+1 + 2*input.cpuA->nnz + 2*input.n*input.dim)*4)/(1e+9);
    perfRes.cuspgemm_bandwidth = gb/t;
    //std::cout<<"cuSpgemm Bandwidth: "<<gb/t<<" GB/s "<<std::endl;
    
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    CHECK_CUDA( cudaFree(dBuffer) )
}
