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
void flexspmm_cuda_wo_pre_v4(int* tileRowPtr,
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
        tile_nxtR_id = tileRowPtr[row_idx+1]; 

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
        //std::cout<<"Verify result accuracy ("<< to_string(tm) << "X" << to_string(tn) << ") ... " <<std::endl; 
        for (int i=0; i<m; ++i){
            for (int j=0; j<n; ++j){
                if (abs(h_gold[i*n+j]-h_res[i*n+j])>=0.01){
                    count++;
                    //if (j==0) 
                    //    std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        perfRes.flex_spgemm_errors.push_back(count);
        if (count>0)
            std::cout<<"Kernel ("<< to_string(tm) << "X" << to_string(tn) << ") errs: " << count<<std::endl;
        memset(h_res, 0, n*m*sizeof(float));
}
void run(DataLoader& input){

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
    NPerf_metric_collect("sm__sass_inst_executed_op_st.sum");
    NPerf_metric_collect
        ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed");
    NPerf_metric_collect
        ("l1tex__m_l1tex2xbar_throughput.avg.pct_of_peak_sustained_elapsed");
    NPerf_metric_collect("dram__bytes.sum");
    
    const int l2_size_chars = info.cuda_prop.l2CacheSize;
    const int a_size = l2_size_chars / sizeof(float) * 2;
    
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
    struct App_Kernel_Info {
         App_Kernel_Info  
         (Kernel_Info& k,const char *name, int i, int nbx, int nby, int nt):
           k_ptr(k.func_ptr),name_base(name),
           shape_idx{i},
           n_threads{nt},n_blocks_x{nbx},n_blocks_y{nby}{}
        GPU_Info_Func k_ptr;
        const char *name_base;
        const int shape_idx;
        const int n_blocks_x, n_blocks_y, n_threads;
    };  
    vector<App_Kernel_Info> kernels;
    vector<mat> spMats;
    
    #define EXAMINE_KERNEL(k,sidx,nbx,nby,nt) \
    {const int idx = kernels.size(); \
        kernels.emplace_back(info.GET_INFO((k)),#k,sidx,nbx,nby,nt); }

    #define SPECIFY_KERNEL(k,sidx,nbx,nby,nt)\
    {const int idx = kernels.size(); \
        spMats.emplace_back(mat(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz, tileConfs[sidx].tm,tileConfs[sidx].tn)); \
        EXAMINE_KERNEL((k<tileConfs[sidx].tm,tileConfs[sidx].tn,4>), sidx, nbx, nby, nt); }
// NBX,NBY,NT are useless currently
#define NBX 1
#define NBY 1
#define NT 1
#ifdef CUBE4X4
    {        
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 0, NBX, NBY, NT);
    }
#endif
#ifdef RECT8X4
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 1, NBX, NBY, NT);
    }
#endif
#ifdef RECT16X4
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 2, NBX, NBY, NT);
    }
#endif
#ifdef RECT32X4
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 3, NBX, NBY, NT);
    }
#endif
#ifdef RECT64X4
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 4, NBX, NBY, NT);
    }
#endif
#ifdef RECT128X4
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 5, NBX, NBY, NT);
    }
#endif
#ifdef RECT256X4
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 6, NBX, NBY, NT);
    }
#endif
#ifdef RECT4X8
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 7, NBX, NBY, NT);
    }
#endif
#ifdef CUBE8X8
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 8, NBX, NBY, NT);
    }
#endif
#ifdef RECT16X8
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 9, NBX, NBY, NT);
    }
#endif
#ifdef RECT32X8
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 10, NBX, NBY, NT);
    }
#endif
#ifdef RECT64X8
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 11, NBX, NBY, NT);
    }
#endif
#ifdef RECT128X8
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 12, NBX, NBY, NT);
    }
#endif
#ifdef RECT256X8
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 13, NBX, NBY, NT);
    }
#endif
#ifdef RECT4X16
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 14, NBX, NBY, NT);
    }
#endif
#ifdef RECT8X16
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 15, NBX, NBY, NT);
    }
#endif
#ifdef CUBE16X16
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 16, NBX, NBY, NT);
    }
#endif
#ifdef RECT32X16
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 17, NBX, NBY, NT);
    }
#endif
#ifdef RECT64X16
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 18, NBX, NBY, NT);
    }
#endif
#ifdef RECT128X16
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 19, NBX, NBY, NT);
    }
#endif
#ifdef RECT256X16
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 20, NBX, NBY, NT);
    }
#endif
#ifdef RECT4X32
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 21, NBX, NBY, NT);
    }
#endif
#ifdef RECT8X32
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 22, NBX, NBY, NT);
    }
#endif
#ifdef RECT16X32
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 23, NBX, NBY, NT);
    }
#endif
#ifdef CUBE32X32
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 24, NBX, NBY, NT);
    }
#endif
#ifdef RECT64X32
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 25, NBX, NBY, NT);
    }
#endif
#ifdef RECT128X32
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 26, NBX, NBY, NT);
    }
#endif
#ifdef RECT256X32
    {
        SPECIFY_KERNEL(flexspmm_cuda_wo_pre_v4, 27, NBX, NBY, NT);
    }
#endif

    pTable table(stdout);
    for( int id=0; id<kernels.size(); ++id){
        //std::string name(to_string(spMats[id].tm)+"X"+to_string(spMats[id].tn));
        //table.entry("Tiling", "%s", name.c_str());
        //table.entry("wp", "%2d", id);
        
        spMats[id].csr2tile();
        // allocate device memory
        // index of the first nz entry in each tile, length = #tiles+1
        int* d_tileNnz; 
        CHECK_CUDA(cudaMalloc(&d_tileNnz, spMats[id].nnzPtr.size()*sizeof(int)));
        
#ifdef V3_KERNEL
        // index of the first tile for each thread block, length = #blocks+1
        int* d_block_tileStart_idx; 
        CHECK_CUDA(cudaMalloc(&d_block_tileStart_idx, spMats[id].block_tileStart_idx.size()*sizeof(int)));
        
        // row index of tiles for each thread block, length = #blocks
        int* d_warp_tileRow_idx; 
        CHECK_CUDA(cudaMalloc(&d_warp_tileRow_idx, spMats[id].warp_tileRow_idx.size()*sizeof(int)));
        
        // row&col index of vals in sparse matrix, length = nnz
        char* d_r_c_Offset; 
        CHECK_CUDA(cudaMalloc(&d_r_c_Offset, spMats[id].rc_Offset.size()*sizeof(char)));
#endif
        // column index of tiles, length = #tiles
        int* d_tileColIdx; 
        CHECK_CUDA(cudaMalloc(&d_tileColIdx, spMats[id].tileLeftColIdx.size()*sizeof(int)));
          
        // non-zero vals of sparse matrix, length = nnz
        float* d_vals; 
        CHECK_CUDA(cudaMalloc(&d_vals, spMats[id].newVals.size()*sizeof(int)));
       

        // v4 kernel
        int* d_tileRowPtr; 
        CHECK_CUDA(cudaMalloc(&d_tileRowPtr, spMats[id].tileRowPtr.size()*sizeof(int)));
        //std::cout<<"@536: metaTile.size() = "<<data.metaTile.size()<<std::endl;
        int* d_nnzTile; 
        CHECK_CUDA(cudaMalloc(&d_nnzTile, spMats[id].nnzTile.size()*sizeof(int)));
        int* d_bitMap; 
        CHECK_CUDA(cudaMalloc(&d_bitMap, spMats[id].bitMap.size()*sizeof(int)));
        int* d_rcOffset; 
        CHECK_CUDA(cudaMalloc(&d_rcOffset, spMats[id].rcOffset.size()*sizeof(int)));
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
        CHECK_CUDA(cudaMalloc(&d_mat_b, spMats[id].m*spMats[id].k*sizeof(float)));
        
        // Matrix C
        float* d_mat_c; 
        CHECK_CUDA(cudaMalloc(&d_mat_c, spMats[id].m*spMats[id].k*sizeof(float)));
        cudaMemset(d_mat_c, 0.0, spMats[id].m*spMats[id].k*sizeof(float));
        cudaDeviceSynchronize(); 
        
        
        // transfer data to device
        cudaMemcpy(d_tileNnz, spMats[id].nnzPtr.data(), spMats[id].nnzPtr.size()*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tileColIdx, spMats[id].tileLeftColIdx.data(), spMats[id].tileLeftColIdx.size()*sizeof(int), cudaMemcpyHostToDevice);
#ifdef V3_KERNEL
        cudaMemcpy(d_block_tileStart_idx, spMats[id].block_tileStart_idx.data(), spMats[id].block_tileStart_idx.size()*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_warp_tileRow_idx, spMats[id].warp_tileRow_idx.data(), spMats[id].warp_tileRow_idx.size()*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_r_c_Offset, spMats[id].rc_Offset.data(), spMats[id].rc_Offset.size()*sizeof(char), cudaMemcpyHostToDevice);
#endif
        cudaMemcpy(d_vals, spMats[id].newVals.data(), spMats[id].newVals.size()*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mat_b, host_mat_b, spMats[id].m*spMats[id].k*sizeof(float), cudaMemcpyHostToDevice);
        //cudaMemcpy(d_mat_c, mat_c, data.m*data.k*sizeof(float), cudaMemcpyHostToDevice);

        // v4 kernel
        cudaMemcpy(d_tileRowPtr, spMats[id].tileRowPtr.data(), spMats[id].tileRowPtr.size()*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nnzTile, spMats[id].nnzTile.data(), spMats[id].nnzTile.size()*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bitMap, spMats[id].bitMap.data(), spMats[id].bitMap.size()*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rcOffset, spMats[id].rcOffset.data(), spMats[id].rcOffset.size()*sizeof(int), cudaMemcpyHostToDevice);
        

        // each thread block has 2 warps
        //dim3 grid(data.block_tileStart_idx.size()-1, (data.k+31)/32);
        //printf("@415:   data.block_tileStart_idx.size() = %d\n",data.block_tileStart_idx.size());
        //printf("@416:   data.k = %d\n",data.k);
        //LOG(INFO) << "Ahead the kernel ...";
        //printf("%d of %s, Ahead the kernel ...\n",__LINE__,__FILE__);
        //std::cout<<"block_tileStart_idx:"<<std::endl;
        //print(block_tileStart_idx);
        //std::cout<<"warp_tileRow_idx:"<<std::endl;
        //print(warp_tileRow_idx);
       
        
        constexpr int wp_sz = 32;
        // Get measured number of instructions executed.
        //
        const double n_insn =
            NPerf_metric_value_get("sm__inst_executed.sum") * wp_sz;
        pTable_Row row(table);
        // Compute the expected number of multiply/add instructions.
        //
        const int64_t n_madd = spMats[id].newVals.size()*spMats[id].k;
        const double n_madd_p_wp = double(n_madd) / wp_sz;
                
        int threads = 128;

        Kernel_Info* const ki = &info.get_info(kernels[id].k_ptr);
        typedef void (*KPtr)(int* tileRowPtr,
                            int* nnzPtr,
                            int* nnz,
                            int* bitMap,
                            int* tileLeftCol,
				            int* rcOffset,
				            float* vals,
				            int spH,
				            float* mat_b,
				            int k,
                            float* mat_c);
        
        float spgemm_duration;
        float elap_t = 0; 
        cudaEvent_t spgemm_start, spgemm_stop;
        cudaEventCreate(&spgemm_start);
        cudaEventCreate(&spgemm_stop);
        cudaEventRecord(spgemm_start);
        for ( NPerf_data_reset(); NPerf_need_run_get(); ){
            int gridx = (spMats[id].m+spMats[id].tm-1)/spMats[id].tm;
            KPtr(ki->func_ptr)<<<gridx,threads>>>(d_tileRowPtr, 
                               d_tileNnz,
                               d_nnzTile,
                               d_bitMap,
                               d_tileColIdx,
                               d_rcOffset,
                               d_vals,
                               spMats[id].m,
                               d_mat_b,
                               spMats[id].k,
                               d_mat_c);
        }
        cudaEventRecord(spgemm_stop);
        cudaEventSynchronize(spgemm_stop);
        cudaEventElapsedTime(&spgemm_duration, spgemm_start, spgemm_stop);
        elap_t += spgemm_duration; 
        cudaDeviceSynchronize(); 
        
        
          // Get and print elapsed time.
          //
          const double et_seconds = NPerf_kernel_et_get();
          table.entry( "t/µs", "%4.0f", et_seconds * 1e6 );

          // Write a heading that will span multiple columns.
          //
          table.header_span_start("Per Mult");

          table.header_span_start("Num Insns");

          table.entry
            ( "Load", "%4.1f",
              NPerf_metric_value_get("sm__sass_inst_executed_op_ld.sum")
              / n_madd_p_wp );
          table.entry
            ( "Store", "%5.2f",
              NPerf_metric_value_get("sm__sass_inst_executed_op_st.sum")
              / n_madd_p_wp );
          table.entry
            ( "All", "%6.2f",
              NPerf_metric_value_get("sm__inst_executed.sum")
              / n_madd_p_wp );

          table.header_span_end();

          // Write an extra header line over the next entry.
          table.header_span("Time",1);
          table.entry
            ( "Cyc", "%6.1f",
              NPerf_metric_value_get("sm__cycles_elapsed.max") * fp32_per_chip
              / n_madd );

          table.header_span("L1←L2",1);
          table.entry
            ( "Bytes", "%4.2f",
              NPerf_metric_value_get("l1tex__m_xbar2l1tex_read_bytes.sum")
              / a_size );

          table.header_span_end();

          table.header_span_start("Entire GPU");

          table.header_span_start("L1 ⇆ L2");
          table.entry
            ( "GB/s", "%6.1f",
              ( NPerf_metric_value_get("l1tex__m_l1tex2xbar_write_bytes.sum")
                + NPerf_metric_value_get("l1tex__m_xbar2l1tex_read_bytes.sum") )
              / et_seconds * 1e-9 );

          table.entry
            ("% Pk", "%5.2f",
             NPerf_metric_value_get
             ("l1tex__m_l1tex2xbar_throughput"
              ".avg.pct_of_peak_sustained_elapsed"));

          table.header_span_end();

          table.header_span_start("L2 ⇆ DRAM");
          table.entry
            ( "GB/s", "%7.1f",
              NPerf_metric_value_get("dram__bytes.sum") / et_seconds * 1e-9 );

          table.entry
            ("% Pk", "%5.1f",
             NPerf_metric_value_get
             ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"));

          table.header_span_end();

          table.header_span( "FP Thpt", 1);
          table.entry( "GFLOP/s", "%9.1f", 1e-9 * n_madd / et_seconds );

          table.header_span_end();
        
        //LOG(INFO) << "After the kernel ...";
        //printf("%d of %s, After the kernel ...\n",__LINE__,__FILE__);

        // transfer data to host
        //LOG(INFO) << "Transfer results back ...";
        //printf("%d of %s, Transfer results back ...\n",__LINE__,__FILE__);
        cudaMemcpy(h_res_c, d_mat_c, spMats[id].m*spMats[id].k*sizeof(float), cudaMemcpyDeviceToHost);
        resCheck(h_ref_c, h_res_c, spMats[id].n, spMats[id].k, perfRes, spMats[id].tm, spMats[id].tn);
        
        float t = elap_t*(1e-3);
        perfRes.flex_spgemm_time.push_back(t);
        
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

    free(h_res_c);
    free(h_ref_c);
    free(host_mat_b);

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
