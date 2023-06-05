#include "flex_spmm.cuh"
__device__ __forceinline__
uint32_t smem_u32addr(const void *smem_ptr) {
    uint32_t addr;
    asm ("{.reg .u64 u64addr;\n"
         " cvta.to.shared.u64 u64addr, %1;\n"
         " cvt.u32.u64 %0, u64addr;}\n"
         : "=r"(addr)
         : "l"(smem_ptr)
    );
    return addr;
}
__device__ __forceinline__
void stg32(const float &reg, void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p st.global.f32 [%0], %1;}\n"
        : : "l"(ptr), "f"(reg), "r"((int)guard)
    );
}
__device__ __forceinline__
void sts8(const uint8_t &reg, const uint32_t &addr) {
    asm volatile (
        "st.shared.u8 [%0], %1;\n"
        : : "r"(addr), "r"((int)reg)
    );
}
__device__ __forceinline__
void lds8(uint8_t &reg, const uint32_t &addr) {
    int temp = reg;
    asm volatile (
        "ld.shared.u8 %0, [%1];\n"
        : "=r"(temp)
        : "r"(addr)
    );
    reg = temp;
}
__device__ __forceinline__
void sts32(const float &reg, const uint32_t &addr) {
    asm volatile (
        "st.shared.f32 [%0], %1;\n"
        : : "r"(addr), "f"(reg)
    );
}
__device__ __forceinline__
void lds32(float &reg, const uint32_t &addr) {
    asm volatile (
        "ld.shared.f32 %0, [%1];\n"
        : "=f"(reg)
        : "r"(addr)
    );
}

// inputs:
//		 block_tileStart_idx: the tile idex of the first tile computed by thread-blocks
//       tileColIdx: the column idex of the first column of each tile
//       tileNnz: the index of the first nze(non-zero entry) in each tile
//       warp_tileRow_idx: row idx of each tile
//       tiles: # of tiles
//       r_c_Offset(8 bits): the row index (upper 4 bits) + column index (lower 4 bits) of each nze in each tile
//       vals: non zero entries
// The real row-col index for a nze (the i-th row tile, the j-th tile of the i-th row tile): 
//  r = i + r_c_Offset[7...4], c = tileColIdx[ tileRowPtr[i] + j ] + r_c_Offset[3...0]

// A: sparse, m * n
// B: dense, n * k   (k << n)
__global__
void flexspgemm_cuda_reg_pre_v1(int* tileNnz,
				int* block_tileStart_idx,
				int* warp_tileRow_idx,
                int* tileColIdx,
				int tiles,
				char* r_c_Offset,
				float* vals,
				int k,
				float* mat_b,
                float* mat_c){
	const uint32_t WARPSZ = 32;
	const uint32_t lane_id = threadIdx.x % WARPSZ;
	const uint32_t warp_id = threadIdx.x / WARPSZ;
	const uint32_t warps = (blockDim.x + WARPSZ - 1)/WARPSZ;
	
	int tile_ref[3] = {-1};
	if (blockIdx.x | 0){
		tile_ref[0] = block_tileStart_idx[blockIdx.x-1];
	}
	tile_ref[1] = block_tileStart_idx[blockIdx.x];
	tile_ref[2] = block_tileStart_idx[blockIdx.x+1];
	const uint32_t warp_tileStart_id = tile_ref[1] + warp_id;
	
	// matrix A: 
	//          non-zero vals: tm*tn * 4bytes * double buffer <= 16*16 * 4 * 2 = 2k
	//          non-zero entry idx: tm*tn * 1bytes * double buffer <= 16*16 * 1 * 2 = 0.5k
	// matrix B:
	//          each thread has 16 regs to store one B column segment
	// Only A is stored in shared memory
	/*  
		| ----- vals ----- | -- idx -- | ----- vals ----- | -- idx -- |
		|       1k         |   256B    |       1k         |   256B    |     
		a_vals_smem ^= 0x0500
		a_idx_smem  ^= 0x0500
	*/
	// 2 warps in a block
	#define WARPS 2
	#define ACC_SH  true
	#if ACC_SH
	//__shared__ char smem[(8*16*16 + 512)*WARPS+TM*32*4+4];
	__shared__ char smem[(8*16*16 + 1024)*WARPS+32*32*4+4];
	#else
	__shared__ char smem[(8*16*16 + 512)*WARPS+4];
	#endif
	//float* a_vals_smem = reinterpret_cast<float *>(smem+warp_id*2560);                 // 2.5k
	float* a_vals_smem = reinterpret_cast<float *>(smem+warp_id*3072);                 // 2.5k
	//uint8_t* a_idx_smem = reinterpret_cast<uint8_t *>(smem+warp_id*2560 + 1024);
	uint8_t* a_idx_smem = reinterpret_cast<uint8_t *>(smem+warp_id*3072 + 1024);
	//float* c_mat_sh = reinterpret_cast<float *>(smem+WARPS*2560);                     // two warps: 2 * 2.5k = 5k
	float* c_mat_sh = reinterpret_cast<float *>(smem+WARPS*3072);                     // two warps: 2 * 2.5k = 5k

	#if ACC_SH
    //uint32_t* mark_c_rows = reinterpret_cast<uint32_t *>(smem+(8*16*16 + 512)*WARPS+TM*32*4);
    uint32_t* mark_c_rows = reinterpret_cast<uint32_t *>(smem+(8*16*16 + 1024)*WARPS+32*32*4);
    //for (uint32_t i=warp_id; i<TM; i+=warps){
    for (uint32_t i=warp_id; i<32; i+=warps){
        c_mat_sh[i*32+lane_id] = 0;
    }
	#else
    uint32_t* mark_c_rows = reinterpret_cast<uint32_t *>(smem+(8*16*16 + 512)*WARPS);
    #endif
    if (threadIdx.x==0) mark_c_rows[0] = 0;

	uint32_t a_vals_sts = smem_u32addr(a_vals_smem);
	uint32_t a_vals_lds = smem_u32addr(a_vals_smem);
	uint32_t a_idx_sts = smem_u32addr(a_idx_smem);
	uint32_t a_idx_lds = smem_u32addr(a_idx_smem);

	// ************ load A tile ( && B rows required by first tile) from glb mem to shared memory (registers) *********************************************
	// row_flag[0]: bits represent existance of B rows, determined by col idx of A entrys 
	// row_flag[1]: bits represent existance of C rows, determined by row idx of A entrys 
	uint32_t row_flag[2] = {0,0};
	
  	float b_reg[2][32];
	// both tileNNz and r_c_offfset have good locality, so no need to optimize their memory access behavior? 
	#define FULL_MASK 0xffffffff
	uint32_t steps = 1;
    //printf("kernel begins ...\n");	
	for (uint32_t entry_idx = tileNnz[warp_tileStart_id]; entry_idx<tileNnz[warp_tileStart_id+1]; entry_idx += steps){
		
		if (tileNnz[warp_tileStart_id+1]-entry_idx>=32){         // if more than 32 non-zero entrys left in current tile
			int rc_idx = r_c_Offset[entry_idx+lane_id]; // in coalesced way
			uint32_t r_offset = (rc_idx & 240)>>4;     // .. & 1111 0000
			uint32_t c_offset = rc_idx & 15;      // .. & 0000 1111
			// load a to shared mem
			uint8_t aVal_idx_tmp = r_c_Offset[entry_idx+lane_id];
			sts8(aVal_idx_tmp, a_idx_sts + (entry_idx+lane_id-tileNnz[warp_tileStart_id])*sizeof(uint8_t));
            float aVal_tmp = vals[entry_idx+lane_id];
			sts32(aVal_tmp, a_vals_sts + (r_offset*16+c_offset)*sizeof(float));

			// load b to registers
			for (uint32_t j=0; j<32; ++j){
				int rc_idx_tmp = __shfl_sync(FULL_MASK, rc_idx, j);
				//char rc_idx_tmp = a_idx_sts[entry_idx+j-tileNnz[warp_tileStart_id]];

				//r_offset = rc_idx_tmp & 240;     // .. & 1111 0000
				c_offset = rc_idx_tmp & 15;      // .. & 0000 1111
				// the i-th bit 1 represents the i-th B row is alrealdy loaded in shared memory
				if ((row_flag[0] & (1<<c_offset)) == 0 ){
					// mark it as loaded
					row_flag[0] |= (1<<c_offset);

					uint32_t entry_col_idx = tileColIdx[warp_tileStart_id] + c_offset;  // .. & 0000 1111
					uint32_t lane_offset = blockIdx.y*32 + lane_id;
					if (lane_offset<min(blockIdx.y*32+32, k)){
						// matrix B is in row major
						// b_mat_sh[warp_id*2 + warp_tileStart_id%2][c_offset*tn+lane_id] = mat_b[entry_col_idx*k + lane_offset];
						b_reg[0][c_offset] = mat_b[entry_col_idx*k + lane_offset];
					}
				}
			}
			steps = 32;
		}else if (tileNnz[warp_tileStart_id+1]-entry_idx>=16){
			int rc_idx = r_c_Offset[entry_idx+lane_id]; // in coalesced way
			uint32_t r_offset = (rc_idx & 240)>>4;     // .. & 1111 0000
			uint32_t c_offset = rc_idx & 15;      // .. & 0000 1111
			//uint32_t mask = __ballot_sync(FULL_MASK, entry_idx<tileNnz[warp_tileStart_id+1]);
			//uint32_t act_thds = __popc(mask);

			// load a to shared mem
			uint8_t aVal_idx_tmp = r_c_Offset[entry_idx+lane_id];
            float aVal_tmp = vals[entry_idx+lane_id];
            if (lane_id<16){
			    sts8(aVal_idx_tmp, a_idx_sts + (entry_idx+lane_id-tileNnz[warp_tileStart_id])*sizeof(uint8_t));
			    sts32(aVal_tmp, a_vals_sts + (r_offset*16+c_offset)*sizeof(float));
            }

			// load b to registers
			for (uint32_t j=0; j<16; ++j){
				int rc_idx_tmp = __shfl_sync(FULL_MASK, rc_idx, j); 
				//char rc_idx_tmp = a_idx_sts[entry_idx+j-tileNnz[warp_tileStart_id]];
				//r_offset = rc_idx_tmp & 240;     // .. & 1111 0000
				c_offset = rc_idx_tmp & 15;      // .. & 0000 1111
				// the i-th bit represents the i-th B row is alrealdy loaded in shared memory
				if ((row_flag[0] & (1<<c_offset)) == 0 ){
					// mark it as loaded
					row_flag[0] |= (1<<c_offset);

					uint32_t entry_col_idx = tileColIdx[warp_tileStart_id] + c_offset;  // .. & 0000 1111
					uint32_t lane_offset = blockIdx.y*32 + lane_id;
					if (lane_offset<min(blockIdx.y*32+32, k)){
						// matrix B is in row major
						// b_mat_sh[warp_id*2 + warp_tileStart_id%2][c_offset*tn+lane_id] = mat_b[entry_col_idx*k + lane_offset];
						b_reg[0][c_offset] = mat_b[entry_col_idx*k + lane_offset];
					}
				}
			}
			steps = 16;
		}else if (tileNnz[warp_tileStart_id+1]-entry_idx>=8){
			int rc_idx = r_c_Offset[entry_idx+lane_id]; // in coalesced way
			uint32_t r_offset = (rc_idx & 240)>>4;     // .. & 1111 0000
			uint32_t c_offset = rc_idx & 15;      // .. & 0000 1111
			//uint32_t mask = __ballot_sync(FULL_MASK, entry_idx<tileNnz[warp_tileStart_id+1]);
			//uint32_t act_thds = __popc(mask);

			// load a to shared mem
			uint8_t aVal_idx_tmp = r_c_Offset[entry_idx+lane_id];
            float aVal_tmp = vals[entry_idx+lane_id];
            if (blockIdx.x==0 && warp_id==0 && lane_id==1 && r_offset==8){
                printf("@228: r_offset = %d, c_offset = %d, aVal_tmp = %f\n", r_offset,c_offset,aVal_tmp);
            }
            if (lane_id<8){
			    sts8(aVal_idx_tmp, a_idx_sts + (entry_idx+lane_id-tileNnz[warp_tileStart_id])*sizeof(uint8_t));
			    sts32(aVal_tmp, a_vals_sts + (r_offset*16+c_offset)*sizeof(float));
            }
            // load b to registers
			for (uint32_t j=0; j<8; ++j){
				int rc_idx_tmp = __shfl_sync(FULL_MASK, rc_idx, j); 
				//char rc_idx_tmp = a_idx_sts[entry_idx+j-tileNnz[warp_tileStart_id]];
				//r_offset = rc_idx_tmp & 240;     // .. & 1111 0000
				c_offset = rc_idx_tmp & 15;      // .. & 0000 1111
				// the i-th bit represents the i-th B row is alrealdy loaded in shared memory
				if ((row_flag[0] & (1<<c_offset)) == 0 ){
					// mark it as loaded
					row_flag[0] |= (1<<c_offset);

					uint32_t entry_col_idx = tileColIdx[warp_tileStart_id] + c_offset;  // .. & 0000 1111
					uint32_t lane_offset = blockIdx.y*32 + lane_id;
					if (lane_offset<min(blockIdx.y*32+32, k)){
						// matrix B is in row major
						// b_mat_sh[warp_id*2 + warp_tileStart_id%2][c_offset*tn+lane_id] = mat_b[entry_col_idx*k + lane_offset];
						b_reg[0][c_offset] = mat_b[entry_col_idx*k + lane_offset];
					}
				}
			}
			steps = 8;
		}else{
			int rc_idx = r_c_Offset[entry_idx];   // in coalesced way
			uint32_t r_offset = (rc_idx & 240)>>4;     // .. & 1111 0000
			uint32_t c_offset = rc_idx & 15;      // .. & 0000 1111

			// load a to shared mem
			uint8_t aVal_idx_tmp = r_c_Offset[entry_idx];
			sts8(aVal_idx_tmp, a_idx_sts + (entry_idx-tileNnz[warp_tileStart_id])*sizeof(uint8_t));
            float aVal_tmp = vals[entry_idx];
			sts32(aVal_tmp, a_vals_sts + (r_offset*16+c_offset)*sizeof(float));

			// the i-th bit represents the i-th B row is alrealdy loaded in shared memory
			// load b to registers
			if ((row_flag[0] & (1<<c_offset)) == 0 ){
				// mark it as loaded
				row_flag[0] |= (1<<c_offset);

				uint32_t entry_col_idx = tileColIdx[warp_tileStart_id] + c_offset;  // .. & 0000 1111
				uint32_t lane_offset = blockIdx.y*32 + lane_id;
				if (lane_offset<min(blockIdx.y*32+32, k)){
					// matrix B is in row major
					// b_mat_sh[warp_id*2 + warp_tileStart_id%2][c_offset*tn+lane_id] = mat_b[entry_col_idx*k + lane_offset];
					b_reg[0][c_offset] = mat_b[entry_col_idx*k + lane_offset];
                    //if (blockIdx.x==1 && warp_id==1 && lane_id==0){
                        //printf("@262:   b_col = %d, b_reg[%d][%d] = %f\n",entry_col_idx,warp_tileStart_id%2,c_offset,b_reg[warp_tileStart_id%2][c_offset]);
                    //}
				}
			}
			steps = 1;
		}
	}
	row_flag[0] = 0;
	//a_vals_sts ^= 0x0500;
	//a_idx_sts  ^= 0x0500;
	a_vals_sts ^= 0x0600;
	a_idx_sts  ^= 0x0600;
	// ***************************************************************************************************************************************************

    if (blockIdx.x==0 && warp_id==0 && lane_id==0){
		float temp_a = 3.0;
        lds32(temp_a, a_vals_lds + (8*16+5)*sizeof(float));
        printf("@296: temp_a = %f\n", temp_a);
    }
	// multiplication loops
	
	float res[32];
    for (int i=0; i<32; ++i) res[i] = 0;
	
	// iterate tiles assigned to the current block
	for (uint32_t i=warp_tileStart_id; i<tile_ref[2]; i += warps){
	    int  nnz_cur_tile = tileNnz[i+1]-tileNnz[i];
        //if (blockIdx.x==11 && warp_id==0 && lane_id==0){
            //printf("tileref[0] = %d,tileref[0] = %d,tileref[0] = %d\n",tile_ref[0],tile_ref[1],tile_ref[2]);
            //printf("tileID = %d, nnz = %d\n",i,nnz_cur_tile);
        //}
            if (blockIdx.x==0 && warp_id==0 && lane_id==0 && i==0){
		        float temp_a = 3.0;
                lds32(temp_a, a_vals_lds + (8*16+5)*sizeof(float));
                printf("@313: temp_a = %f\n", temp_a);
            }
		uint32_t b_double_buf_idx = (i-warp_tileStart_id)/warps%2;
		// ************ load B rows required by "next tile" from glb mem to shmem **********
		// both tileNNz and r_c_offfset have good locality, so no need to optimize their memory access behavior?
		if (i+warps<tile_ref[2]){
			//uint32_t boundry = tileNnz[i+warps+1]; 
			for (uint32_t entry_idx = tileNnz[i+warps]; entry_idx<tileNnz[i+warps+1]; entry_idx += steps){
                
                if (blockIdx.x==0 && warp_id==0 && lane_id==0 && i==0){
		            float temp_a = 3.0;
                    lds32(temp_a, a_vals_lds + (8*16+5)*sizeof(float));
                    printf("@325: entry_idx = %d       temp_a = %f\n", entry_idx, temp_a);
                }
			    	
				if (tileNnz[i+warps+1]-entry_idx>=32){
					int rc_idx = r_c_Offset[entry_idx+lane_id]; // in coalesced way
					uint32_t r_offset = (rc_idx & 240)>>4;     // .. & 1111 0000
					uint32_t c_offset = rc_idx & 15;      // .. & 0000 1111

					// load a to shared mem
			        uint8_t aVal_idx_tmp = r_c_Offset[entry_idx+lane_id];
			        sts8(aVal_idx_tmp, a_idx_sts + (entry_idx+lane_id-tileNnz[i+warps])*sizeof(uint8_t));
                    float aVal_tmp = vals[entry_idx+lane_id];
			        sts32(aVal_tmp, a_vals_sts + (r_offset*16+c_offset)*sizeof(float));

					// load b to registers
					for (uint32_t j=0; j<32; ++j){
						int rc_idx_tmp = __shfl_sync(FULL_MASK, rc_idx, j); 
						//r_offset = rc_idx_tmp & 240;     // .. & 1111 0000
						c_offset = rc_idx_tmp & 15;      // .. & 0000 1111
						// the i-th bit represents the i-th B row is alrealdy loaded in shared memory
						if ((row_flag[0] & (1<<c_offset)) == 0 ){
							// mark it as loaded
							row_flag[0] |= (1<<c_offset);

							uint32_t entry_col_idx = tileColIdx[i+warps] + c_offset;  // .. & 0000 1111
							uint32_t lane_offset = blockIdx.y*32 + lane_id;
							if (lane_offset<min(blockIdx.y*32+32, k)){
								// matrix B is in row major
								// b_mat_sh[warp_id*2 + warp_tileStart_id%2][c_offset*tn+lane_id] = mat_b[entry_col_idx*k + lane_offset];
								b_reg[(b_double_buf_idx+1)%2][c_offset] = mat_b[entry_col_idx*k + lane_offset];
							}
						}
					}
					steps = 32;
				}else if (tileNnz[i+warps+1]-entry_idx>=16){
					int rc_idx = r_c_Offset[entry_idx+lane_id]; // in coalesced way
					uint32_t r_offset = (rc_idx & 240)>>4;     // .. & 1111 0000
					uint32_t c_offset = rc_idx & 15;      // .. & 0000 1111

					// load a to shared mem
			        uint8_t aVal_idx_tmp = r_c_Offset[entry_idx+lane_id];
                    float aVal_tmp = vals[entry_idx+lane_id];
			        if (lane_id<16){
                        sts8(aVal_idx_tmp, a_idx_sts + (entry_idx+lane_id-tileNnz[i+warps])*sizeof(uint8_t));
			            sts32(aVal_tmp, a_vals_sts + (r_offset*16+c_offset)*sizeof(float));
                    }
					// load b to registers
					for (uint32_t j=0; j<16; ++j){
						int rc_idx_tmp = __shfl_sync(FULL_MASK, rc_idx, j); 
						// r_offset = rc_idx_tmp & 240;     // .. & 1111 0000
						c_offset = rc_idx_tmp & 15;      // .. & 0000 1111
						// the i-th bit represents the i-th B row is alrealdy loaded in shared memory
						if ((row_flag[0] & (1<<c_offset)) == 0 ){
							// mark it as loaded
							row_flag[0] |= (1<<c_offset);

							uint32_t entry_col_idx = tileColIdx[i+warps] + c_offset;  // .. & 0000 1111
							uint32_t lane_offset = blockIdx.y*32 + lane_id;
							if (lane_offset<min(blockIdx.y*32+32, k)){
								// matrix B is in row major
								// b_mat_sh[warp_id*2 + warp_tileStart_id%2][c_offset*tn+lane_id] = mat_b[entry_col_idx*k + lane_offset];
								b_reg[(b_double_buf_idx+1)%2][c_offset] = mat_b[entry_col_idx*k + lane_offset];
							}
						}
					}
					steps = 16;
				}else if (tileNnz[i+warps+1]-entry_idx>=8){
					int rc_idx = r_c_Offset[entry_idx+lane_id]; // in coalesced way
					uint32_t r_offset = (rc_idx & 240)>>4;     // .. & 1111 0000
					uint32_t c_offset = rc_idx & 15;      // .. & 0000 1111

					// load a to shared mem
			        uint8_t aVal_idx_tmp = r_c_Offset[entry_idx+lane_id];
                    float aVal_tmp = vals[entry_idx+lane_id];
                    if (lane_id<8){
                if (blockIdx.x==0 && warp_id==0 && lane_id==0 && i==0){
		            float temp_a = 3.0;
                    lds32(temp_a, a_vals_lds + (8*16+5)*sizeof(float));
                    printf("@403: entry_idx = %d       temp_a = %f\n", entry_idx, temp_a);
                }
			            sts8(aVal_idx_tmp, a_idx_sts + (entry_idx+lane_id-tileNnz[i+warps])*sizeof(uint8_t));
                if (blockIdx.x==0 && warp_id==0 && lane_id==0 && i==0){
		            float temp_a = 3.0;
                    lds32(temp_a, a_vals_lds + (8*16+5)*sizeof(float));
                    printf("@409: entry_idx = %d       temp_a = %f\n", entry_idx, temp_a);
                }
			            sts32(aVal_tmp, a_vals_sts + (r_offset*16+c_offset)*sizeof(float));
                    }
                if (blockIdx.x==0 && warp_id==0 && lane_id==0 && i==0){
		            float temp_a = 3.0;
                    lds32(temp_a, a_vals_lds + (8*16+5)*sizeof(float));
                    printf("@416: entry_idx = %d       temp_a = %f\n", entry_idx, temp_a);
                }
					// load b to registers
					for (uint32_t j=0; j<8; ++j){
						int rc_idx_tmp = __shfl_sync(FULL_MASK, rc_idx, j); 
						// r_offset = rc_idx_tmp & 240;     // .. & 1111 0000
						c_offset = rc_idx_tmp & 15;      // .. & 0000 1111
						// the i-th bit represents the i-th B row is alrealdy loaded in shared memory
						if ((row_flag[0] & (1<<c_offset)) == 0 ){
							// mark it as loaded
							row_flag[0] |= (1<<c_offset);

							uint32_t entry_col_idx = tileColIdx[i+warps] + c_offset;  // .. & 0000 1111
							uint32_t lane_offset = blockIdx.y*32 + lane_id;
							if (lane_offset<min(blockIdx.y*32+32, k)){
								// matrix B is in row major
								// b_mat_sh[warp_id*2 + warp_tileStart_id%2][c_offset*tn+lane_id] = mat_b[entry_col_idx*k + lane_offset];
								b_reg[(b_double_buf_idx+1)%2][c_offset] = mat_b[entry_col_idx*k + lane_offset];
							}
						}
					}		
					steps = 8;
				}else{
					int rc_idx = r_c_Offset[entry_idx];   // broadcast
					uint32_t r_offset = (rc_idx & 240)>>4;     // .. & 1111 0000
					uint32_t c_offset = rc_idx & 15;      // .. & 0000 1111
					// load a to shared mem
			        uint8_t aVal_idx_tmp = r_c_Offset[entry_idx];
			        sts8(aVal_idx_tmp, a_idx_sts + (entry_idx-tileNnz[i+warps])*sizeof(uint8_t));
                    float aVal_tmp = vals[entry_idx];
			        sts32(aVal_tmp, a_vals_sts + (r_offset*16+c_offset)*sizeof(float));

                    //if (blockIdx.x==0 && warp_id==0 && lane_id==0){
                        //printf("a_idx = %d\n",rc_idx);
                        //printf("@394:   r_offset = %d, c_offset = %d, aVal_tmp = %f\n",r_offset,c_offset,aVal_tmp);
                    //}
					// load b to registers
					// the i-th bit represents the i-th B row is alrealdy loaded in shared memory
					if ((row_flag[0] & (1<<c_offset)) == 0 ){
						// mark it as loaded
						row_flag[0] |= (1<<c_offset);

						uint32_t entry_col_idx = tileColIdx[i+warps] + c_offset;  // .. & 0000 1111
						uint32_t lane_offset = blockIdx.y*32 + lane_id;
						if (lane_offset<min(blockIdx.y*32+32, k)){
							// matrix B is in row major
							// b_mat_sh[warp_id*2 + warp_tileStart_id%2][c_offset*tn+lane_id] = mat_b[entry_col_idx*k + lane_offset];
							b_reg[(b_double_buf_idx+1)%2][c_offset] = mat_b[entry_col_idx*k + lane_offset];
                            //if (blockIdx.x==1 && warp_id==1 && lane_id==0){
                                //printf("@412:   b_col = %d, b_reg[%d][%d] = %f\n",entry_col_idx,(i+1)%2,c_offset,b_reg[(i+1)%2][c_offset]);
                            //}
						}
					}
					steps = 1;
				}
			}
			row_flag[0] = 0;
			//a_vals_sts ^= 0x0500;
			//a_idx_sts  ^= 0x0500;
			a_vals_sts ^= 0x0600;
			a_idx_sts  ^= 0x0600;
		} // end if
		// ************************************************************************************

	    float a_reg[2] = {0.0,0.0};
		if (nnz_cur_tile <= 1*1024){
            // Cuda cores
			
			// visit all nze in the current tile
			// both tileNNz and r_c_offfset have good locality, so no need to optimize their memory access behavior?
            uint8_t a_idx_tmp;
            lds8(a_idx_tmp, a_idx_lds + 0*sizeof(uint8_t));
			//uint32_t r_offset = a_idx_lds[0] & 240;     // .. & 1111 0000
			//uint32_t c_offset = a_idx_lds[0] & 15;      // .. & 0000 1111
			uint32_t r_offset = (a_idx_tmp & 240)>>4;     // .. & 1111 0000
			uint32_t c_offset = a_idx_tmp & 15;      // .. & 0000 1111
            
            //if (blockIdx.x==0 && warp_id==0 && lane_id==1){
                //printf("a_idx_tmp = %d\n",(int)a_idx_tmp);
                //printf("tileID = %d, r_offset = %d, c_offset = %d\n",i,r_offset,c_offset);
            //}
			//a_reg[tileNnz[i]%2] = a_vals_lds[r_offset*16+c_offset];
			lds32(a_reg[tileNnz[i]%2], a_vals_lds + (r_offset*16+c_offset)*sizeof(float));
			for (uint32_t entry_idx = tileNnz[i]; entry_idx<tileNnz[i+1]; ++entry_idx){
				uint32_t r_offset_tmp, c_offset_tmp;

				// preload A vals
				if ((entry_idx+1)<tileNnz[i+1]){
                    uint8_t a_idx_tmp;
                    lds8(a_idx_tmp, a_idx_lds + (entry_idx+1-tileNnz[i])*sizeof(uint8_t));
					//r_offset_tmp = a_idx_lds[entry_idx+1-tileNnz[i]] & 240;     // .. & 1111 0000
					//c_offset_tmp = a_idx_lds[entry_idx+1-tileNnz[i]] & 15;      // .. & 0000 1111
					r_offset_tmp = (a_idx_tmp & 240)>>4;     // .. & 1111 0000
					c_offset_tmp = a_idx_tmp & 15;      // .. & 0000 1111
					//a_reg[(entry_idx+1)%2] = a_vals_lds[r_offset_tmp*16+c_offset_tmp];
			        lds32(a_reg[(entry_idx+1)%2], a_vals_lds + (r_offset_tmp*16+c_offset_tmp)*sizeof(float));
                    
                    if (blockIdx.x==0 && warp_id==0 && lane_id==0 && i==0 && r_offset==7 && c_offset==7){
                        printf("@493: tileID = %d, entry_idx = %d, r_offset = %d, c_offset = %d, a_reg[%d] = %f\n",i,entry_idx,r_offset_tmp,c_offset_tmp,(entry_idx+1)%2,a_reg[(entry_idx+1)%2]);
                    }
				}
				// bits to mark C rows to write back
				// here if condition can be removed
				if ((row_flag[1] & (1<<r_offset)) == 0){
					row_flag[1] |= (1<<r_offset);
				}
				
				
				uint32_t lane_offset = blockIdx.y*32 + lane_id;
				if (lane_offset<min(blockIdx.y*32+32, k)){
					// multiplication
					// accumulate in local registers
                    if (blockIdx.x==0 && warp_id==0 && lane_id==0 && i==0 && r_offset==8){
                        printf("@495:   a_reg[%d] = %f, c_offset = %d, b_reg[%d][%d] = %f\n", entry_idx%2, a_reg[entry_idx%2], c_offset, b_double_buf_idx, c_offset, b_reg[b_double_buf_idx][c_offset]);
                    }
                    if (blockIdx.x==0 && warp_id==0 && lane_id==0 && i==0){
                        //printf("@496:   r_offset = %d     entry_idx = %d\n",r_offset,entry_idx);
                    }
					res[r_offset] += a_reg[entry_idx%2] * b_reg[b_double_buf_idx][c_offset];	
                    //if (blockIdx.x==0 && warp_id==0 && lane_id==0 && i==0 && r_offset==8){
                    //    printf("@500:   res[%d] = %f\n",r_offset,res[r_offset]);
                    //}
				}
				r_offset = r_offset_tmp;
				c_offset = c_offset_tmp;
			}
			//a_vals_lds ^= 0x0500;
			//a_idx_lds  ^= 0x0500;
			a_vals_lds ^= 0x0600;
			a_idx_lds  ^= 0x0600;
		}else{
            printf("Using tensor cores\n");
            printf("tileID = %d, nnz = %d\n",i,nnz_cur_tile);
			// Tensor cores
		}
	}
    if (lane_id==0) atomicOr(&mark_c_rows[0], row_flag[1]);
    __syncthreads();
    row_flag[1] = mark_c_rows[0];
	
	if (ACC_SH){
		// accumulate partial products on shared memory among warps within one thread block
		// w/o bank conflict since a thread access one bank
		// here row_flags[1] helps to reduce atomicAdd
		uint32_t lane_offset = blockIdx.y*32 + lane_id;
		if (lane_offset<min(blockIdx.y*32+32, k)){
            //if (blockIdx.x==9 && warp_id==0 && lane_id==0)
            //    printf("TM = %d, row_flag[1] = %d\n",TM,row_flag[1]);
			//for (uint32_t j=0; j<TM; ++j){
			for (uint32_t j=0; j<32; ++j){
                if (row_flag[1] & (1<<j)){
			//for (uint32_t j=0; j<TM && (row_flag[1] & (1<<j))!=0; ++j){
				    atomicAdd(&c_mat_sh[j*k+lane_id], res[j]);
                /*
                if (blockIdx.x==5 && warp_id==1 && lane_id==0){
                    printf("@508: c_mat_h[%d] = %f\n",j*32+lane_id,c_mat_sh[j*k+lane_id]);
                }*/
                }
			}
		}
		__syncthreads();
	}
	
	// no need synchronization? because no corporation among warps
	//__syncthreads();
	if (!ACC_SH){
		uint32_t lane_offset = blockIdx.y*32 + lane_id;
		if (lane_offset<min(blockIdx.y*32+32, k)){
			for (uint32_t i=0; i<16 && (row_flag[1] & (1<<i)); ++i){
				uint32_t r = warp_tileRow_idx[blockIdx.x] + i;
				atomicAdd(&mat_c[r*k+lane_offset], res[i]);
			}
		}
        //printf("Not acc in shared mem\n");	
	}else{
		// transfer results from shared mem to glb mem
		if((blockIdx.x && (warp_tileRow_idx[blockIdx.x]==warp_tileRow_idx[blockIdx.x-1])) 
                || warp_tileRow_idx[blockIdx.x]==warp_tileRow_idx[blockIdx.x+1]){
			// multi blocks work on one row tiles
			// global memory atomic write, in coalesced way
			uint32_t lane_offset = blockIdx.y*32 + lane_id;
			if (lane_offset<min(blockIdx.y*32+32, k)){
                    //if (blockIdx.x==5 && warp_id==1 && lane_id==0)
                      //  printf("@538: TM = %d, row_flag[1] = %d\n",TM,row_flag[1]);
				// each warp transfer one row segment of C
				//for (uint32_t i=warp_id; i<16 && (row_flag[1] & (1<<i)); i+=WARPS){
				//for (uint32_t i=warp_id; i<TM; i+=WARPS){
				for (uint32_t i=warp_id; i<32; i+=WARPS){
                    if (row_flag[1] & (1<<i)){
					    uint32_t r = warp_tileRow_idx[blockIdx.x] + i;
					    atomicAdd(&mat_c[r*k+lane_offset], c_mat_sh[i*32+lane_id]);
                        /*
                        if (blockIdx.x==4 && warp_id==0 && lane_id==0){
                            // printf("@547: c_sh[%d] = %f, mat_c[%d] = %f\n", i*32+lane_id, c_mat_sh[i*32+lane_id], r*k+lane_offset, mat_c[r*k+lane_offset]);
                        }
                        if (blockIdx.x==5 && warp_id==0 && lane_id==0){
                            //printf("@551: c_sh[%d] = %f, mat_c[%d] = %f\n", i*32+lane_id, c_mat_sh[i*32+lane_id], r*k+lane_offset, mat_c[r*k+lane_offset]);
                        }
                        if (blockIdx.x==5 && warp_id==1 && lane_id==0){
                            //printf("@554: c_sh[%d] = %f, mat_c[%d] = %f\n", i*32+lane_id, c_mat_sh[i*32+lane_id], r*k+lane_offset, mat_c[r*k+lane_offset]);
                        }
                        */
                    }
				}
			}
            //printf("Acc1 in shared mem\n");	
		}else{
            //printf("@551 ___________-----\n");
			// global memory write, in coalesced way
			// transfer results from shared to glb
			uint32_t lane_offset = blockIdx.y*32 + lane_id;
			if (lane_offset<min(blockIdx.y*32+32, k)){
				// each warp transfer one row segment of C
                        //if (blockIdx.x==9 && warp_id==0 && lane_id==0){
                        //     printf("@592: row_flag[1] = %d\n", row_flag[1]);
                        //}
				//for (uint32_t i=warp_id; i<TM && (row_flag[1] & (1<<i)); i+=WARPS){
				for (uint32_t i=warp_id; i<32 && (row_flag[1] & (1<<i)); i+=WARPS){
					uint32_t r = warp_tileRow_idx[blockIdx.x] + i;
					mat_c[r*k+lane_offset] = c_mat_sh[i*32+lane_id];
					//atomicAdd(&mat_c[r*k+lane_offset], c_mat_sh[i*32+lane_id]);
                    //printf("r = %d, c = %d, val = %f\n",r,lane_offset,mat_c[r*k+lane_offset]);
				}
                //printf("Acc2 in shared mem\n");	
			}	
		}
	}
	return ;
}
// inputs:
//		 block_tileStart_idx: the tile idex of the first tile computed by thread-blocks
//       tileColIdx: the column idex of the first column of each tile
//       tileNnz: the index of the first nze(non-zero entry) in each tile
//       warp_tileRow_idx: row idx of each tile
//       tiles: # of tiles
//       r_c_Offset(8 bits): the row index (upper 4 bits) + column index (lower 4 bits) of each nze in each tile
//       vals: non zero entries
// The real row-col index for a nze (the i-th row tile, the j-th tile of the i-th row tile): 
//  r = i + r_c_Offset[7...4], c = tileColIdx[ tileRowPtr[i] + j ] + r_c_Offset[3...0]

// A: sparse, m * n
// B: dense, n * k   (k << n)
__global__
void flexspgemm_cuda_reg_pre_v2(int* tileNnz,
				int* block_tileStart_idx,
				int* warp_tileRow_idx,
                int* tileColIdx,
				int tiles,
				char* r_c_Offset,
				float* vals,
				int k,
				float* mat_b,
                float* mat_c){
	const uint32_t WARPSZ = 32;
	uint32_t lane_id = threadIdx.x % WARPSZ;
    uint32_t warp_id = threadIdx.x / WARPSZ;
	uint32_t warps = (blockDim.x + WARPSZ - 1)/WARPSZ;

	int tile_ref[3] = {-1};
	if (blockIdx.x | 0){
		tile_ref[0] = block_tileStart_idx[blockIdx.x-1];
	}
	tile_ref[1] = block_tileStart_idx[blockIdx.x];
	tile_ref[2] = block_tileStart_idx[blockIdx.x+1];
	const uint32_t warp_tileStart_id = tile_ref[1] + warp_id;
	
	// matrix A: 
	//          non-zero vals: tm*tn * 4bytes * double buffer <= 16*16 * 4 * 2 = 2k
	//          non-zero entry idx: tm*tn * 1bytes * double buffer <= 16*16 * 1 * 2 = 0.5k
	// matrix B:
	//          each thread has 16 regs to store one B column segment
	// Only A is stored in shared memory
	// 2 warps in a block
	#define WARPS 2
	#define ACC_SH  true
	#if ACC_SH
	//__shared__ char smem[(8*16*16 + 1024)*WARPS+32*32*4+4];
    // the largest tile is 32*16, we assume the nnz is no greater than half.
    // here we didn't consider smem for Result C tile.we hope to reuse the smem
    // currently, at most 256 non-zeros per tile is allowed and can be loaded to smem
	__shared__ int smem[2][(16*16 + 64)*WARPS+1];
	#else
	//__shared__ char smem[2][];
	#endif
	//       |------------ warp0 -------------|--------------- warp1 ---------------|-- c row mark(1) -|
    // buf0  |--- vals(16*16) ---|-- rc(64) --|----- vals(16*16) ------|-- rc(64) --|
    // buf1  |--- vals(16*16) ---|-- rc(64) --|----- vals(16*16) ------|-- rc(64) --|


    uint32_t duo_switch = 0;
    float* a_vals_smem = reinterpret_cast<float *>(smem[duo_switch]+warp_id*(16*16+64));                
	uint8_t* a_idx_smem = reinterpret_cast<uint8_t *>(smem[duo_switch]+warp_id*(16*16+64)+16*16);
	float* c_mat_sh = reinterpret_cast<float *>(smem);  // reuse smem

    uint32_t* mark_c_rows = reinterpret_cast<uint32_t *>(smem[0]+(16*16 + 64)*WARPS);
    for (uint32_t i=warp_id; i<32; i+=warps){
        c_mat_sh[i*32+lane_id] = 0;
    }
    if (threadIdx.x==0) mark_c_rows[0] = 0;

	float* a_vals_sts = a_vals_smem;
	float* a_vals_lds = a_vals_smem;
	uint8_t* a_idx_sts = a_idx_smem;
	uint8_t* a_idx_lds = a_idx_smem;

	// ************ load A tile ( && B rows required by first tile) from glb mem to shared memory (registers) *********************************************
	// row_flag[0]: bits represent existance of B rows, determined by col idx of A entrys 
	// row_flag[1]: bits represent existance of C rows, determined by row idx of A entrys 
	uint32_t row_flag[2] = {0,0};
	
  	
    float b_reg[2][32];
	// both tileNNz and r_c_offfset have good locality, so no need to optimize their memory access behavior? 
	#define FULL_MASK 0xffffffff
	uint32_t steps = 0;
	for (uint32_t entry_idx = tileNnz[warp_tileStart_id]; entry_idx<tileNnz[warp_tileStart_id+1]; entry_idx += steps){
	
        uint32_t dist = entry_idx - tileNnz[warp_tileStart_id];
        // get active threads
        unsigned mask = __ballot_sync(FULL_MASK, entry_idx+lane_id<tileNnz[warp_tileStart_id+1]);
        // compute steps
        // it represents the # of threads participating loading A non-zeros
        steps = __popc(mask);
        
        //load A to shared memory
		int rc_idx = 0; // in coalesced way
        uint32_t r_offset,c_offset;
        if (entry_idx+lane_id<tileNnz[warp_tileStart_id+1]){
		    rc_idx = r_c_Offset[entry_idx+lane_id]; // in coalesced way
			
            // continuous store in shared memory
            a_idx_sts[dist+lane_id] = r_c_Offset[entry_idx+lane_id];
            a_vals_sts[dist+lane_id] = vals[entry_idx+lane_id];
        } 
        __syncwarp();
        
                    //if (blockIdx.x==0 && warp_id==0){
					//    uint32_t r_offset_tmp = (rc_idx & 240)>>4;     // .. & 1111 0000
					//    uint32_t c_offset_tmp = rc_idx & 15;      // .. & 0000 1111
                    //    float a_reg = a_vals_sts[dist+lane_id];
                    //    printf("@756: steps = %d, lane_id = %d, a_reg = %f, r_offset = %d, c_offset = %d\n", steps, lane_id, a_reg, r_offset_tmp, c_offset_tmp);
                    //}
        // load B
        for (uint32_t j=0; j<steps; ++j){
            // OR load from smem: rc_idx_tmp = a_idx_sts[j], benefit: broadcast
			int rc_idx_tmp = __shfl_sync(FULL_MASK, rc_idx, j);

			r_offset = (rc_idx_tmp & 240)>>4;     // .. & 1111 0000
			c_offset = rc_idx_tmp & 15;      // .. & 0000 1111
        

			// the i-th bit 1 represents the i-th B row is alrealdy loaded in shared memory
			if ((row_flag[0] & (1<<c_offset)) == 0 ){
				// mark it as loaded
				row_flag[0] |= (1<<c_offset);

				uint32_t entry_col_idx = tileColIdx[warp_tileStart_id] + c_offset;  // .. & 0000 1111
				uint32_t lane_offset = blockIdx.y*32 + lane_id;
				if (lane_offset<min(blockIdx.y*32+32, k)){
					// matrix B is in row major, glb mem->regs, coalesced access
					b_reg[duo_switch][c_offset] = mat_b[entry_col_idx*k + lane_offset];
				}
			}    
        }    
	}

    // mark B rows to avoid redundant/repeated B loading
	row_flag[0] = 0;

        
    a_vals_sts = reinterpret_cast<float *>(smem[duo_switch^1]+warp_id*(16*16+64));
	a_idx_sts  = reinterpret_cast<uint8_t *>(smem[duo_switch^1]+warp_id*(16*16+64)+16*16);
	
    // ***************************************************************************************************************************************************

	// multiplication loops	
	float res[32];
    for (int i=0; i<32; ++i) res[i] = 0;
	
    // iterate tiles assigned to the current block
    for (uint32_t i=warp_tileStart_id; i<tile_ref[2]; i += warps){

        //if (blockIdx.x==0 && warp_id==0 && lane_id==0){
        //    printf("@804: i = %d\n", i);
        //}

        int  nnz_cur_tile = tileNnz[i+1]-tileNnz[i];
        
        //if (blockIdx.x==0 && warp_id==0 && lane_id==0){
        //    printf("@810:   tileref[0] = %d,tileref[1] = %d,tileref[2] = %d\n",tile_ref[0],tile_ref[1],tile_ref[2]);
        //}

        // ************ load B rows required by "next A tile" from glb mem to shmem **********
		// both tileNNz and r_c_offfset have good locality, so no need to optimize their memory access behavior?
		if (i+warps<tile_ref[2]){
			for (uint32_t entry_idx = tileNnz[i+warps]; entry_idx<tileNnz[i+warps+1]; entry_idx += steps){
                
                uint32_t dist = entry_idx - tileNnz[i+warps];
                // get active threads
                uint32_t mask = __ballot_sync(FULL_MASK, entry_idx+lane_id<tileNnz[i+warps+1]);
                // compute steps
                // it represents the # of threads participating loading A non-zeros
                steps = __popc(mask);
        
                //load A to shared memory
		        int rc_idx = 0; // in coalesced way
                uint32_t r_offset,c_offset;
                if (entry_idx+lane_id<tileNnz[i+warps+1]){
		            rc_idx = r_c_Offset[entry_idx+lane_id]; // in coalesced way
		            uint32_t r_offset = (rc_idx & 240)>>4;     // .. & 1111 0000
		            uint32_t c_offset = rc_idx & 15;      // .. & 0000 1111
			
                    // continuous store in shared memory
                    a_idx_sts[dist+lane_id] = r_c_Offset[entry_idx+lane_id];
                    a_vals_sts[dist+lane_id] = vals[entry_idx+lane_id]; 
                }
                __syncwarp(); // is it necessary for correctness of steps?
                // load B
                for (uint32_t j=0; j<steps; ++j){
			        int rc_idx_tmp = __shfl_sync(FULL_MASK, rc_idx, j);

			        c_offset = rc_idx_tmp & 15;      // .. & 0000 1111
                    
			        
                    // the i-th bit 1 represents the i-th B row is alrealdy loaded in shared memory
			        if ((row_flag[0] & (1<<c_offset)) == 0 ){
				        // mark it as loaded
				        row_flag[0] |= (1<<c_offset);

				        uint32_t entry_col_idx = tileColIdx[i+warps] + c_offset;  // .. & 0000 1111
				        uint32_t lane_offset = blockIdx.y*32 + lane_id;
				        if (lane_offset<min(blockIdx.y*32+32, k)){
					        // matrix B is in row major
					        // b_mat_sh[warp_id*2 + warp_tileStart_id%2][c_offset*tn+lane_id] = mat_b[entry_col_idx*k + lane_offset];
					        b_reg[duo_switch^1][c_offset] = mat_b[entry_col_idx*k + lane_offset];
				        }
			        }    
                }

			}
			row_flag[0] = 0;
		} // end if
		// ************************************************************************************
        
	    float a_reg = 0.0;
		if (nnz_cur_tile <= 1*1024){
            // Cuda cores
			
            //if (blockIdx.x==0 && warp_id==0 && lane_id==1){
                //printf("a_idx_tmp = %d\n",(int)a_idx_tmp);
                //printf("tileID = %d, r_offset = %d, c_offset = %d\n",i,r_offset,c_offset);
            //}
            uint32_t jumps = 0;

            // all threads need to go into the for loop
			for (uint32_t entry_idx = tileNnz[i]; entry_idx<tileNnz[i+1]; entry_idx += jumps){
                uint32_t dist = entry_idx - tileNnz[i];
                // get active threads
                uint32_t mask_join = __ballot_sync(FULL_MASK, entry_idx+lane_id<tileNnz[i+1]);
                jumps = __popc(mask_join); 
                
                uint32_t rc_idx = 0; 
                if (entry_idx+lane_id<tileNnz[i+1]){
                    rc_idx = a_idx_lds[dist+lane_id];
			        a_reg = a_vals_lds[dist+lane_id];
                }
                __syncwarp(); // is it necessary for correctness of jumps?
                
                //if (blockIdx.x==0 && warp_id==0 && lane_id==3){
                //    printf("@882:    jumps = %d \n",jumps);
                //}
               
                    //if (blockIdx.x==0 && warp_id==0 && lane_id<3){
					//    uint32_t r_offset_tmp = (rc_idx & 240)>>4;     // .. & 1111 0000
					//    uint32_t c_offset_tmp = rc_idx & 15;      // .. & 0000 1111
                    //    printf("@885:    lane_id = %d, a_reg = %f, r_offset = %d, c_offset = %d\n",lane_id,a_reg, r_offset_tmp, c_offset_tmp);
                    //}
                // in a round robin way  
                uint32_t r_offset_tmp, c_offset_tmp;
                for (uint32_t j=0; j<jumps; ++j){
			        uint32_t rc_idx_tmp = __shfl_sync(FULL_MASK, rc_idx, j);   // may be inefficient because only 25% efficiency
			        float a_reg_tmp = __shfl_sync(FULL_MASK, a_reg, j);

					r_offset_tmp = (rc_idx_tmp & 240)>>4;     // .. & 1111 0000
					c_offset_tmp = rc_idx_tmp & 15;      // .. & 0000 1111
				   
                    // bits to mark C rows to write back
				    // here if condition can be removed
				    //if ((row_flag[1] & (1<<r_offset_tmp)) == 0){
					row_flag[1] |= (1<<r_offset_tmp);
				    //}

                    uint32_t lane_offset = blockIdx.y*32 + lane_id;
				    if (lane_offset<min(blockIdx.y*32+32, k)){
					    res[r_offset_tmp] += a_reg_tmp * b_reg[duo_switch][c_offset_tmp];
				    }
                } 
            }
			a_vals_lds = a_vals_sts;
			a_idx_lds  = a_idx_sts;
            duo_switch ^= 1;
            a_vals_sts = reinterpret_cast<float *>(smem[duo_switch]+warp_id*(16*16+64));
	        a_idx_sts  = reinterpret_cast<uint8_t *>(smem[duo_switch]+warp_id*(16*16+64)+16*16);
		}else{
            printf("Using tensor cores\n");
            printf("tileID = %d, nnz = %d\n",i,nnz_cur_tile);
			// Tensor cores
		}
        
	}

    if (lane_id==0) atomicOr(mark_c_rows, row_flag[1]);
    __syncthreads();
    row_flag[1] = mark_c_rows[0];

	if (ACC_SH){
        for (int ii=warp_id; ii<32; ii += WARPS){
            c_mat_sh[ii*32+lane_id] = 0;
        }
        __syncthreads();
        
		// accumulate partial products on shared memory among warps within one thread block
		// w/o bank conflict since a thread access one bank
		// here row_flags[1] helps to reduce atomicAdd
		uint32_t lane_offset = blockIdx.y*32 + lane_id;
		if (lane_offset<min(blockIdx.y*32+32, k)){
            //if (blockIdx.x==9 && warp_id==0 && lane_id==0)
            //    printf("TM = %d, row_flag[1] = %d\n",TM,row_flag[1]);
			//for (uint32_t j=0; j<TM; ++j){
			for (uint32_t j=0; j<32; ++j){
                if (row_flag[1] & (1<<j)){
			//for (uint32_t j=0; j<TM && (row_flag[1] & (1<<j))!=0; ++j){
				    atomicAdd(&c_mat_sh[j*32+lane_id], res[j]);
                
                }
			}
		}
		__syncthreads();
	}
	
	// no need synchronization? because no corporation among warps
	//__syncthreads();
	if (!ACC_SH){
		uint32_t lane_offset = blockIdx.y*32 + lane_id;
		if (lane_offset<min(blockIdx.y*32+32, k)){
			for (uint32_t i=0; i<16 && (row_flag[1] & (1<<i)); ++i){
				uint32_t r = warp_tileRow_idx[blockIdx.x] + i;
				atomicAdd(&mat_c[r*k+lane_offset], res[i]);
			}
		}
        //printf("Not acc in shared mem\n");	
	}else{
		// transfer results from shared mem to glb mem
		if((blockIdx.x && (warp_tileRow_idx[blockIdx.x]==warp_tileRow_idx[blockIdx.x-1])) 
                || warp_tileRow_idx[blockIdx.x]==warp_tileRow_idx[blockIdx.x+1]){
			// multi blocks work on one row tiles
			// global memory atomic write, in coalesced way
			uint32_t lane_offset = blockIdx.y*32 + lane_id;
			if (lane_offset<min(blockIdx.y*32+32, k)){
                    //if (blockIdx.x==5 && warp_id==1 && lane_id==0)
                      //  printf("@538: TM = %d, row_flag[1] = %d\n",TM,row_flag[1]);
				// each warp transfer one row segment of C
				for (uint32_t i=warp_id; i<32; i+=WARPS){
                    if (row_flag[1] & (1<<i)){
					    uint32_t r = warp_tileRow_idx[blockIdx.x] + i;
					    atomicAdd(&mat_c[r*k+lane_offset], c_mat_sh[i*32+lane_id]);
                        
                        //if (blockIdx.x==4 && warp_id==0 && lane_id==0){
                            // printf("@547: c_sh[%d] = %f, mat_c[%d] = %f\n", i*32+lane_id, c_mat_sh[i*32+lane_id], r*k+lane_offset, mat_c[r*k+lane_offset]);
                        //}
                        //if (blockIdx.x==5 && warp_id==0 && lane_id==0){
                            //printf("@551: c_sh[%d] = %f, mat_c[%d] = %f\n", i*32+lane_id, c_mat_sh[i*32+lane_id], r*k+lane_offset, mat_c[r*k+lane_offset]);
                        //}
                        //if (blockIdx.x==5 && warp_id==1 && lane_id==0){
                            //printf("@554: c_sh[%d] = %f, mat_c[%d] = %f\n", i*32+lane_id, c_mat_sh[i*32+lane_id], r*k+lane_offset, mat_c[r*k+lane_offset]);
                        //}
                        
                    }
				}
			}
            //printf("Acc1 in shared mem\n");	
		}else{
            //printf("@983 ___________-----\n");
			// global memory write, in coalesced way
			// transfer results from shared to glb
			uint32_t lane_offset = blockIdx.y*32 + lane_id;
			if (lane_offset<min(blockIdx.y*32+32, k)){
				// each warp transfer one row segment of C
                        //if (blockIdx.x==9 && warp_id==0 && lane_id==0){
                        //     printf("@592: row_flag[1] = %d\n", row_flag[1]);
                        //}
				for (uint32_t i=warp_id; i<32 && (row_flag[1] & (1<<i)); i+=WARPS){
					uint32_t r = warp_tileRow_idx[blockIdx.x] + i;
					mat_c[r*k+lane_offset] = c_mat_sh[i*32+lane_id];
					//atomicAdd(&mat_c[r*k+lane_offset], c_mat_sh[i*32+lane_id]);
                    //printf("r = %d, c = %d, val = %f\n",r,lane_offset,mat_c[r*k+lane_offset]);
				}
                //printf("Acc2 in shared mem\n");	
			}	
		}
        
	} 
	return ;
}
// inputs:
//		 block_tileStart_idx: the tile idex of the first tile computed by thread-blocks
//       tileColIdx: the column idex of the first column of each tile
//       tileNnz: the index of the first nze(non-zero entry) in each tile
//       warp_tileRow_idx: row idx of each tile
//       tiles: # of tiles
//       r_c_Offset(8 bits): the row index (upper 4 bits) + column index (lower 4 bits) of each nze in each tile
//       vals: non zero entries
// The real row-col index for a nze (the i-th row tile, the j-th tile of the i-th row tile): 
//  r = i + r_c_Offset[7...4], c = tileColIdx[ tileRowPtr[i] + j ] + r_c_Offset[3...0]

// A: sparse, m * n
// B: dense, n * k   (k << n)
__global__
void flexspgemm_cuda_wo_pre_v3(int* tileNnz,
				int* block_tileStart_idx,
				int* warp_tileRow_idx,
                int* tileColIdx,
				int tiles,
				char* r_c_Offset,
				float* vals,
				int k,
				float* mat_b,
                float* mat_c){
	const uint32_t WARPSZ = 32;
	uint32_t lane_id = threadIdx.x % WARPSZ;
    uint32_t warp_id = threadIdx.x / WARPSZ;
	uint32_t warps = (blockDim.x + WARPSZ - 1)/WARPSZ;

	int tile_ref[3] = {-1};
	if (blockIdx.x | 0){
		tile_ref[0] = block_tileStart_idx[blockIdx.x-1];
	}
	tile_ref[1] = block_tileStart_idx[blockIdx.x];
	tile_ref[2] = block_tileStart_idx[blockIdx.x+1];
	const uint32_t warp_tileStart_id = tile_ref[1] + warp_id;
	
	// matrix A: 
	//          non-zero vals: tm*tn * 4bytes * double buffer <= 16*16 * 4 * 2 = 2k
	//          non-zero entry idx: tm*tn * 1bytes * double buffer <= 16*16 * 1 * 2 = 0.5k
	// matrix B:
	//          each thread has 16 regs to store one B column segment
	// Only A is stored in shared memory
	// 2 warps in a block
	#define WARPS 2
	#define FULL_MASK 0xffffffff
	#define ACC_SH  false
	float* c_mat_sh;  // reuse smem
	#if ACC_SH
    // currently, at most 16*32 output C tile per block is allowed
	__shared__ float smem[16*32];
	c_mat_sh = smem;  // reuse smem
    for (uint32_t i=warp_id; i<32; i+=warps){
        c_mat_sh[i*32+lane_id] = 0;
    }
	#endif



  	
    float b_reg[32];
	float res[32];
    for (int i=0; i<32; ++i) res[i] = 0;
    
    uint32_t b_row_flag = 0;
    uint32_t c_row_flag = 0;
    // iterate tiles assigned to the current block
    for (uint32_t i=warp_tileStart_id; i<tile_ref[2]; i += warps){

        //if (blockIdx.x==0 && warp_id==0 && lane_id==0){
        //    printf("@804: i = %d\n", i);
        //}

        int  nnz_cur_tile = tileNnz[i+1]-tileNnz[i];
        
	    float a_reg = 0.0;
		if (nnz_cur_tile <= 1*1024){
            // Cuda cores
			
            //if (blockIdx.x==0 && warp_id==0 && lane_id==1){
                //printf("a_idx_tmp = %d\n",(int)a_idx_tmp);
                //printf("tileID = %d, r_offset = %d, c_offset = %d\n",i,r_offset,c_offset);
            //}
            uint32_t jumps = 0;

            // all threads need to go into the for loop
			for (uint32_t entry_idx = tileNnz[i]; entry_idx<tileNnz[i+1]; entry_idx += jumps){
                
                // get active threads
                uint32_t mask_join = __ballot_sync(FULL_MASK, entry_idx+lane_id<tileNnz[i+1]);
                jumps = __popc(mask_join); 
                
                uint32_t rc_idx = 0; 
                if (entry_idx+lane_id<tileNnz[i+1]){
                    rc_idx = r_c_Offset[entry_idx+lane_id]; // in coalesced way
			        a_reg = vals[entry_idx+lane_id]; // in coalesced way
                }
                __syncwarp(); // is it necessary for correctness of jumps?
                
                //if (blockIdx.x==0 && warp_id==0 && lane_id==3){
                //    printf("@882:    jumps = %d \n",jumps);
                //}
               
                    //if (blockIdx.x==0 && warp_id==0 && lane_id<3){
					//    uint32_t r_offset_tmp = (rc_idx & 240)>>4;     // .. & 1111 0000
					//    uint32_t c_offset_tmp = rc_idx & 15;      // .. & 0000 1111
                    //    printf("@885:    lane_id = %d, a_reg = %f, r_offset = %d, c_offset = %d\n",lane_id,a_reg, r_offset_tmp, c_offset_tmp);
                    //}
                // in a round robin way  
                uint32_t r_offset_tmp, c_offset_tmp;
                for (uint32_t j=0; j<jumps; ++j){
			        uint32_t rc_idx_tmp = __shfl_sync(FULL_MASK, rc_idx, j);   // may be inefficient because only 25% efficiency
			        float a_reg_tmp = __shfl_sync(FULL_MASK, a_reg, j);

					r_offset_tmp = (rc_idx_tmp & 240)>>4;     // .. & 1111 0000
					c_offset_tmp = rc_idx_tmp & 15;      // .. & 0000 1111
				   
                    // bits to mark C rows to write back
				    // here if condition can be removed

                    uint32_t lane_offset = blockIdx.y*32 + lane_id;
				    if (lane_offset<min(blockIdx.y*32+32, k)){
				       
                        //if (blockIdx.x==0 && warp_id==0 &&  lane_id==0){
                        //    printf("@1144: r_offset_tmp = %d, c_offset_tmp = %d, val = %f\n", r_offset_tmp, c_offset_tmp, a_reg_tmp);
                        //} 
					    c_row_flag |= (1<<r_offset_tmp);
				        uint32_t entry_col_idx = tileColIdx[i] + c_offset_tmp;  // .. & 0000 1111
                        //if (blockIdx.x==0 && warp_id==0 &&  lane_id==0){
                        //    printf("@1149: entry_col_idx = %d, val = %f\n", entry_col_idx, mat_b[entry_col_idx*k + lane_offset]);
                        //} 
                        if ((b_row_flag & (1<<c_offset_tmp)) == 0){
					        b_row_flag |= (1<<c_offset_tmp);
                            b_reg[c_offset_tmp] = mat_b[entry_col_idx*k + lane_offset]; // coalesced access
				        }

					    res[r_offset_tmp] += a_reg_tmp * b_reg[c_offset_tmp];
                        
                        //if (blockIdx.x==0 && warp_id==0 &&  lane_id==0){
                        //    printf("@1158: r_offset_tmp = %d, c_offset_tmp = %d, val = %f\n", r_offset_tmp, c_offset_tmp, res[r_offset_tmp]);
                        //} 
				    }
                } 
            }
		}else{
            printf("Using tensor cores\n");
            printf("tileID = %d, nnz = %d\n",i,nnz_cur_tile);
			// Tensor cores
		}
        b_row_flag = 0; 
	}

	if (ACC_SH){	
        // accumulate partial products on shared memory among warps within one thread block
		// w/o bank conflict since a thread access one bank
		// here c_row_flags helps to reduce atomicAdd
		uint32_t lane_offset = blockIdx.y*32 + lane_id;
		if (lane_offset<min(blockIdx.y*32+32, k)){
            //if (blockIdx.x==9 && warp_id==0 && lane_id==0)
            //    printf("TM = %d, row_flag[1] = %d\n",TM,row_flag[1]);
			//for (uint32_t j=0; j<TM; ++j){
			for (uint32_t j=0; j<32; ++j){
                if (c_row_flag & (1<<j)){
				    atomicAdd(&c_mat_sh[j*32+lane_id], res[j]);
                }
			}
		}
		__syncthreads();
		// transfer results from shared mem to glb mem
		if((blockIdx.x && (warp_tileRow_idx[blockIdx.x]==warp_tileRow_idx[blockIdx.x-1])) 
                || warp_tileRow_idx[blockIdx.x]==warp_tileRow_idx[blockIdx.x+1]){
			// multi blocks work on one row tiles
			// global memory atomic write, in coalesced way
			uint32_t lane_offset = blockIdx.y*32 + lane_id;
			if (lane_offset<min(blockIdx.y*32+32, k)){
                    //if (blockIdx.x==5 && warp_id==1 && lane_id==0)
                      //  printf("@538: TM = %d, row_flag[1] = %d\n",TM,row_flag[1]);
				// each warp transfer one row segment of C
				for (uint32_t i=warp_id; i<32; i+=WARPS){
                    if (c_row_flag & (1<<i)){
					    uint32_t r = warp_tileRow_idx[blockIdx.x] + i;
					    atomicAdd(&mat_c[r*k+lane_offset], c_mat_sh[i*32+lane_id]);
                        
                        //if (blockIdx.x==4 && warp_id==0 && lane_id==0){
                            // printf("@547: c_sh[%d] = %f, mat_c[%d] = %f\n", i*32+lane_id, c_mat_sh[i*32+lane_id], r*k+lane_offset, mat_c[r*k+lane_offset]);
                        //}
                        //if (blockIdx.x==5 && warp_id==0 && lane_id==0){
                            //printf("@551: c_sh[%d] = %f, mat_c[%d] = %f\n", i*32+lane_id, c_mat_sh[i*32+lane_id], r*k+lane_offset, mat_c[r*k+lane_offset]);
                        //}
                        //if (blockIdx.x==5 && warp_id==1 && lane_id==0){
                            //printf("@554: c_sh[%d] = %f, mat_c[%d] = %f\n", i*32+lane_id, c_mat_sh[i*32+lane_id], r*k+lane_offset, mat_c[r*k+lane_offset]);
                        //}
                        
                    }
				}
			}
            //printf("Acc1 in shared mem\n");	
		}else{
            //printf("@983 ___________-----\n");
			// global memory write, in coalesced way
			// transfer results from shared to glb
			uint32_t lane_offset = blockIdx.y*32 + lane_id;
			if (lane_offset<min(blockIdx.y*32+32, k)){
				// each warp transfer one row segment of C
                        //if (blockIdx.x==9 && warp_id==0 && lane_id==0){
                        //     printf("@592: row_flag[1] = %d\n", row_flag[1]);
                        //}
				for (uint32_t i=warp_id; i<32; i+=WARPS){
					if (c_row_flag & (1<<i)){
                        uint32_t r = warp_tileRow_idx[blockIdx.x] + i;
					    mat_c[r*k+lane_offset] = c_mat_sh[i*32+lane_id];
                    }
				}
                //printf("Acc2 in shared mem\n");	
			}	
		}
	}else{ 
		uint32_t lane_offset = blockIdx.y*32 + lane_id;
		if (lane_offset<min(blockIdx.y*32+32, k)){
			for (uint32_t i=0; i<32; ++i){
				if (c_row_flag & (1<<i)){
                    uint32_t r = warp_tileRow_idx[blockIdx.x] + i;
				    atomicAdd(&mat_c[r*k+lane_offset], res[i]);
                        
                    //if (blockIdx.x==0 && warp_id==0 &&  lane_id==0){
                    //    printf("@1241: r = %d, val = %f\n", r, res[i]);
                    //} 
                }
			}
		}
    }
    
	return ;
}


