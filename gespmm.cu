
__global__ 
void spmm_test0()
{
    // this code is adapted from "https://github.com/hgyhungry/ge-spmm.git"
    
    // kernel for B_ncols in [1,32)
    // grid: dim3( (A_nrows+128/B_ncols-1)/(128/B_ncols), 1, 1  ) 
    // block: dim3( B_ncols, 128/B_ncols, 1 )
    // shared m: 32*8*(sizeof(int)+sizeof(float)) 
    const Mat_POD& md = mat_dev;
    
    int rid = blockDim.y*blockIdx.x+threadIdx.y;
    if (rid<md.md.A_nrows) {
    int cid = (blockIdx.y<<5)+threadIdx.x;
    int lb = md.A_csrRowPtr[rid];
    int hb = md.A_csrRowPtr[(rid+1)];
    int offset = 0;
    float acc=0;
    if (blockIdx.y!=gridDim.y-1){
        for (int ptr = lb; ptr<hb; ptr++) {
            offset = md.A_csrColInd[ptr]*md.B_ncols+cid;
            acc += md.A_csrVal[ptr]*md.mat_b_dev[offset];
        }
        md.mat_c_dev[(rid*md.B_ncols+cid)] = acc;
    }
    else {
        for (int ptr = lb; ptr<hb; ptr++) {
            if (cid<md.B_ncols) {
            offset = md.A_csrColInd[ptr]*md.B_ncols+cid;}
            acc += md.A_csrVal[ptr]*md.mat_b_dev[offset];
        }
        if (cid<md.B_ncols) {
        md.mat_c_dev[(rid*md.B_ncols+cid)] = acc;}
    }
    }
}

__global__ 
void spmm_test1()
{
    // this code is adapted from "https://github.com/hgyhungry/ge-spmm.git"
    
    // kernel for B_ncols in [32,64)
    // grid: dim3( (A_nrows+4-1)/4, (B_ncols+31)/32, 1  ) 
    // block: dim3( 32,4,1 )
    // shared m: 32*4*(sizeof(int)+sizeof(float)) 
    const Mat_POD& md = mat_dev;
    
    extern __shared__ int sh[];
    int *colInd_sh = sh;
    float *val_sh = (float *)&sh[(blockDim.y<<5)];
    int shmem_offset = (threadIdx.y<<5);
    int thread_idx = shmem_offset+threadIdx.x;

    int rid = blockDim.y*blockIdx.x+threadIdx.y;

    if (rid<md.md.A_nrows) {
        int cid = (blockIdx.y<<5)+threadIdx.x;
        int lb = md.A_csrRowPtr[rid];
        int hb = md.A_csrRowPtr[(rid+1)];
        int ptr = lb+threadIdx.x;
        int offset;
        float acc=0;

        if (blockIdx.y != gridDim.y-1) {
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = md.A_csrVal[ptr];
                    colInd_sh[thread_idx] = md.B_ncols*md.A_csrColInd[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    acc += val_sh[(shmem_offset+kk)]*md.mat_b_dev[offset];
                }
                __syncwarp();
            }
            md.mat_c_dev[(rid*md.B_ncols+cid)] = acc;
        }
        else {
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = md.A_csrVal[ptr];
                    colInd_sh[thread_idx] = md.B_ncols*md.A_csrColInd[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    if (cid<md.B_ncols) {
                    acc += val_sh[(shmem_offset+kk)]*md.mat_b_dev[offset];
                    }
                }
                __syncwarp();
            }
            if (cid<md.B_ncols) {
            md.mat_c_dev[(rid*md.B_ncols+cid)] = acc;
            }
        }
    }
}

__global__ 
void spmm_test2()
{

    // this code is adapted from "https://github.com/hgyhungry/ge-spmm.git"
    
    // kernel for B_ncols in [64,+oo)
    // grid: dim3( (A_nrows+8-1)/8, (B_ncols+63)/64, 1  ) 
    // block: dim3( 32,8,1 )
    // shared m: 32*8*(sizeof(int)+sizeof(float)) 

    const Mat_POD& md = mat_dev;
    
    extern __shared__ int sh[];
    int *colInd_sh = sh;
    float *val_sh = (float *)&sh[(blockDim.y<<5)];
    int shmem_offset = (threadIdx.y<<5);
    int thread_idx = shmem_offset+threadIdx.x;

    int rid = blockDim.y*blockIdx.x+threadIdx.y;

   if (rid<md.A_nrows) {
        int cid = (blockIdx.y<<6)+threadIdx.x;
        int lb = md.A_csrRowPtr[rid];
        int hb = md.A_csrRowPtr[(rid+1)];
        int ptr = lb+threadIdx.x;
        int offset;
        float acc1=0, acc2=0, val;

        if (blockIdx.y != gridDim.y-1) {
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = md.A_csrVal[ptr];
                    colInd_sh[thread_idx] = md.B_ncols*md.A_csrColInd[ptr];
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
            offset = rid*md.B_ncols+cid;
            md.mat_c_dev[offset] = acc1;
            md.mat_c_dev[offset+32] = acc2;
        }
        else {
            int nout = (md.B_ncols-cid+31)/32;
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = md.A_csrVal[ptr];
                    colInd_sh[thread_idx] = md.B_ncols*md.A_csrColInd[ptr];
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
            offset = rid*md.B_ncols+cid;
            if (nout>0) {
            md.mat_c_dev[offset] = acc1;
            }
            if (nout>1) {
            md.mat_c_dev[(offset+32)] = acc2;
            }
        }
    }
}
