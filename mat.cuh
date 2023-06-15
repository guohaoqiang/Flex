#ifndef MAT_H
#define MAT_H 
#include <vector>
#include <iostream>
#include <algorithm>
#include "common.h"
#include "DataLoader.cuh"
#define DEBUG
using namespace std;
class Mat_POD{
    public:
	int m,n,k;
	int nnz;
	int tm,tn;
	
    int* tileNnz_dev;
    int* tileColIdx_dev;
    float* vals_dev;
    int* tileRowPtr_dev;
    int* nnzTile_dev;
    int* bitMap_dev;
    int* rcOffset_dev;

    float* mat_b_dev;
    float* mat_c_dev;
};
extern __constant__ Mat_POD mat_dev;
class Mat : public Mat_POD{
    public:
	int pos;
	std::vector<unsigned int>& rowPtr;
    std::vector<unsigned int>& colIdx;
	std::vector<float>& vals;

    Mat(DataLoader& mat, int tileh, int tilew);
	void print1();
	void print2();
    
	void csr2tile();

	std::vector<unsigned int> tileNnz;
	std::vector<unsigned int> tileColIdx;
	std::vector<float> newVals;
	std::vector<unsigned int> tileRowPtr; 
    std::vector<int> nnzTile;
    std::vector<int> bitMap;
    std::vector<int> rcOffset;
    
    int64_t est_fp = 0;
    int64_t est_ld_bytes = 0;
    int64_t est_st_bytes = 0;
    int tileNnz_bytes; 
    int tileColIdx_bytes; 
    int vals_bytes;
    int tileRowPtr_bytes;
    int nnzTile_bytes;
    int bitMap_bytes;
    int rcOffset_bytes;
    int mat_b_bytes;
    int mat_c_bytes;

	void csr2flex(int i);
    void transfer(float*);
    void dataVolume_est();
    void launch_prep();

    
    void freeMatGPU(){
        CHECK_CUDA(cudaFree(tileNnz_dev));
        CHECK_CUDA(cudaFree(tileColIdx_dev));
        CHECK_CUDA(cudaFree(vals_dev)); 
        CHECK_CUDA(cudaFree(tileRowPtr_dev));
        CHECK_CUDA(cudaFree(nnzTile_dev));
        CHECK_CUDA(cudaFree(bitMap_dev));
        CHECK_CUDA(cudaFree(rcOffset_dev));
        
        CHECK_CUDA(cudaFree(mat_b_dev));
        CHECK_CUDA(cudaFree(mat_c_dev));
    }
    
};
#endif /* MAT_H */
