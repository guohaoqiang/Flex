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
    DataLoader& dl;
	std::vector<unsigned int>& rowPtr;
    std::vector<unsigned int>& colIdx;
	std::vector<float>& vals;

    Mat(DataLoader& mat, int tileh, int tilew);
	void print1();
	void print2();
    void stats_collect(bool print);
    
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

  // Statistics
  std::vector<uint> tile_p_row_histo;  // Histogram of number of tiles per row.
  std::vector<uint> tile_nnz_histo;    // Histogram of number of nz per tile.
  int64_t n_col_sum; // Sum of population of bitMaps == num nz cols in tiles.

	void csr2flex(int i);
    void transfer();
    void dataVolume_est();
    void launch_prep();

    
    void freeMatGPU(){
      cuda_freez(tileNnz_dev);
      cuda_freez(tileColIdx_dev);
      cuda_freez(vals_dev); 
      cuda_freez(tileRowPtr_dev);
      cuda_freez(nnzTile_dev);
      cuda_freez(bitMap_dev);
      cuda_freez(rcOffset_dev);
    }
    
};
#endif /* MAT_H */
