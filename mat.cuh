#ifndef MAT_H
#define MAT_H 
#include <vector>
#include <queue>
#include <stack>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <stdio.h>
#include "common.h"
#include "DataLoader.cuh"
#define DEBUG
#define NNZ_LIMIT 128
using namespace std;
class Mat_POD{
    public:
	int m,n,k;
	int nnz;
	int tm,tn;
    int n_segs;
    int sms;
	
    int* tileNnz_dev;
    int* tileColIdx_dev;
    float* vals_dev;
    int* tileRowPtr_dev;
    int* nnzTile_dev;
    int* bitMap_dev;
    int* rcOffset_dev;
    int* voMp_dev;

    float* mat_b_dev;
    float* shadow_b_dev;
    float* mat_c_dev;
	
    int* segPtr_dev; 
	int* segVoMap_dev; 
	int* segNzRCIdx_dev; 
	int* segNzRowIdx_dev; 
	int* segNzColIdx_dev; 
	int* grouped_tailSeg_dev; 
	int* next_seg_dev; 
    // kernel v31
    int* seg_rowPtr_dev;
    float* segNzCV_dev;


    // ge-spmm
    unsigned int* csr_rowPtr_dev;
    unsigned int* csr_col_dev;
    float* csr_vals_dev;
    float* csr_mat_b_dev;

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
	void print3(int );
    void stats_collect(FILE *stream = nullptr);
    void stats_collect2(FILE *stream = nullptr);
    
	void csr2tile();
    void permute_segs();
    int checkSim(vector<int>&, vector<int>&);
    void sortSegs();

	std::vector<unsigned int> tileNnz;
	std::vector<unsigned int> tileColIdx;
	std::vector<float> newVals;
	std::vector<unsigned int> tileRowPtr; 
    std::vector<int> nnzTile;
    std::vector<int> bitMap;
    std::vector<int> rcOffset;
    std::vector<int> voMp;
    
	std::vector<unsigned int> segPtr; 
	std::vector<unsigned int> segVoMap; 
	std::vector<unsigned int> segNzRowIdx; 
	std::vector<unsigned int> segNzColIdx; 
	std::vector<unsigned int> segNzRCIdx; 
    std::vector<int> grouped_tailSeg;
    std::vector<int> next_seg;
    std::unordered_map<int,int> id2r;

    std::queue<pair<int,int>> aux_seg;
    std::unordered_map<int,int> count_segs;
    std::unordered_map<int,vector<int>> cols_seg;  // {seg_idx, colidx sequences}
   
    // kernel v31
    std::vector<int> seg_rowPtr;
    std::vector<float> segNzCV;

    int nnz_limit;
	int segPtr_bytes = 0; 
	int segVoMap_bytes = 0; 
	int segNzRowIdx_bytes = 0; 
	int segNzColIdx_bytes = 0; 
	int segNzRCIdx_bytes = 0; // == RowIdx + ColIdx
    int grouped_tailSeg_bytes;
    int next_seg_bytes;
    int64_t atomic_op;
    int seg_rowPtr_bytes;
    int segNzCV_bytes;
    
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
    int voMp_bytes;
    int mat_b_bytes; // == size of shadow_b_dev 
    int mat_c_bytes;

  // Statistics
  std::vector<uint> tile_p_row_histo;  // Histogram of number of tiles per row.
  std::vector<uint> tile_nnz_histo;    // Histogram of number of nz per tile.
  std::vector<uint> panel_lg_nnz_histo;
  std::vector<uint> seg_lg_nnz_histo;
  int64_t n_col_sum; // Sum of population of bitMaps == num nz cols in tiles.

  int row_nnz_get(int r) const { return rowPtr[r+1] - rowPtr[r]; }

	void csr2seg_Cmajor(int i);
	void csr2flex_Rmajor(int i);
	void csr2flex_Cmajor(int i);
    void transfer();
    void transfer2();
    void dataVolume_est();
    void dataVolume_est2();
    void launch_prep();

    
    void freeMatGPU2(){
      cuda_freez(segNzRowIdx_dev);
      cuda_freez(segNzColIdx_dev);
      cuda_freez(segNzRCIdx_dev);
      cuda_freez(segPtr_dev);
      cuda_freez(segVoMap_dev);
      cuda_freez(voMp_dev);
      cuda_freez(grouped_tailSeg_dev);
      cuda_freez(next_seg_dev);
      
      cuda_freez(segNzCV_dev);
      cuda_freez(seg_rowPtr_dev);
    }
    void freeMatGPU(){
      cuda_freez(tileNnz_dev);
      cuda_freez(tileColIdx_dev);
      cuda_freez(vals_dev); 
      cuda_freez(tileRowPtr_dev);
      cuda_freez(nnzTile_dev);
#ifndef COL_MAJ_TILE
      cuda_freez(bitMap_dev);
#endif
      cuda_freez(rcOffset_dev);
#ifdef VO_RECOVER
      cuda_freez(voMp_dev);
#endif
    }
    
};
class cmp{
    public:
    bool operator()(const pair<int,int>& a, const pair<int,int>& b){
        return a.second>b.second;
    }
};
#endif /* MAT_H */
