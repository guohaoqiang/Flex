#ifndef MAT_H
#define MAT_H 
#include <vector>
#include <iostream>
#include <algorithm>
#include "common.h"
#define DataType float
#define DEBUG
using namespace std;

//template<int TM, int TN>
class mat{
public:
	int m,n,k;
	int nnz;
	int pos;
	int tm,tn;
	std::vector<unsigned int>& rowPtr;
	std::vector<unsigned int>& colIdx;
	std::vector<DataType>& vals;

	mat(std::vector<unsigned int>& rowPtr, 
        std::vector<unsigned int>& colIdx, 
        std::vector<DataType>& vals, 
        int h, int w, int n,
        int tileh, int tilew);
	void print1();
	void print2();
    
	void csr2tile();

	std::vector<unsigned int> tileRowPtr;
	std::vector<unsigned int> nnzPtr;
	std::vector<unsigned int> tileLeftColIdx;
    std::vector<char> rc_Offset;
	std::vector<DataType> newVals;
    
    std::vector<unsigned int> rowOffset;
	std::vector<unsigned int> tileColIdx;

	vector<unsigned int> block_tileStart_idx;
	vector<unsigned int> warp_tileRow_idx;

    // v4  kernel
    vector<int> nnzTile;
    vector<int> bitMap;
    vector<int> rcOffset;

	// regular sparse-tile storage
	std::vector<unsigned int> rgl_tileRowPtr;
	std::vector<unsigned int> rgl_tileColIdx;
	std::vector<unsigned int> rgl_nnzPtr;
	std::vector<unsigned int> rgl_rowOffset;
	std::vector<unsigned int> rgl_colOffset;
	std::vector<DataType> rgl_newVals;

	void csr2regular(int i);
	void csr2flex(int i);
};

#endif /* MAT_H */
