#ifndef MAT_H
#define MAT_H 
#include <vector>
#include <iostream>
#include <algorithm>
#include "common.h"
#define DataType float
#define DEBUG
using namespace std;

template<int TM, int TN>
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
        int h, int w, int n);
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

template<int TM, int TN>
mat<TM,TN>::mat(std::vector<unsigned int>& r, 
		std::vector<unsigned int>& c, 
		std::vector<DataType>& v, 
		int h, 
		int kk,
		int nz):m(h),k(kk),nnz(nz),rowPtr(r),colIdx(c),vals(v){
            n = m;
			tm = TM,tn = TN;
			tileRowPtr.push_back(0);
			nnzPtr.push_back(0);
            // v4 kernel
            //metaTile.push_back(0);
            
            rc_Offset.resize(nnz);
            // v4 kernel
            //rcOffset.resize(nnz);
			newVals.resize(nnz);

			rowOffset.resize(nnz);
			tileColIdx.resize(nnz);
			// the length of nnzPtr is unknown so far

			pos = 0; 

			// regular sparse-tile storage
			rgl_tileRowPtr.push_back(0);
			rgl_nnzPtr.push_back(0);
			rgl_rowOffset.resize(nnz);
			rgl_colOffset.resize(nnz);
			rgl_newVals.resize(nnz);
}
template<int TM, int TN>
void mat<TM,TN>::print1(){
#ifdef DEBUG
	for (int i=0; i<rgl_tileRowPtr.size(); ++i)
		std::cout<<rgl_tileRowPtr[i]<<" ";
	std::cout<<std::endl;
	for (int i=0; i<rgl_tileColIdx.size(); ++i)
		std::cout<<rgl_tileColIdx[i]<<" ";
	std::cout<<std::endl;
	for (int i=0; i<rgl_nnzPtr.size(); ++i)
		std::cout<<rgl_nnzPtr[i]<<" ";
	std::cout<<std::endl;
	for (int i=0; i<rgl_rowOffset.size(); ++i)
		std::cout<<rgl_rowOffset[i]<<" ";
	std::cout<<std::endl;
	for (int i=0; i<rgl_colOffset.size(); ++i)
		std::cout<<rgl_colOffset[i]<<" ";
	std::cout<<std::endl;
	for (int i=0; i<rgl_newVals.size(); ++i)
		std::cout<<rgl_newVals[i]<<" ";
	std::cout<<std::endl;
#endif
	std::cout<<"Regular Tiles: "<<rgl_nnzPtr.size()-1<<std::endl;
}

template<int TM, int TN>
void mat<TM,TN>::print2(){
#ifdef DEBUG
    for (int i=0; i<tileRowPtr.size(); ++i)
		std::cout<<tileRowPtr[i]<<" ";
	std::cout<<std::endl;
    
	/*
	for (int i=0; i<nnzPtr.size(); ++i)
		std::cout<<nnzPtr[i]<<" ";
	std::cout<<std::endl;
	
    for (int i=0; i<tileLeftColIdx.size(); ++i)
		std::cout<<tileLeftColIdx[i]<<" ";
	std::cout<<std::endl;
    std::cout<<"------- tile elements: -------"<<std::endl;
	for (int i=0; i<rowOffset.size(); ++i)
		std::cout<<rowOffset[i]<<" ";
	std::cout<<std::endl;
	for (int i=0; i<tileColIdx.size(); ++i)
		std::cout<<tileColIdx[i]<<" ";
    std::cout<<std::endl<<"rc:"<<std::endl;
	for (int i=0; i<rc_Offset.size(); ++i)
		std::cout<<(int)rc_Offset[i]<<" ";
	std::cout<<std::endl;
	for (int i=0; i<newVals.size(); ++i)
		std::cout<<newVals[i]<<" ";
	std::cout<<std::endl;
    */
#endif
    std::cout<<std::endl<<"nnzTile:"<<std::endl;
    //for (int i=0; i<nnzTile.size(); ++i){
    for (int i=0; i<20; ++i){
        std::cout<<nnzTile[i]<<" ";
    }
    std::cout<<std::endl<<"bitMap:"<<std::endl;
    //for (int i=0; i<bitMap.size(); ++i){
    for (int i=0; i<20; ++i){
        std::cout<<bitMap[i]<<" ";
    }
    std::cout<<std::endl<<"rc:"<<std::endl;
	//for (int i=0; i<rcOffset.size(); ++i){
	for (int i=0; i<20; ++i){
		int r = rcOffset[i]>>16;
		int c = rcOffset[i] & 0x0000FFFF;
        std::cout<<"{"<<r<<","<<c<<"}"<<" ";
    }
	
//    std::cout<<std::endl<<"vals:"<<std::endl;
//	for (int i=0; i<newVals.size(); ++i)
//		std::cout<<newVals[i]<<" ";
	std::cout<<std::endl;
	std::cout<<"Flex Tiles: "<<nnzPtr.size()-1<<std::endl;
}

template<int TM, int TN>
void mat<TM,TN>::csr2tile(){
	
	int tileRows = (m+tm-1)/tm;
		
    //tileRowPtr.resize(tileRows+1);

    //std::cout<<"@98:"<<tileRows<<std::endl;
	for (int i=0; i<tileRows; ++i){
		csr2flex(i);
		//csr2regular(i);
	}
    
	// workload_per_block: tiles assigned to a block
	uint32_t workload_per_block = 4;
    // block_tileStart_idx[i]: the idx of the begining tile assigned to the i-th block
	// the last one is the totall number of tiles
	for (uint32_t i=0; i<tileRows; ++i){
		uint32_t blocks_req = (tileRowPtr[i+1] - tileRowPtr[i] + workload_per_block-1)/workload_per_block;
		for (uint32_t ki=0; ki<blocks_req; ++ki){
			block_tileStart_idx.push_back(tileRowPtr[i] + ki*workload_per_block);
			// tile row idx for tiles assigned to each thread block
			warp_tileRow_idx.push_back(i*tm);
		}
	}
	block_tileStart_idx.push_back(tileRowPtr.back());
}
// convert a row of tiles to FlexSpTiles
template<int TM, int TN>
void mat<TM,TN>::csr2flex(int ridx){
	// row tile upper bound and lower bound
	int rowStart = ridx * tm;
	int rowEnd = min(m, (ridx+1)*tm); // exclusive

	// keep track of the cols in each row
	std::vector<unsigned int> cIdx(tm, -1); 
	std::vector<unsigned int> cOffset(tm, 0);
	// get the left bound
	// iterate over rows to get the smallest col idx
	unsigned int left = n;
	for (int i=rowStart; i<rowEnd; ++i){
		// here, we assume there is no empty row
		left = min((int)left, (int)colIdx[rowPtr[i]]);
		cIdx[i-rowStart] = colIdx[rowPtr[i]];
	}

	// right bound (exclusive)
	unsigned int right = min((int)left + tn, n);
	int nnzInRows = 0;
    int tiles_in_cur_row = 0;

    int tileStart = rowPtr[ridx];
    while (pos<rowPtr[rowEnd]){
		int nnzInTile = 0;
        tiles_in_cur_row++;
		// collect tiles in the tile-row
        int bit_map = 0;
		for (int i=rowStart; i<rowEnd; ++i){
			// absolute position of the nze in csr, idx = base + offset
			int c = rowPtr[i] + cOffset[i-rowStart];
			//  #nze in the i-th row
			
			// c check is necessary because it constraines nze within the i-th row
			while (c<rowPtr[i+1] && colIdx[c]>=left && colIdx[c]<right){
                char rc = 0;
                int rc16 = 0;

				// currently, it is not 4-bit
				rowOffset[pos] = i-rowStart;
                rc |= (rowOffset[pos]<<4);
                rc16 |= (rowOffset[pos]<<16);

				// real col idx
				tileColIdx[pos] = cIdx[i-rowStart];
                rc |= (tileColIdx[pos]-left);
                rc16 |= (tileColIdx[pos]-left);
			    bit_map |= 1<<(tileColIdx[pos]-left);	
                // nze values
				newVals[pos] = vals[c];
                rc_Offset[pos] = rc;
                //rcOffset[pos] = rc16;
                rcOffset.push_back(rc16);

				cIdx[i-rowStart] = colIdx[++c];
				pos++;
				cOffset[i-rowStart]++;
				nnzInTile++;
				nnzInRows++;
			}
		}
        
        // ---------- v4 -------
        tileStart = nnzPtr.back()+nnzInTile; 
        // mark the last tile in current row-tile
        //if (pos>=rowPtr[rowEnd]){
        //    nnzInTile |= (1<<31);
        //}
        nnzTile.push_back(nnzInTile); 
        bitMap.push_back(bit_map); 
        // ---------------------
		
		nnzPtr.push_back(nnzPtr.back()+nnzInTile);
        tileLeftColIdx.push_back(left);
        // update left and right bound for next tile
		left = n;
		for (int i=rowStart; i<rowEnd; ++i){
			// check whether the column goes to the next row
			int rnnz = rowPtr[i+1]-rowPtr[i];
			if (cOffset[i-rowStart]<rnnz){
				left = min((int)left, (int)cIdx[i-rowStart]);
			}
		}
		right = min((int)left + tn, n);

	}
	//tileRowPtr.push_back(tileRowPtr.back()+nnzInRows);
	tileRowPtr.push_back(tileRowPtr.back()+tiles_in_cur_row);
}
// convert a row of tiles to regular tiles
template<int TM, int TN>
void mat<TM,TN>::csr2regular(int ridx){
	// row tile upper bound and lower bound
	int rowStart = ridx * tm;
	int rowEnd = min(m, (ridx+1)*tm); // exclusive

	// keep track of the cols in each row
	std::vector<unsigned int> cIdx(tm, -1); 
	std::vector<unsigned int> cOffset(tm, 0);
	
	unsigned int left = 0;
	for (int i=rowStart; i<rowEnd; ++i){
		cIdx[i-rowStart] = colIdx[rowPtr[i]];
	}

	// right bound (exclusive)
	unsigned int right = min((int)left + tn, n);
	int nnzInRows = 0;
	while (pos<rowPtr[rowEnd]){
		int nnzInTile = 0;
		int tileIndex = -1;
		// collect tiles in the tile-row
		for (int i=rowStart; i<rowEnd; ++i){
			// absolute position of the nze in csr, idx = base + offset
			unsigned int c = rowPtr[i] + cOffset[i-rowStart];
			//  #nze in the i-th row
			
			// c check is necessary because it constraines nze within the i-th row
			while (c<rowPtr[i+1] && colIdx[c]>=left && colIdx[c]<right){
				tileIndex = colIdx[c]/tn;
				// currently, it is not 4-bit
				rgl_rowOffset[pos] = i-rowStart;

				// relative col idx
				rgl_colOffset[pos] = cIdx[i-rowStart]-left;
				// nze values
				rgl_newVals[pos] = vals[c];

				cIdx[i-rowStart] = colIdx[++c];
				pos++;
				cOffset[i-rowStart]++;
				nnzInTile++;
				nnzInRows++;
			}
		}
		if (nnzInTile) rgl_nnzPtr.push_back(rgl_nnzPtr.back()+nnzInTile);
		if (tileIndex!=-1) rgl_tileColIdx.push_back(tileIndex);
		// update left and right bound for next tile
		left += tn;
		right = min((int)left + tn, n);
	}
	rgl_tileRowPtr.push_back(rgl_tileRowPtr.back()+nnzInRows);
}
#endif /* MAT_H */
