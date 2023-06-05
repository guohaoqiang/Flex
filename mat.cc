#include "mat.h"
/*
mat::mat(std::vector<unsigned int>& r, 
		std::vector<unsigned int>& c, 
		std::vector<DataType>& v, 
		int h, 
		int kk,
		int nz):m(h),k(kk),nnz(nz),rowPtr(r),colIdx(c),vals(v){
            n = m;
			tm = TM,tn = TN;
			tileRowPtr.push_back(0);
			nnzPtr.push_back(0);
            
            rc_Offset.resize(nnz);
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
void mat::print1(){
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
void mat::print2(){
#ifdef DEBUG
	for (int i=0; i<tileRowPtr.size(); ++i)
		std::cout<<tileRowPtr[i]<<" ";
	std::cout<<std::endl;
    //
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
    //
#endif
	std::cout<<"Flex Tiles: "<<nnzPtr.size()-1<<std::endl;
}
// inputs: 
//      CSR (rowPtr, colIdx, val), 
//		tile_size (tm, tn), 
//		
// outputs:
//		 tileRowPtr: the tile idex of the first tile in each tile row. Like rowPtr
//       nnzPtr: the index of the first nze(non-zero entry) in each tile
//       rowOffset(4 bit): the row index of each nze in each tile
//       tileColIdx: the col index of each nze in each tile
// The real row-col index for a nze (the i-th row tile, the k-th nze): 
//  r = tileRowPtr[i] + rowOffset[k], c = colIdx[k]
void mat::csr2tile(){
	
	int tileRows = (m+TM-1)/TM;
	//tileRowPtr.resize(tileRows+1);

    //std::cout<<"@98:"<<tileRows<<std::endl;
	for (int i=0; i<tileRows; ++i){
		csr2flex(i);
		//csr2regular(i);
	}
}
// convert a row of tiles to regular tiles
void mat::csr2regular(int ridx){
	// row tile upper bound and lower bound
	int rowStart = ridx * TM;
	int rowEnd = min(m, (ridx+1)*TM); // exclusive

	// keep track of the cols in each row
	std::vector<unsigned int> cIdx(TM, -1); 
	std::vector<unsigned int> cOffset(TM, 0);
	
	unsigned int left = 0;
	for (int i=rowStart; i<rowEnd; ++i){
		cIdx[i-rowStart] = colIdx[rowPtr[i]];
	}

	// right bound (exclusive)
	unsigned int right = min((int)left + TN, n);
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
				tileIndex = colIdx[c]/TN;
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
		left += TN;
		right = min((int)left + TN, n);
	}
	rgl_tileRowPtr.push_back(rgl_tileRowPtr.back()+nnzInRows);
}
// convert a row of tiles to FlexSpTiles
void mat::csr2flex(int ridx){
	// row tile upper bound and lower bound
	int rowStart = ridx * TM;
	int rowEnd = min(m, (ridx+1)*TM); // exclusive

	// keep track of the cols in each row
	std::vector<unsigned int> cIdx(TM, -1); 
	std::vector<unsigned int> cOffset(TM, 0);
	// get the left bound
	// iterate over rows to get the smallest col idx
	unsigned int left = n;
	for (int i=rowStart; i<rowEnd; ++i){
		// here, we assume there is no empty row
		left = min((int)left, (int)colIdx[rowPtr[i]]);
		cIdx[i-rowStart] = colIdx[rowPtr[i]];
	}

	// right bound (exclusive)
	unsigned int right = min((int)left + TN, n);
	int nnzInRows = 0;
    int tiles_in_cur_row = 0;
	while (pos<rowPtr[rowEnd]){
		int nnzInTile = 0;
        tiles_in_cur_row++;
		// collect tiles in the tile-row
		for (int i=rowStart; i<rowEnd; ++i){
			// absolute position of the nze in csr, idx = base + offset
			int c = rowPtr[i] + cOffset[i-rowStart];
			//  #nze in the i-th row
			
			// c check is necessary because it constraines nze within the i-th row
			while (c<rowPtr[i+1] && colIdx[c]>=left && colIdx[c]<right){
                char rc = 0;
				// currently, it is not 4-bit
				rowOffset[pos] = i-rowStart;
                rc |= (rowOffset[pos]<<4);

				// real col idx
				tileColIdx[pos] = cIdx[i-rowStart];
                rc |= (tileColIdx[pos]-left);
				// nze values
				newVals[pos] = vals[c];
                rc_Offset[pos] = rc;

				cIdx[i-rowStart] = colIdx[++c];
				pos++;
				cOffset[i-rowStart]++;
				nnzInTile++;
				nnzInRows++;
			}
		}
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
		right = min((int)left + TN, n);
	}
	//tileRowPtr.push_back(tileRowPtr.back()+nnzInRows);
	tileRowPtr.push_back(tileRowPtr.back()+tiles_in_cur_row);
}
*/
