#include "mat.cuh"
#include <bit>

__constant__ Mat_POD mat_dev;

Mat::Mat(DataLoader& input, int tileh,int tilew)
         :dl(input),rowPtr(input.rowPtr),colIdx(input.col),vals(input.vals){
            m = input.n;
            n = m;
            k = input.dim;
            nnz = input.nnz;
			tm = tileh;
            tn = tilew;
			tileRowPtr.push_back(0);
			tileNnz.push_back(0);
			newVals.resize(input.nnz);
			pos = 0;
            bitMap_bytes = 0; 
}
void Mat::launch_prep(){
    dl.gpuC_zero();
    mat_b_dev = dl.gpuX;
    mat_c_dev = dl.gpuC;
    Mat_POD for_dev(*this);
    cudaMemcpyToSymbol(mat_dev, &for_dev, sizeof(for_dev), 0, cudaMemcpyHostToDevice);
}
void Mat::transfer(){
#   define CMALC(var)                                   \
     var##_bytes = var.size() * sizeof( var[0] );        \
     CHECK_CUDA(cudaMalloc( &var##_dev, var##_bytes )) ;

     CMALC( tileNnz ); CMALC( tileColIdx ); CMALC( vals );
     CMALC( tileRowPtr ); CMALC( nnzTile ); CMALC( rcOffset );
#ifndef COL_MAJ_TILE
CMALC( bitMap );
#endif
#   undef CMALC

    // transfer data to device
    cudaMemcpy(tileNnz_dev, tileNnz.data(), tileNnz.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tileColIdx_dev, tileColIdx.data(), tileColIdx.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(vals_dev, newVals.data(), newVals.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(tileRowPtr_dev, tileRowPtr.data(), tileRowPtr.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(nnzTile_dev, nnzTile.data(), nnzTile.size()*sizeof(int), cudaMemcpyHostToDevice);
#ifndef COL_MAJ_TILE
    cudaMemcpy(bitMap_dev, bitMap.data(), bitMap.size()*sizeof(int), cudaMemcpyHostToDevice);
#endif
    cudaMemcpy(rcOffset_dev, rcOffset.data(), rcOffset.size()*sizeof(int), cudaMemcpyHostToDevice);
}
void Mat::dataVolume_est(){
    est_fp = int64_t(nnz)*k;
    est_ld_bytes = int64_t(tileNnz_bytes) + 
                    tileColIdx_bytes + 
                    vals_bytes + 
                    dl.gpuX_bytes +
                    tileRowPtr_bytes + 
                    nnzTile_bytes + 
                    bitMap_bytes + 
                    rcOffset_bytes;
    est_st_bytes = dl.gpuC_bytes;
}

void Mat::csr2tile(){
	
	int tileRows = (m+tm-1)/tm;
		
	for (int i=0; i<tileRows; ++i){
		csr2flex_Rmajor(i);
		//csr2flex_Cmajor(i);
		//csr2regular(i);
	} 
}

// convert a row of tiles to FlexSpTiles
void Mat::csr2flex_Rmajor(int ridx){
	// row tile upper bound and lower bound
	int rowStart = ridx * tm;
	int rowEnd = min(m, (ridx+1)*tm); // exclusive

        const int n_tiles_limit = ( n + tn - 1 ) / tn;

	// keep track of the cols in each row
	std::vector<int> cIdx(tm, -1); 
	std::vector<int> cOffset(tm, 0);
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

    //int tileStart = rowPtr[ridx];
    while (pos<rowPtr[rowEnd]){
		int nnzInTile = 0;
        tiles_in_cur_row++;
        assert( tiles_in_cur_row <= n_tiles_limit );
		// collect tiles in the tile-row
        int bit_map = 0;
		for (int i=rowStart; i<rowEnd; ++i){
			// absolute position of the nze in csr, idx = base + offset
			int c = rowPtr[i] + cOffset[i-rowStart];
			//  #nze in the i-th row
			
			// c check is necessary because it constraines nze within the i-th row
                        while ( c<rowPtr[i+1] && colIdx[c]<right ){
                //char rc = 0;
                int rc16 = 0;

				// currently, it is not 4-bit
				int temp_rowOffset = i-rowStart;
                //rc |= (temp_rowOffset<<4);
                rc16 |= (temp_rowOffset<<16);

				// real col idx
				int temp_tileColIdx = cIdx[i-rowStart];
                                assert( temp_tileColIdx >= left );
                //rc |= (temp_tileColIdx-left);
                rc16 |= (temp_tileColIdx-left);
			    bit_map |= 1<<(temp_tileColIdx-left);	
                
                // nze values
				newVals[pos] = vals[c];
                rcOffset.push_back(rc16);

				cIdx[i-rowStart] = colIdx[++c];
				pos++;
				cOffset[i-rowStart]++;
				nnzInTile++;
				nnzInRows++;
			}
		}
        
        // ---------- v4 -------
        //tileStart = tileNnz.back()+nnzInTile; 
        // mark the last tile in current row-tile
        //if (pos>=rowPtr[rowEnd]){
        //    nnzInTile |= (1<<31);
        //}
        nnzTile.push_back(nnzInTile); 
        bitMap.push_back(bit_map); 
        // ---------------------
		
		tileNnz.push_back(tileNnz.back()+nnzInTile);
        tileColIdx.push_back(left);
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
	tileRowPtr.push_back(tileRowPtr.back()+tiles_in_cur_row);
}

// convert a row of tiles to FlexSpTiles
void Mat::csr2flex_Cmajor(int ridx){
	// row tile upper bound and lower bound
	int rowStart = ridx * tm;
	int rowEnd = min(m, (ridx+1)*tm); // exclusive

    const int n_tiles_limit = ( n + tn - 1 ) / tn;

	// keep track of the cols in each row
	std::vector<int> cIdx(tm, -1); 
	std::vector<int> cOffset(tm, 0);
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

    //int tileStart = rowPtr[ridx];
    while (pos<rowPtr[rowEnd]){
		int nnzInTile = 0;
        tiles_in_cur_row++;
        assert( tiles_in_cur_row <= n_tiles_limit );
		// collect tiles in the tile-row
        int bit_map = 0;
        for (int i_tn=0; i_tn<tn; ++i_tn){
            
		    for (int i=rowStart; i<rowEnd; ++i){
                // absolute position of the nze in csr, idx = base + offset
                int c = rowPtr[i] + cOffset[i-rowStart];
                if ( colIdx[c]==left+i_tn && c<rowPtr[i+1] && colIdx[c]<right ){
                    int rc16 = 0;

                    // currently, it is not 4-bit
                    int temp_rowOffset = i-rowStart;
                    //rc |= (temp_rowOffset<<4);
                    rc16 |= (temp_rowOffset<<16);

                    // real col idx
                    int temp_tileColIdx = cIdx[i-rowStart];
                    assert( temp_tileColIdx >= left );
                    //rc |= (temp_tileColIdx-left);
                    rc16 |= (temp_tileColIdx-left);
			        bit_map |= 1<<(temp_tileColIdx-left);	
                    
                    // nze values
                    newVals[pos] = vals[c];
                    rcOffset.push_back(rc16);

                    cIdx[i-rowStart] = colIdx[++c];
                    pos++;
                    cOffset[i-rowStart]++;
                    nnzInTile++;
                    nnzInRows++; 
                }
            }
        }
        nnzTile.push_back(nnzInTile); 	
        bitMap.push_back(bit_map); 
		
        tileNnz.push_back(tileNnz.back()+nnzInTile);
        tileColIdx.push_back(left);
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
	tileRowPtr.push_back(tileRowPtr.back()+tiles_in_cur_row);
}

void
Mat::stats_collect(bool print)
{
  const uint tmn = tm * tn;

  const uint tile_m = ( m + tm - 1 ) / tm;
  const uint tile_n = ( n + tn - 1 ) / tn;
  tile_p_row_histo.resize(tile_n+1);
  uint max_n_tiles = 0;
  for ( uint tile_r = 0;  tile_r < tile_m;  tile_r++ )
    {
      const uint n_tiles = tileRowPtr[tile_r+1] - tileRowPtr[tile_r];
      assert( n_tiles <= tile_n );
      set_max( max_n_tiles, n_tiles );
      tile_p_row_histo[n_tiles]++;
    }
  tile_p_row_histo.resize(max_n_tiles+1);
  const uint n_tiles = nnzTile.size();

  tile_nnz_histo.resize(tmn+1);
  n_col_sum = 0;
  uint max_t_nnz = 0;
  for ( uint t_idx = 0; t_idx < n_tiles; t_idx++ )
    {
      const auto nnz = nnzTile[t_idx];
      tile_nnz_histo[nnz]++;
      set_max( max_t_nnz, nnz );
      const uint n_col = popcount(uint(bitMap[t_idx]));
      n_col_sum += n_col;
    }
  tile_nnz_histo.resize(max_t_nnz+1);

  vector<int> tiles_bucket(6,0);
  int total = 0;
  int remain = 0;
  for (int i=0; i<tile_p_row_histo.size(); ++i){
    int counts = tile_p_row_histo[i]; 
    if (counts>=1 && counts<8){
        tiles_bucket[0] += counts;
    }else if (counts>=8 && counts<16){
        tiles_bucket[1] += counts;
    }else if (counts>=16 && counts<32){
        tiles_bucket[2] += counts;
    }else if (counts>=32 && counts<64){
        tiles_bucket[3] += counts;
    }else if (counts>=64 && counts<128){
        tiles_bucket[4] += counts;
    }else if (counts>=128){
        tiles_bucket[5] += counts;
    }else{
        remain += counts;
    }
    total += counts;
  }
  if ( !print ) return;
  printf("[1,8): %f%%   ", tiles_bucket[0]*100.0/tile_m); 
  printf("[8,16): %f%%    ", tiles_bucket[1]*100.0/tile_m); 
  printf("[16,32): %f%%   ", tiles_bucket[2]*100.0/tile_m); 
  printf("[32,64): %f%%   ", tiles_bucket[3]*100.0/tile_m); 
  printf("[64,128): %f%%  ", tiles_bucket[4]*100.0/tile_m); 
  printf("[128, +OO): %f%%\n", tiles_bucket[5]*100.0/tile_m); 

  printf("Arrays m=%d, n=%d, k=%d. Tile %d × %d.   nnz=%d  Avg deg=%.1f\n",
         m, n, k, tm, tn, nnz, double(nnz)/m);

  printf("nnz / tile: %.3f  Load / B elt  %.3f\n",
         double(nnz) / n_tiles,
         n_col_sum / double(nnz) );
  int n_t_hist_pr = 0;
  printf("Tile nnz histogram: (n_tiles %d)\n",n_tiles);
  for ( uint i=0; i<tile_nnz_histo.size(); i++ )
    if ( auto tnnz = tile_nnz_histo[i]; tnnz )
      {
        if ( n_t_hist_pr++ > 6 ) break;
        printf("%3d %5.2f%%, ", i, tnnz * 100.0 / n_tiles);
      }
  printf("\n");
}


void Mat::print2(){
#ifdef DEBUG
    for (int i=0; i<tileRowPtr.size(); ++i)
		std::cout<<tileRowPtr[i]<<" ";
	std::cout<<std::endl;
    
	/*	
    for (int i=0; i<tileLeftColIdx.size(); ++i)
		std::cout<<tileLeftColIdx[i]<<" ";
	std::cout<<std::endl;
    std::cout<<"------- tile elements: -------"<<std::endl;
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
    for (int i=0; i<nnzTile.size(); ++i){
    //for (int i=0; i<20; ++i){
        std::cout<<nnzTile[i]<<" ";
    }
#ifndef COL_MAJ_TILE
    std::cout<<std::endl<<"bitMap:"<<std::endl;
    //for (int i=0; i<bitMap.size(); ++i){
    for (int i=0; i<20; ++i){
        std::cout<<bitMap[i]<<" ";
    }
#endif
    std::cout<<std::endl<<"rc:"<<std::endl;
	for (int i=0; i<rcOffset.size(); ++i){
	//for (int i=0; i<20; ++i){
		int r = rcOffset[i]>>16;
		int c = rcOffset[i] & 0x0000FFFF;
        std::cout<<"{"<<r<<","<<c<<"}"<<" ";
    }
	
//    std::cout<<std::endl<<"vals:"<<std::endl;
//	for (int i=0; i<newVals.size(); ++i)
//		std::cout<<newVals[i]<<" ";
	std::cout<<std::endl;
	std::cout<<"Flex Tiles: "<<tileNnz.size()-1<<std::endl;
}
