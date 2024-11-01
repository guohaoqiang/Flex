#include "mat.cuh"
#include <bit>
#include <ranges>

__constant__ Mat_POD mat_dev;

Mat::Mat(DataLoader& input, int tileh,int tilew)
         :dl(input),rowPtr(input.rowPtr),colIdx(input.col),vals(input.vals),voMp(input.vo_mp){
            m = input.n;
            n = m;
            k = input.dim;
            nnz = input.nnz;
			tm = tileh;
            tn = tilew;
            uni_nb = input.uni_nb;
			tileRowPtr.push_back(0);
			segPtr.push_back(0);
			tileNnz.push_back(0);
			newVals.resize(input.nnz);
			pos = 0;
            bitMap_bytes = 0; 
            voMp_bytes = 0; 
            nnz_limit = NNZ_LIMIT;
            atomic_op = 0;


            csr_rowPtr_dev = dl.rowPtr_dev;
            csr_col_dev = dl.col_dev;
            csr_vals_dev = dl.vals_dev;
            csr_mat_b_dev = dl.gpuX;
}
void Mat::launch_prep(){
    dl.gpuC_zero();
    mat_b_dev = dl.gpuX;
    if (dl.vertex_order_abbr == "OVO"){
       shadow_b_dev = dl.gpuX; 
    }
    mat_c_dev = dl.gpuC;
    Mat_POD for_dev(*this);
    CHECK_CUDA(cudaMemcpyToSymbol(mat_dev, &for_dev, sizeof(for_dev), 0, cudaMemcpyHostToDevice));
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
#ifdef VO_RECOVER
CMALC( voMp );
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
#ifdef VO_RECOVER
    cudaMemcpy(voMp_dev, voMp.data(), voMp.size()*sizeof(int), cudaMemcpyHostToDevice);
    if (dl.vertex_order_abbr != "OVO"){
        CHECK_CUDA(cudaMalloc( &shadow_b_dev,  m*k*sizeof(float))) ;
        CHECK_CUDA(cudaMemset( shadow_b_dev,  0, m*k*sizeof(float))) ;
    }
#endif
}
void Mat::transfer2(){
#   define CMALC(var)                                   \
     var##_bytes = var.size() * sizeof( var[0] );        \
     CHECK_CUDA(cudaMalloc( &var##_dev, var##_bytes )) ;

     CMALC( segPtr ); CMALC( segNzRCIdx ); CMALC( segNzRowIdx ); CMALC( segNzColIdx ); 
     CMALC( vals ); CMALC( voMp ); CMALC( segVoMap ); 
     CMALC( grouped_tailSeg ); CMALC( next_seg );
     CMALC( seg_rowPtr ); CMALC( segNzCV );
#   undef CMALC

    // transfer data to device
    cudaMemcpy(segNzRCIdx_dev, segNzRCIdx.data(), segNzRCIdx.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(segNzRowIdx_dev, segNzRowIdx.data(), segNzRowIdx.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(segNzColIdx_dev, segNzColIdx.data(), segNzColIdx.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(vals_dev, newVals.data(), newVals.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(segPtr_dev, segPtr.data(), segPtr.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(segVoMap_dev, segVoMap.data(), segVoMap.size()*sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMemcpy(seg_rowPtr_dev, seg_rowPtr.data(), seg_rowPtr.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(segNzCV_dev, segNzCV.data(), segNzCV.size()*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(voMp_dev, voMp.data(), voMp.size()*sizeof(int), cudaMemcpyHostToDevice);
    if (dl.vertex_order_abbr != "OVO"){
        CHECK_CUDA(cudaMalloc( &shadow_b_dev,  m*k*sizeof(float))) ;
        CHECK_CUDA(cudaMemset( shadow_b_dev,  0, m*k*sizeof(float))) ;
    }
    cudaMemcpy(grouped_tailSeg_dev, grouped_tailSeg.data(), grouped_tailSeg.size()*sizeof(int), cudaMemcpyHostToDevice);
}
void Mat::dataVolume_est2(){
  
/*********** compute B rows to be loaded for each sm **************************/
    int64_t validate_nnz = 0; 
    vector<unordered_set<int>> col_st(sms+1, unordered_set<int>());  
    unordered_map<int,int> c_sm;
    // the first SM buckets 
    for (int i=0; i<sms; ++i){
        
        if ( next_seg[ i ]<grouped_tailSeg[ i ] ){
            for (int j=seg_rowPtr[ next_seg[ i ]*(tm+1) ]; 
                    j<seg_rowPtr[ grouped_tailSeg[ i ]*(tm+1) -1]; ++j){
                validate_nnz++;
                col_st[i].insert( (int)segNzCV[j*2] );
            }
        }
        // collect long row break and draw the pie
        for ( auto &cc: col_st[i] ){
            c_sm[cc]++;
        }
    }
    
    // the last bucket, which is used for workload balance
    acc_col = 0;
    for (int i=next_seg[sms]; i<n_segs; ++i){
       unordered_set<int> last_tile_col;
       for (int ii=0; ii<tm; ++ii) 
        for (int j=seg_rowPtr[ i*(tm+1)+ii ]; 
                j<seg_rowPtr[ i*(tm+1)+ii+1 ]; ++j){
            validate_nnz++;
            col_st[sms].insert( (int)segNzCV[j*2] );
            last_tile_col.insert( (int)segNzCV[j*2] );
        }

       acc_col += last_tile_col.size();
        for ( auto &cc: col_st[sms] ){
            c_sm[cc]++;
        }
    }

    if (false){
        bool draw_pie = true;
        int pie[6] = { 0 };
        const char* pie_sm = "c_sm.csv";
        FILE *pie_c_sm = fopen(pie_sm,"aw");
        if (draw_pie){
            for ( auto &p:c_sm ){
                if ( p.second==1 ){
                    pie[0]++;
                }else if ( p.second==2 ){
                    pie[1]++;
                }else if ( p.second==3 ){
                    pie[2]++;
                }else if ( p.second>3 && p.second<=5 ){
                    pie[3]++;
                }else if ( p.second>5 && p.second<=10 ){
                    pie[4]++;
                }else{
                    pie[5]++;
                }
            }
        }
        fprintf(pie_c_sm, "%ld,",uni_nb);
        for (int ii=0; ii<6; ++ii){
            if (ii<5) fprintf(pie_c_sm, "%d,",pie[ii]);
            else fprintf(pie_c_sm, "%d\n",pie[ii]);
        }
        fclose(pie_c_sm);
    }

    bool collect_b_loads = false;
    bool collect_ops = true;
    bool collect_tile_alloc = false;
    const char* l1cache = "l1cache.csv";
    if (collect_b_loads){
        l1cache = "b_loads_per_sm.csv";
    }
    if (collect_ops){
        l1cache = "ops_per_sm.csv";
    }
    if (collect_tile_alloc){
        l1cache = "tiles_per_sm.csv";
    }
    FILE *l1_est = fopen(l1cache,"aw");
    fprintf(l1_est,"%s,",dl.graph_name.c_str());
    fprintf(l1_est,"%s,",dl.vertex_order_abbr.c_str());
    fprintf(l1_est,"%d\n",tm);
    if (validate_nnz!=nnz){
        printf("%d of %s, val = %d, nnz = %d\n",__LINE__,__FILE__,(int)validate_nnz,(int)nnz);
    }
    assert(validate_nnz==nnz);
    int validate_segs = n_segs;
    for (int j=0; j<sms; ++j){
        acc_col += col_st[j].size();
        if (collect_b_loads){
            fprintf(l1_est,"%d,", (int)col_st[j].size());
        }
        if (collect_ops){
            int nnz_in_sm = 0;
            if ( next_seg[ j ]<grouped_tailSeg[ j ] ){
                nnz_in_sm = seg_rowPtr[grouped_tailSeg[j]*(tm+1)-1] - seg_rowPtr[next_seg[j]*(tm+1)];
                int tiles_in_sm = grouped_tailSeg[j] - next_seg[j];
                assert(tiles_in_sm>0);
            }
            fprintf(l1_est,"%d,", nnz_in_sm);
        }
        if (collect_tile_alloc){
            fprintf(l1_est,"%d,", grouped_tailSeg[j]-next_seg[j]);
            validate_segs -= (grouped_tailSeg[j]-next_seg[j]);
        }
    }
    if (collect_b_loads){
        fprintf(l1_est,"%d\n", (int)col_st[ sms ].size());
    }
    if(collect_ops){
        if ( next_seg[ sms ]<grouped_tailSeg[ sms ] ){
            fprintf(l1_est,"%d\n", seg_rowPtr[grouped_tailSeg[sms]*(tm+1)-1]-
                                                seg_rowPtr[next_seg[sms]*(tm+1)] );
        }else{
            fprintf(l1_est,"%d\n", 0);
        }
    }
    if (collect_tile_alloc){
        fprintf(l1_est,"%d\n", grouped_tailSeg[ sms ]-next_seg[ sms ]);
        validate_segs -= (grouped_tailSeg[ sms ]-next_seg[ sms ]);
        assert( validate_segs==0 );
    }
/******************************************************************************/

    est_fp = int64_t(nnz)*k;
    // shadow_b_bytes is identical to gpuX_bytes when perform v9
    // so dl.gpuX_bytes can be seen shadow_b_bytes when v9

    //int64_t est_ld_bytes1 = int64_t(segNzRowIdx_bytes) + segNzColIdx_bytes + 
    //                vals_bytes + segPtr_bytes; 
    
    int64_t est_ld_bytes2 = int64_t(segNzCV_bytes) + seg_rowPtr_bytes;
    
    raw_ld_bytes = vals_bytes +
                   dl.gpuX_bytes;

    est_ld_bytes = est_ld_bytes2 +
                   segVoMap_bytes +
                   dl.gpuX_bytes +
                  grouped_tailSeg_bytes +
                   next_seg_bytes;    
        
    est_ld_bytes_tiling_ideal = est_ld_bytes2 +
                                segVoMap_bytes +
                                n_col_sum*k*4 +
                                grouped_tailSeg_bytes +
                                next_seg_bytes;    
    
    est_ld_bytes_tiling_sm_ideal = est_ld_bytes2 +
                                   segVoMap_bytes +
                                   acc_col*k*4 +
                                   grouped_tailSeg_bytes +
                                   next_seg_bytes;    

    // acc_col should be less than n_col_sum
    if (false)  printf("%d of %s, n_col_sum = %ld, acc_col = %ld\n",__LINE__,__FILE__,n_col_sum,acc_col);
    
    est_st_bytes = dl.gpuC_bytes;
}
void Mat::dataVolume_est(){
    est_fp = int64_t(nnz)*k;
    // shadow_b_bytes is identical to gpuX_bytes when perform v9
    // so dl.gpuX_bytes can be seen shadow_b_bytes when v9
    est_ld_bytes = int64_t(tileNnz_bytes) + 
                    tileColIdx_bytes + 
                    vals_bytes + 
                    dl.gpuX_bytes +
                    tileRowPtr_bytes + 
                    nnzTile_bytes + 
                    bitMap_bytes + 
                    rcOffset_bytes +
                    voMp_bytes;
    est_st_bytes = dl.gpuC_bytes;
}
void Mat::permute_segs(){
	std::vector<unsigned int> segPtr1(1,0);
	std::vector<unsigned int> segNzRCIdx1;
	std::vector<float> newVals1;
	std::vector<unsigned int> segVoMap1;

    // {seg row_idx,seg_idx}
	std::pair last{-1,-1};

	while (!aux_seg.empty()){
		auto top = aux_seg.front();
		aux_seg.pop();
		
		if ( count_segs[top.first] != aux_seg.size()+1 && last.first!=-1 && top.first == last.first ){
			aux_seg.push(top);
		}else{
            count_segs[top.first]--;
			int seg_idx = top.second;
            int seg_nnz = segPtr[seg_idx+1] - segPtr[seg_idx];
			for (int i=segPtr[seg_idx]; i<segPtr[seg_idx+1]; ++i){
				segNzRCIdx1.push_back(segNzRCIdx[2*i]);
				segNzRCIdx1.push_back(segNzRCIdx[2*i+1]);
				newVals1.push_back(newVals[i]);
			}
			segPtr1.push_back(segPtr1.back()+seg_nnz);
			for (int i=0; i<tm; ++i){
				segVoMap1.push_back(segVoMap[seg_idx*tm+i]);
			}

			last = top;
		}
	}
	swap(segPtr, segPtr1);
	swap(segNzRCIdx, segNzRCIdx1);
	swap(newVals, newVals1);
	swap(segVoMap, segVoMap1);
	return ;
}
int Mat::checkSim(vector<int>& a, vector<int>& b){
    // check the number of colnum overlap (non-zeros)
    // to improve (temporal) locality of dense input in L1 
    int sim = 0;
    int i = 0;
    int j = 0;
    while ( i<a.size() && j<b.size() ){
        if ( a[i]<b[j] ){
            i++;
        }else if ( a[i]>b[j] ){
            j++;
        }else{
            sim++;i++;j++;
        }
    }
    return sim;
}
void Mat::dfsSegs(){

    unordered_set<int> insular;
    // construct graph
    // { idx, col overlaps } min heap, sort by col overlaps
    // enable the max col overlap be on the top of the stack when DFS
    vector< priority_queue<pair<int,int>, vector<pair<int,int>>, cmp> >
      g(n_segs);

    vector< vector<int> > col_to_seg(n);
    for (int i=0; i<n_segs; i++ )
      for ( auto c: cols_seg[i] ) col_to_seg[c].push_back(i);

    vector<int> mark(n_segs,-1);

    for (int i=0; i<n_segs; ++i)
      {
        for ( auto col_i: cols_seg[i] )
          for ( auto seg_j: col_to_seg[col_i] )
            {
              // 
              if ( seg_j <= i ) continue;
              // pruning, two segs cannot be on the same row panel
              if ( id2r[i] == id2r[seg_j] ) continue; 
              // pruning, check if the seg_j-th seg has been paired
              if ( mark[seg_j] == i ) continue;
              mark[seg_j] = i;
              int sim = checkSim(cols_seg[i],cols_seg[seg_j]);
              if ( sim ){
                g[i].push({seg_j, sim});
                g[seg_j].push({i, sim});
              }
            }
        if ( g[i].empty() ) {
          insular.insert(i);
          g[i].push( {i,0} );
        }
      }

    if ( insular.size()>0 ){
        printf("insular segs = %lu\n",insular.size());
        //assert( insular.size()==0 );
    }
	std::vector<unsigned int> segPtr1(1,0);
	std::vector<unsigned int> segNzRCIdx1;
	std::vector<unsigned int> segNzRowIdx1;
	std::vector<unsigned int> segNzRowIdx_2bit1;
	std::vector<unsigned int> segNzColIdx1;
	std::vector<float> newVals1;
	std::vector<unsigned int> segVoMap1;

	std::vector<float> segNzCV1;
	std::vector<int> seg_rowPtr1;
    // DFS reorder segs
    // explore L1 reuse
    vector<bool> visited(n_segs,false);
    stack<int> st;
    unsigned val_seg = 0;
    unsigned val_w = 0;
    for (int src=0; src<n_segs; ++src){
        if (visited[src] || insular.find(src)!=insular.end())   continue;
        st.push(src);
        while ( !st.empty() ){
            
            int node = st.top();
            st.pop();
            if (visited[node])  continue;
            visited[node] = true;
            val_seg++;

            int seg_nnz = segPtr[node+1] - segPtr[node];
            val_w += seg_nnz;
            for (int i=segPtr[node]; i<segPtr[node+1]; ++i){
                segNzRCIdx1.push_back(segNzRCIdx[2*i]);
                segNzRowIdx1.push_back(segNzRCIdx[2*i]);
                segNzRCIdx1.push_back(segNzRCIdx[2*i+1]);
                segNzColIdx1.push_back(segNzRCIdx[2*i+1]);
                
                newVals1.push_back(newVals[i]);
                
                segNzCV1.push_back(segNzCV[2*i]);
                segNzCV1.push_back(segNzCV[2*i+1]);
        
            }        
            segPtr1.push_back(segPtr1.back()+seg_nnz);
            for (int i=0; i<tm; ++i){
                segVoMap1.push_back(segVoMap[node*tm+i]);
            }
            
            if ( seg_rowPtr1.empty() )  seg_rowPtr1.push_back(0);
            else    seg_rowPtr1.push_back( seg_rowPtr1.back() );
            for (int i=0; i<tm; ++i){
                seg_rowPtr1.push_back( seg_rowPtr1.back() + (seg_rowPtr[node*(tm+1)+i+1] - seg_rowPtr[node*(tm+1)+i]) );
            }
            
            while ( !g[node].empty() ){

                auto nb = g[node].top();
                g[node].pop();
                if ( !visited[nb.first] ){
                    st.push(nb.first);
                }
            }            
        }
    }
    for (int node:insular){
    
        int seg_nnz = segPtr[node+1] - segPtr[node];
        for (int i=segPtr[node]; i<segPtr[node+1]; ++i){
            segNzRCIdx1.push_back(segNzRCIdx[2*i]);
            segNzRowIdx1.push_back(segNzRCIdx[2*i]);
            segNzRCIdx1.push_back(segNzRCIdx[2*i+1]);
            segNzColIdx1.push_back(segNzRCIdx[2*i+1]);
            
            newVals1.push_back(newVals[i]); 
            
            segNzCV1.push_back(segNzCV[2*i]);
            segNzCV1.push_back(segNzCV[2*i+1]);
        } 
        segPtr1.push_back(segPtr1.back()+seg_nnz);
        for (int i=0; i<tm; ++i){
            segVoMap1.push_back(segVoMap[node*tm+i]);
        }
       
        if ( seg_rowPtr1.empty() )  seg_rowPtr1.push_back(0);
        else    seg_rowPtr1.push_back( seg_rowPtr1.back() );
        for (int i=0; i<tm; ++i){
            seg_rowPtr1.push_back( seg_rowPtr1.back() + (seg_rowPtr[node*(tm+1)+i+1] - seg_rowPtr[node*(tm+1)+i]) );
        }
    }

	assert( segPtr.size()==segPtr1.size() );
	assert( segNzRCIdx.size()==segNzRCIdx1.size() );
	assert( segNzRowIdx.size()==segNzRowIdx1.size() );
	assert( segNzColIdx.size()==segNzColIdx1.size() );
	assert( newVals.size()==newVals1.size() );
	assert( segVoMap.size()==segVoMap1.size() );
    assert( segNzCV.size()==segNzCV1.size() );
    assert( seg_rowPtr.size()==seg_rowPtr1.size() );
      
	swap(segPtr, segPtr1);
	swap(segNzRCIdx, segNzRCIdx1);
	swap(segNzRowIdx, segNzRowIdx1);
	swap(segNzColIdx, segNzColIdx1);
	swap(newVals, newVals1);
	swap(segVoMap, segVoMap1);
    swap(segNzCV, segNzCV1);
    swap(seg_rowPtr, seg_rowPtr1);
}
int Mat::checkSim2(map<int,int>& a, vector<int>& b){
    // check the number of colnum overlap (non-zeros)
    // to improve (temporal) locality of dense input in L1 
    int sim = 0;
    int j = 0;
    while ( j<b.size() ){
        if ( a.find(b[j++])!=a.end() ){
            sim++;
        }
    }
    return sim;
}
void Mat::sliWinSegs(){

    vector<int> insular;

    vector< vector<int> > col_to_seg(n);
    for (int i=0; i<n_segs; i++ )
      for ( auto c: cols_seg[i] ) col_to_seg[c].push_back(i);


	std::vector<unsigned int> segPtr1(1,0);
	std::vector<unsigned int> segNzRCIdx1;
	std::vector<unsigned int> segNzRowIdx1;
	std::vector<unsigned int> segNzRowIdx_2bit1;
	std::vector<unsigned int> segNzColIdx1;
	std::vector<float> newVals1;
	std::vector<unsigned int> segVoMap1;

	std::vector<float> segNzCV1;
	std::vector<int> seg_rowPtr1;
    // DFS reorder segs
    // explore L1 reuse
    vector<bool> visited(n_segs,false);
    vector<int> seg_ord;
    int window = 64; // expected active_warps 
    for (int src=0; src<n_segs; ++src){
        if ( visited[src] )   continue;
        visited[src] = true;
       
        // {col_idx, freq}, like a sliding window to kepp track of the latest #window segs 
        map<int,int> col_in_cache;
        std::transform( std::begin(cols_seg[src]),std::end(cols_seg[src]),
                std::inserter(col_in_cache, col_in_cache.end()),
                [](int colID) {return std::make_pair(colID,1);} ); 
        vector<int> tree_seg_ord;
        // while loop for current graph component traversal
        int val_seg = 0;
        while (true)
        { 
            val_seg++;
            assert(val_seg<=n_segs); // just to detect infinite loop

            int candidate_seg = src;

            // at least 1 col overlap between two tiles
            int mx_sim = 1;
            // explore the tile having maximum col overlaps 
            // with previous #window tiles
            for ( auto col_i: col_in_cache )
              for ( auto seg_j: col_to_seg[col_i.first] ) // all segs having col_i.first
                {
                  if ( visited[seg_j] )   continue;
                  int sim = checkSim2(col_in_cache,cols_seg[seg_j]);
                  if ( sim>mx_sim ){
                      mx_sim = sim;
                      candidate_seg = seg_j;
                  }
                }
            // no matter it is insular or new seg
            visited[candidate_seg] = true;
            // check if we find a potential to grow the tree 
            if(candidate_seg!=src){
                // found one potential
                // update col_in_cache
                // works like a sliding window
                if (tree_seg_ord.size()>=window){
                    int to_be_evict = *(tree_seg_ord.rbegin()+window-1);
                    for ( auto col_ii: cols_seg[to_be_evict] ){
                        if (--col_in_cache[col_ii]==0){
                           col_in_cache.erase(col_ii); 
                        }
                    }
                }
                for ( auto col_ii: cols_seg[candidate_seg] ){
                    col_in_cache[col_ii]++; 
                }

                // push the candidate into seg_ord
                if (tree_seg_ord.size()==0) tree_seg_ord.push_back(src);
                tree_seg_ord.push_back(candidate_seg); 
            }else{
                // leaf OR insular
                if (tree_seg_ord.size()==0){
                    // insular
                    insular.push_back(src);
                }
                break;
            }
        }
        // merge the current tree into our forest
        seg_ord.insert(seg_ord.end(),tree_seg_ord.begin(),tree_seg_ord.end()); 

    }
    if (false)
    {
        printf("%d of %s : #segs = %d\n",__LINE__,__FILE__,n_segs);
        printf("%d of %s : #seg_ord = %lu, #insular = %lu\n",__LINE__,__FILE__,
                seg_ord.size(),insular.size());
    }
    // merge the current tree into our forest
    seg_ord.insert(seg_ord.end(),insular.begin(),insular.end()); 
    assert(seg_ord.size()==n_segs);
    for (int node:seg_ord){
    
        int seg_nnz = segPtr[node+1] - segPtr[node];
        for (int i=segPtr[node]; i<segPtr[node+1]; ++i){
            segNzRCIdx1.push_back(segNzRCIdx[2*i]);
            segNzRowIdx1.push_back(segNzRCIdx[2*i]);
            segNzRCIdx1.push_back(segNzRCIdx[2*i+1]);
            segNzColIdx1.push_back(segNzRCIdx[2*i+1]);
            
            newVals1.push_back(newVals[i]); 
            
            segNzCV1.push_back(segNzCV[2*i]);
            segNzCV1.push_back(segNzCV[2*i+1]);
        } 
        segPtr1.push_back(segPtr1.back()+seg_nnz);
        for (int i=0; i<tm; ++i){
            segVoMap1.push_back(segVoMap[node*tm+i]);
        }
       
        if ( seg_rowPtr1.empty() )  seg_rowPtr1.push_back(0);
        else    seg_rowPtr1.push_back( seg_rowPtr1.back() );
        for (int i=0; i<tm; ++i){
            seg_rowPtr1.push_back( seg_rowPtr1.back() + (seg_rowPtr[node*(tm+1)+i+1] - seg_rowPtr[node*(tm+1)+i]) );
        }
    }

	assert( segPtr.size()==segPtr1.size() );
	assert( segNzRCIdx.size()==segNzRCIdx1.size() );
	assert( segNzRowIdx.size()==segNzRowIdx1.size() );
	assert( segNzColIdx.size()==segNzColIdx1.size() );
	assert( newVals.size()==newVals1.size() );
	assert( segVoMap.size()==segVoMap1.size() );
    assert( segNzCV.size()==segNzCV1.size() );
    assert( seg_rowPtr.size()==seg_rowPtr1.size() );
      
	swap(segPtr, segPtr1);
	swap(segNzRCIdx, segNzRCIdx1);
	swap(segNzRowIdx, segNzRowIdx1);
	swap(segNzColIdx, segNzColIdx1);
	swap(newVals, newVals1);
	swap(segVoMap, segVoMap1);
    swap(segNzCV, segNzCV1);
    swap(seg_rowPtr, seg_rowPtr1);
}



void Mat::csr2tile(){


  const int nnz_csr = rowPtr[m];

    bool print_bucket = false;
	int tileRows = (m+tm-1)/tm;
	for (int i=0; i<tileRows; ++i){
		//csr2flex_Rmajor(i);
		//csr2flex_Cmajor(i);
		//csr2regular(i);
        csr2seg_Cmajor(i);
	} 

    assert( nnz_csr == seg_rowPtr.back() );

    n_segs = segPtr.size()-1;
    if (print_bucket) printf("%d of %s, n_segs = %d\n",__LINE__, __FILE__, n_segs); 
    bool seg_sort = true;
    if (seg_sort) {
        //permute_segs();
        dfsSegs();
        //sliWinSegs();
    }
    
        int device_id;
        cudaDeviceProp prop;
        cudaGetDevice( &device_id );
        cudaGetDeviceProperties( &prop, device_id );
        int n_sm = prop.multiProcessorCount;
        sms = n_sm; 
        
        // distribute segs into n_sm+1 buckets, contiguous segs are in a bucket
        // according to #non zeros ( wkload per sm )
        // to balance workload, the last bucket is to offer segs when faster SMs are free   
        int nnz = newVals.size(); 

        assert( nnz == nnz_csr );

        int wkload = nnz / n_sm; 
        int seg_head_sm = 0;
        int seg_tail_sm;
        int validate_nnz = 0;
        
        // assign segs to each sm bucket
        for (int i=0; i<n_sm; ++i){
            next_seg.push_back( seg_head_sm );
            int nz = segPtr[seg_head_sm+1] - segPtr[seg_head_sm];
            
            seg_tail_sm = seg_head_sm + 1;
            while ( seg_tail_sm < n_segs && nz<(int)(0.98*wkload) ){
                nz += (segPtr[seg_tail_sm+1] - segPtr[seg_tail_sm]);
                seg_tail_sm++;
            }
            validate_nnz += nz;
            grouped_tailSeg.push_back( min(n_segs,seg_tail_sm) );
            if ( seg_head_sm==min(n_segs,seg_tail_sm) ){
                empty_bucket++;
            }
            seg_head_sm = seg_tail_sm;
        }
        
        // the last bucket is used for workload balance among SMs 
        // if seg_head_sm==n_segs, then n_segs==seg_head_sm
        next_seg.push_back( seg_head_sm );
        grouped_tailSeg.push_back( n_segs );
        validate_nnz += segPtr[n_segs]-segPtr[seg_head_sm];
        assert( validate_nnz==segPtr.back() );
        assert( grouped_tailSeg.size()==n_sm+1 );
        assert( next_seg.size()==n_sm+1 );
    
}
void Mat::print3(int l){
    if ( true ){
        printf("\nSegPtr: \n");
        for (int i=0; i<(l?l:segPtr.size()); ++i){
            printf("(%d:%d)  ",i,segPtr[i]);
        }
        printf("\nSegNzRC: \n");
        for (int i=0; i<(l?l:segNzRCIdx.size()/2); ++i){
            printf("(%d:%d)  ",segNzRCIdx[2*i],segNzRCIdx[2*i+1]);
        }
        if (false){
            printf("\nSegRowNzIdx: %d\n",(int)segNzRowIdx.size());
            for (int i=0; i<(l?l:segNzRowIdx.size()); ++i){
                printf("%d  ",segNzRowIdx[i]);
            }
            printf("\nSegColNzIdx: %d\n",(int)segNzColIdx.size());
            for (int i=0; i<(l?l:segNzColIdx.size()); ++i){
                printf("%d  ",segNzColIdx[i]);
            }
        }
    }
    printf("\nSegVoMap: %d\n",(int)segVoMap.size());
    for (int i=0; i<(l?l:segVoMap.size()); ++i){
        printf("%d->%d  ",i,segVoMap[i]&0x7fffffff);
    }
    printf("\n");
}

void Mat::csr2seg_Cmajor(int ridx){
	// row tile upper bound and lower bound
	int rowStart = ridx * tm;
	int rowEnd = min(m, (ridx+1)*tm); // exclusive

	// keep track of the cols in each row
	std::vector<int> cOffset(tm, 0);

    // {col, val}, for kernel v31 
    std::vector<std::vector<std::pair<int,float>>> segcv(tm, std::vector<std::pair<int,float>>()); 


    int dif = 0.1*nnz_limit; 
    int nnzInSeg = 0;
    int nnz_cur_panel = rowPtr[rowEnd] - rowPtr[rowStart];    

    // If n_nodes_z_out>0 some panels can be empty, which tiling can't handle.
    assert( !dl.dl_original->n_nodes_z_out );
    vector<int> atom(tm, 0);

    map<int,int> occ_cols;
    for ( auto c: views::iota(rowPtr[rowStart],rowPtr[rowEnd]) )
      occ_cols[colIdx[c]]++;
    const auto last_col = occ_cols.rbegin()->first;
    // collect segs in the panel
    for ( auto [j,ncol]: occ_cols ) {
        
        int segId = segPtr.size()-1;
        for ( int i=rowStart; i<rowEnd; ++i ){
            // absolute position of the nze in csr, idx = base + offset
            int c = rowPtr[i] + cOffset[i-rowStart];
            if ( colIdx[c]==j && c<rowPtr[i+1] ){
                // nze values
                segNzRowIdx.push_back(i-rowStart);
                segNzColIdx.push_back(j);
                
                segcv[i-rowStart].push_back({j,vals[c]}); // for v31 kernel

                segNzRCIdx.push_back(i-rowStart); 
                segNzRCIdx.push_back(j);
                newVals[pos++] = vals[c];
                cOffset[i-rowStart]++;
                atom[i-rowStart]++;
                nnzInSeg++;

                if ( !cols_seg.count(segId) || cols_seg[segId].back()!=j ){
                    cols_seg[segId].push_back(j);
                }
            }
        }
        if ( (j==last_col && nnzInSeg) || (nnz_limit - nnzInSeg)<=dif || nnzInSeg>nnz_limit ){
        
            // for kernel v31
            if ( !seg_rowPtr.empty() ) seg_rowPtr.push_back( seg_rowPtr.back() + 0 );
            else seg_rowPtr.push_back( 0 );
            
            for ( int i=0; i<tm; ++i ){
                seg_rowPtr.push_back( seg_rowPtr.back() + segcv[i].size() );
                for ( auto &p:segcv[i] ){
                    segNzCV.push_back((float)p.first); // col of the nz is stored in float
                    segNzCV.push_back(p.second);
                }
                segcv[i].clear();
            }


            aux_seg.push({ ridx, segPtr.size()-1 }); // {seg_row, seg_idx}
            id2r[segPtr.size()-1] = ridx;
            count_segs[ridx]++;
            segPtr.push_back(segPtr.back()+nnzInSeg);
            nnzInSeg = 0;
           
            for (int i=rowStart; i<rowStart+tm; ++i){
                if ( i<rowEnd ){
                    if ( atom[i-rowStart]>=0 && atom[i-rowStart]<(rowPtr[i+1]-rowPtr[i]) ){
                        // if the #nz in a specific row of a seg 
                        // is less than that of the whole row,
                        // the row requires "atomic add".
                        // use MSB to mark it.
                        segVoMap.push_back( voMp[i] | (1<<31) );
                    }else{ 
                        segVoMap.push_back( voMp[i] );
                    }
                }else{
                    // for the last panel, the rows may be less than tm 
                    segVoMap.push_back(1<<(bit_width((uint)m)+1));
                }
                
                atom[ i-rowStart ] = 0;
            }
        }
    }
}

void
Mat::stats_collect2(FILE *stream)
{
  //const uint seg_m = ( m + tm - 1 ) / tm;
  //const uint seg_m_floor = m / tm;
  
  const uint seg_nnz_lim = tm * n;
  assert( seg_nnz_lim == tm * uint64_t(n) ); // Overflow check.
  const uint seg_lg_nnz_lim = bit_width(seg_nnz_lim);
  uint seg_lg_nnz_max = 0, seg_lg_nnz_min = seg_lg_nnz_lim;
  seg_lg_nnz_histo.resize(seg_lg_nnz_lim+1);

  const uint n_segs = segPtr.size()-1;

  n_col_sum = 0;
  int sp_seg1 = 0;
  int sp_seg2 = 0;
  int sp_seg3 = 0;
  int sp_seg4 = 0;
  for ( uint seg_idx = 0; seg_idx < n_segs; seg_idx++ )
    {
      const uint nnz_seg = segPtr[seg_idx+1] - segPtr[seg_idx];
      if ( nnz_seg<=NNZ_LIMIT/4 ) sp_seg1++;
      else if ( nnz_seg<=NNZ_LIMIT/2 ) sp_seg2++;
      else if ( nnz_seg<=NNZ_LIMIT ) sp_seg3++;
      else if ( nnz_seg>NNZ_LIMIT ) sp_seg4++;
      
      const uint lg_nnz = bit_width(nnz_seg);
      set_max( seg_lg_nnz_max, lg_nnz );
      set_min( seg_lg_nnz_min, lg_nnz );
      seg_lg_nnz_histo[lg_nnz]++;
      
      unordered_set<int> colset; 
      for (int i=segPtr[seg_idx]; i<segPtr[seg_idx+1]; ++i){
        colset.insert(segNzColIdx[i]);
      }
      n_col_sum += colset.size();
    }

  for ( uint32_t v: segVoMap ) if ( v & (1u<<31) ) atomic_op++;

  if ( !stream ) return;

  fprintf(stream, "Ordering %s.  Segments %d × *\n",
          dl.vertex_order_abbr.c_str(), tm);

  fprintf(stream, "Histogram of lg non-zeros per tile-segment.\n");

  pTable lg_nnz_tab(stream);
  for ( auto i: views::iota(seg_lg_nnz_min,seg_lg_nnz_max+1) )
    {
      pTable_Row _(lg_nnz_tab);
      lg_nnz_tab.entry("Lg", "%2d", i);
      lg_nnz_tab.entry("Seg", "%7d", seg_lg_nnz_histo[i] );
      lg_nnz_tab.entry
        ("Pct", "%6.2f", seg_lg_nnz_histo[i] * 100.0 / n_segs );
    }

  fprintf(stream,"Arrays m=%d, n=%d, k=%d. tile-seg %d × *.\n"
          "              n_segs=%u, nnz=%d  Avg deg=%.1f\n",
         m, n, k, tm, n_segs, nnz, double(nnz)/m);

  fprintf(stream,"nnz / seg: %.3f     Load / B elt  %.3f\n",
         double(nnz) / n_segs,
         n_col_sum / double(nnz) );
  fprintf(stream,"nnz in segs:  (0,%d]: %d  %.3f     (%d,%d]: %d  %.3f     (%d,%d]: %d  %.3f      (%d,%d]: %d  %.3f\n",
         NNZ_LIMIT/4, sp_seg1, double(sp_seg1) / n_segs,
         NNZ_LIMIT/4, NNZ_LIMIT/2, sp_seg2, double(sp_seg2) / n_segs,
         NNZ_LIMIT/2, NNZ_LIMIT, sp_seg3, double(sp_seg3) / n_segs,
         NNZ_LIMIT, NNZ_LIMIT+tm, sp_seg4, double(sp_seg4) / n_segs );
  fprintf(stream,"\n");
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
Mat::stats_collect(FILE *stream)
{
  const uint tmn = tm * tn;

  const uint tile_m = ( m + tm - 1 ) / tm;
  const uint tile_m_floor = m / tm;
  const uint tile_n = ( n + tn - 1 ) / tn;
  tile_p_row_histo.resize(tile_n+1);
  uint max_n_tiles = 0;

  const uint panel_nnz_lim = tm * n;
  assert( panel_nnz_lim == tm * uint64_t(n) ); // Overflow check.
  const uint panel_lg_nnz_lim = bit_width(panel_nnz_lim);
  uint panel_lg_nnz_max = 0, panel_lg_nnz_min = panel_lg_nnz_lim;
  panel_lg_nnz_histo.resize(panel_lg_nnz_lim+1);

  for ( uint tile_r = 0;  tile_r < tile_m;  tile_r++ )
    {
      const uint n_tiles = tileRowPtr[tile_r+1] - tileRowPtr[tile_r];
      assert( n_tiles <= tile_n );
      set_max( max_n_tiles, n_tiles );
      tile_p_row_histo[n_tiles]++;
      if ( tile_r >= tile_m_floor ) continue;
      const auto tile_start = tileRowPtr[tile_r];
      const auto tile_stop = tileRowPtr[tile_r+1];
      const auto tidx_start = tileNnz[tile_start];
      const auto tidx_stop = tileNnz[tile_stop];
      if (tidx_stop <= tidx_start){
        printf("tm = %d, tn = %d, tidx_stop = %d, tidx_satrt = %d\n",tm, tn, tidx_stop, tidx_start);
      }
      assert( tidx_stop > tidx_start );
      const uint nnz_panel = tidx_stop - tidx_start;
      const uint lg_nnz = bit_width(nnz_panel);
      assert( lg_nnz < panel_lg_nnz_histo.size() );
      assert( lg_nnz );
      set_max( panel_lg_nnz_max, lg_nnz );
      set_min( panel_lg_nnz_min, lg_nnz );
      panel_lg_nnz_histo[lg_nnz]++;
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
  if ( !stream ) return;

  fprintf(stream, "Ordering %s.  Tile %d × %d\n",
          dl.vertex_order_abbr.c_str(), tm, tn);

  fprintf(stream, "Histogram of lg non-zeros per panel (tile row).\n");

  pTable lg_nnz_tab(stream);
  for ( auto i: views::iota(panel_lg_nnz_min,panel_lg_nnz_max+1) )
    {
      pTable_Row _(lg_nnz_tab);
      lg_nnz_tab.entry("Lg", "%2d", i);
      lg_nnz_tab.entry("Panels", "%7d", panel_lg_nnz_histo[i] );
      lg_nnz_tab.entry
        ("Pct", "%6.2f", panel_lg_nnz_histo[i] * 100.0 / tile_m_floor );
    }

  fprintf(stream,"[1,8): %f%%   ", tiles_bucket[0]*100.0/tile_m); 
  fprintf(stream,"[8,16): %f%%    ", tiles_bucket[1]*100.0/tile_m); 
  fprintf(stream,"[16,32): %f%%   ", tiles_bucket[2]*100.0/tile_m); 
  fprintf(stream,"[32,64): %f%%   ", tiles_bucket[3]*100.0/tile_m); 
  fprintf(stream,"[64,128): %f%%  ", tiles_bucket[4]*100.0/tile_m); 
  fprintf(stream,"[128, +OO): %f%%\n", tiles_bucket[5]*100.0/tile_m); 

  fprintf(stream,"Arrays m=%d, n=%d, k=%d. Tile %d × %d.   nnz=%d  Avg deg=%.1f\n",
         m, n, k, tm, tn, nnz, double(nnz)/m);

  fprintf(stream,"nnz / tile: %.3f  Load / B elt  %.3f\n",
         double(nnz) / n_tiles,
         n_col_sum / double(nnz) );
  int n_t_hist_pr = 0;
  fprintf(stream,"Tile nnz histogram: (n_tiles %d)\n",n_tiles);
  for ( uint i=0; i<tile_nnz_histo.size(); i++ )
    if ( auto tnnz = tile_nnz_histo[i]; tnnz )
      {
        if ( n_t_hist_pr++ > 6 ) break;
        fprintf(stream,"%3d %5.2f%%, ", i, tnnz * 100.0 / n_tiles);
      }
  fprintf(stream,"\n");
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
