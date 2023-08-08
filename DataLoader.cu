#include "DataLoader.cuh"
#include "edgelist.cuh"
#include "flex.cuh"

#include <ranges>

DataLoader::DataLoader(const std::string& data_path, const int di):dim(di){
    std::string data_name = data_path.substr(data_path.find_last_of("/")+1);
    graph_name = data_name.substr(0, data_name.find(".")); 

    vertex_order_abbr = "OVO"; // Original Vertex Order.
    std::fstream fin;
    fin.open(data_path,std::ios::in);
    std::string line, word;
    
    std::getline(fin,line);
    std::stringstream ss1(line);
    while(std::getline(ss1,word,',')){
        rowPtr.push_back(std::stoi(word));        
    }
    
    std::getline(fin,line);
    std::stringstream ss2(line);
    while(std::getline(ss2,word,',')){
        col.push_back(std::stoi(word));        
    }
    
    if (data_name == "amazon.csv"){
        // amazon.csv only contains row offset and col indice
        fin.close(); 
        std::cout<<"Amazon n = "<<rowPtr.size()-1<<std::endl;
        std::cout<<"Amazon nnz = "<<col.size()<<std::endl;
        for (size_t i=0; i<col.size(); ++i){
            unsigned int temp_v = (rand()<<16)|rand();
            temp_v = (temp_v&0x7fffff) | 0x40000000; 
            vals.push_back( *((float*)&temp_v) - 3.0f );
        }
    }else{
        std::getline(fin,line);
        std::stringstream ss3(line);
        while(std::getline(ss3,word,',')){
            vals.push_back(std::stof(word));        
        }
        fin.close(); 
    }
    //int debug_check = 16652;
    //print4(debug_check, false);
    assert(col.size()==vals.size());
    m = rowPtr.size()-1; 
    n = m;
    nnz = col.size();

    if (data_name == "polblogs.csv"){
        c = 2; 
    }else if(data_name == "cora.csv"){
        c = 7; 
    }else if (data_name == "citeseer.csv"){
        c = 6; 
    }else if (data_name == "pubmed.csv"){
        c = 3; 
    }else if (data_name == "ppi.csv"){
        c = 121; 
    }else if (data_name == "reddit.csv"){
        c = 41; 
    }else if (data_name == "flickr.csv"){
        c = 7; 
    }else if (data_name == "yelp.csv"){
        c = 100; 
    }else if (data_name == "amazon.csv"){
        c = 107; 
    }else{
        //std::cout<<"not supported data"<<std::endl;
        //exit(0);
        c = 100;
    }
    vo_mp.resize(m);
    std::iota(vo_mp.begin(), vo_mp.end(), 0);
    cuda_alloc_cpy();
}

void 
DataLoader::print4(int l, bool s){
    if (s){
        for (int i=0; i<rowPtr.size()-1; ++i){
            for (int j=rowPtr[i]; j<rowPtr[i+1]; ++j){
                if (i==l){
                    printf("r = %d, c = %d, v = %f\n", i, col[j], vals[j]);
                }
            }
        }
    }else{
        for (int i=0; i<rowPtr.size()-1; ++i){
            for (int j=rowPtr[i]; j<rowPtr[i+1]; ++j){
                if (col[j]==l){
                    printf("r = %d, c = %d, v = %f\n", i, col[j], vals[j]);
                }
            }
        }
    }
}
void
DataLoader::cuda_alloc_cpy()
{
#ifdef AXW
    //LOG(INFO) << "Initialize X & W ...";
    for (int i=0; i<c*dim; ++i){
        cpuW.push_back((float)rand()/RAND_MAX);
    }
    CUDA_CHECK(cudaMalloc(&gpuW, sizeof(float) * c * dim));
    CUDA_CHECK(cudaMemcpy(gpuW, cpuW, sizeof(float)*dim*c, cudaMemcpyHostToDevice));
      
    CUDA_CHECK(cudaMalloc(&gpuRef1, sizeof(float) * c * n));
    CUDA_CHECK(cudaMemset(gpuRef1, 0, sizeof(float)*n*c));
    
    CUDA_CHECK(cudaMalloc(&gpuRef2, sizeof(float) * c * n));
    CUDA_CHECK(cudaMemset(gpuRef2, 0, sizeof(float)*n*c));
#endif

    CUDA_CHECK(cudaMalloc(&rowPtr_dev, sizeof(unsigned int) * (m+1)));
    CUDA_CHECK(cudaMemcpy(rowPtr_dev, rowPtr.data(), sizeof(unsigned int)*(n+1), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&col_dev, sizeof(unsigned int) * nnz));
    CUDA_CHECK(cudaMemcpy(col_dev, col.data(), sizeof(unsigned int)*nnz, cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&vals_dev, sizeof(float) * nnz)); 
    CUDA_CHECK(cudaMemcpy(vals_dev, vals.data(), sizeof(float)*nnz, cudaMemcpyHostToDevice));
    

    C_elts = m * dim;
    gpuC_bytes = C_elts * sizeof( cpuC[0] );
    CUDA_CHECK(cudaMalloc(&gpuC, gpuC_bytes));
    CUDA_CHECK(cudaMemset(gpuC, 0, gpuC_bytes));
    if (vertex_order_abbr == "OVO"){
        for (int i=0; i<n; ++i){
            for (int j=0; j<dim; ++j){
                unsigned int temp_v = (rand()<<16)|rand();
                temp_v = (temp_v&0x7fffff) | 0x40000000; 
                cpuX.push_back( *((float*)&temp_v) - 3.0f );
                //cpuX.push_back(i);
                //cpuX.push_back(i*dim+j);
            }
        }
    }

    //gpuX_bytes = cpuX.size() * sizeof( cpuX[0] );
    gpuX_bytes = n * dim * sizeof( float );

    if (vertex_order_abbr == "OVO"){
        CUDA_CHECK(cudaMalloc(&gpuX, gpuX_bytes ) );
        CUDA_CHECK(cudaMemcpy(gpuX, cpuX.data(), gpuX_bytes, cudaMemcpyHostToDevice));
    }
}

void
DataLoader::c_cuSpmm_run(Perfs& perfRes)
{
  cuSpmm(*this, perfRes);
  h_ref_c.resize( C_elts );
  CUDA_CHECK
    ( cudaMemcpy( h_ref_c.data(), gpuC, gpuC_bytes, cudaMemcpyDeviceToHost ));
}

void
DataLoader::gpuC_zero()
{
  CUDA_CHECK( cudaMemset( gpuC, 0, gpuC_bytes ) );
}


DataLoader::DataLoader(const DataLoader& dl)
{
  #define CPY(m) m = dl.m
  CPY(m); CPY(n); CPY(dim); CPY(c); CPY(nnz); CPY(graph_name);
  #undef CPY
}

DataLoaderDFS::DataLoaderDFS(const DataLoader& dl):DataLoader(dl)
{
    //for ( auto bb:dl.cpuX ){
    //    cpuX.push_back(bb);
    //}
  gpuX = dl.gpuX;
  // Renumber the vertices based on a depth-first search of the graph in
  // dl, starting at vertex 0.

  vertex_order_abbr = "DFS";

  assert( dl.rowPtr.size() == n + 1 );

  auto dst_iter_make = [&](uint s)
  { return ranges::subrange(&dl.col[dl.rowPtr[s]],&dl.col[dl.rowPtr[s+1]]); };

  vector<uint> vo_to_dfs(n);  // original Vertex Order to DFS order
  vo_to_dfs[0] = n; // Will be changed back to zero.

  col.resize( dl.col.size() );
  vals.resize( dl.col.size() );

  //
  // Perform Depth-First Search (DFS)
  //
  auto root = dst_iter_make(0);
  vector< decltype(root) > stack { root };

  rowPtr.reserve( n+1 );
  rowPtr.push_back( 0 );
  rowPtr.push_back( root.size() );

  while ( !stack.empty() )
    {
      auto& dst_iter = stack.back();

      while ( dst_iter && vo_to_dfs[ dst_iter.front() ] ) dst_iter.advance(1);
      if ( !dst_iter ) { stack.pop_back();  continue; }

      const uint dst_vo  = dst_iter.front();  dst_iter.advance(1);
      const uint dst_dfs = rowPtr.size() - 1;
      vo_to_dfs[ dst_vo ] = dst_dfs;
      auto dst_node_iterator = dst_iter_make( dst_vo );
      stack.push_back( dst_node_iterator );
      // Update edge list pointer. (Row Number to vals/col array index.)
      rowPtr.push_back( rowPtr.back() + dst_node_iterator.size() );
    }
  vo_to_dfs[0] = 0;

  vo_mp.resize(m);
  for (ul i=0; i<vo_to_dfs.size(); ++i){
      ul v = vo_to_dfs[i];
      vo_mp[v] = i;
  }
  //
  // Copy destinations (col) and edge weights (vals) from dl to this object.
  //
  for ( auto src_vo: views::iota(size_t(0),n) )
    {
      const auto src_dfs = vo_to_dfs[src_vo];
      const int d = dl.rowPtr[src_vo+1] - dl.rowPtr[src_vo];
      if( rowPtr[src_dfs] + d != rowPtr[src_dfs+1] ){
          printf("rowPtr_len = %zd, rowPtr[%d] + %d = %d,  rowPtr[%d+1] = %d\n", 
                  rowPtr.size(),   src_dfs, d, rowPtr[src_dfs] + d, src_dfs,rowPtr[src_dfs+1]);
          assert( rowPtr[src_dfs] + d == rowPtr[src_dfs+1] );
      }

      // Sort destinations.  Tiling algorithm needs dests sorted.
      vector< pair<float,uint> > perm;  perm.reserve(d);
      const auto e_idx_vo = dl.rowPtr[ src_vo ];
      for ( auto e: views::iota( e_idx_vo, e_idx_vo + d ) )
        perm.emplace_back( dl.vals[ e ], vo_to_dfs[ dl.col[ e ] ] );
      ranges::sort(perm, ranges::less(), [](auto& v) { return v.second; } );

      uint e_idx_dfs_i = rowPtr[src_dfs];
      for ( auto& [val, dst_new]: perm )
        {
          col[ e_idx_dfs_i ] = dst_new;
          vals[ e_idx_dfs_i++ ] = val;
        }
    }

  //
  // Perform a rough test of whether the two graphs match.
  //

  vector<int64_t> check_vo(n);
  vector<int64_t> check_dfs(n);
  vector<double> checkw_vo(n);
  vector<double> checkw_dfs(n);

  for ( uint src_vo: views::iota(size_t(0),n) )
    {
      const auto src_dfs = vo_to_dfs[src_vo];
      const int d = dl.rowPtr[src_vo+1] - dl.rowPtr[src_vo];
      assert( rowPtr[src_dfs] + d == rowPtr[src_dfs+1] );
      const int inc = src_vo & 0xf;
      for ( auto n_idx: views::iota(0,d) )
        {
          const uint e_idx_vo = dl.rowPtr[src_vo] + n_idx;
          const uint e_idx_dfs = rowPtr[src_dfs] + n_idx;
          check_vo[ dl.col[ e_idx_vo ] ] += inc;
          check_dfs[ col[ e_idx_dfs ] ] += inc;
          checkw_vo[ dl.col[ e_idx_vo ] ] += dl.vals[ e_idx_vo ];
          checkw_dfs[ col[ e_idx_dfs ] ] += vals[ e_idx_dfs ];
        }
    }

  for ( uint src_vo: views::iota(size_t(0),n) )
    {
      assert( check_vo[src_vo] == check_dfs[ vo_to_dfs[src_vo] ] );
      assert( checkw_vo[src_vo] == checkw_dfs[ vo_to_dfs[src_vo] ] );
    }

  cuda_alloc_cpy();
}

DataLoaderDeg::DataLoaderDeg(const DataLoader& dl):DataLoader(dl)
{
  //  for ( auto bb:dl.cpuX ){
  //      cpuX.push_back(bb);
  //  }
  gpuX = dl.gpuX;
  vertex_order_abbr = "DEG";

  assert( dl.rowPtr.size() == n + 1 );

  vector<ul> vo_to_deg;  // original Vertex Order to DEG order
  
  col.resize( dl.col.size() );
  vals.resize( dl.col.size() );

  //
  // Convert CSR to edge lists
  //
  Edgelist h(dl);
  vo_to_deg.reserve(m);
  vo_to_deg = order_deg(h,true);
  //
  // According to renumbered vertex order, generate rowPtr 
  //  
  vo_mp.resize(m);
  for (ul i=0; i<vo_to_deg.size(); ++i){
      ul v = vo_to_deg[i];
      vo_mp[v] = i;
  }
  rowPtr.push_back( 0 );
  for (auto v:vo_mp){
    rowPtr.push_back(rowPtr.back()+dl.rowPtr[v+1]-dl.rowPtr[v]);
  }
  //
  // Copy destinations (col) and edge weights (vals) from dl to this object.
  //
  for ( auto src_vo: views::iota(size_t(0),n) )
    {
      const auto src_deg = vo_to_deg[src_vo];
      const int d = dl.rowPtr[src_vo+1] - dl.rowPtr[src_vo];
      //printf("src_vo = %d, src_deg = %d, d = %d\n",src_vo, src_deg, d);
      assert( rowPtr[src_deg] + d == rowPtr[src_deg+1] );

      // Sort destinations.  Tiling algorithm needs dests sorted.
      vector< pair<float,uint> > perm;  perm.reserve(d);
      const auto e_idx_vo = dl.rowPtr[ src_vo ];
      for ( auto e: views::iota( e_idx_vo, e_idx_vo + d ) )
        perm.emplace_back( dl.vals[ e ], vo_to_deg[ dl.col[ e ] ] );
      ranges::sort(perm, ranges::less(), [](auto& v) { return v.second; } );

      uint e_idx_deg_i = rowPtr[src_deg];
      for ( auto& [val, dst_new]: perm )
        {
          col[ e_idx_deg_i ] = dst_new;
          vals[ e_idx_deg_i++ ] = val;
        }
    }

  //
  // Perform a rough test of whether the two graphs match.
  //

  cuda_alloc_cpy();
}

DataLoaderRcm::DataLoaderRcm(const DataLoader& dl):DataLoader(dl)
{
  //  for ( auto bb:dl.cpuX ){
  //      cpuX.push_back(bb);
  //  }
  gpuX = dl.gpuX;
  vertex_order_abbr = "RCM";

  assert( dl.rowPtr.size() == n + 1 );

  vector<ul> vo_to_rcm;  // original Vertex Order to RCM order

  col.resize( dl.col.size() );
  vals.resize( dl.col.size() );

  //
  // Convert CSR to edge lists
  //

  Edgelist h(dl);
  vo_to_rcm.reserve(m);
  vo_to_rcm = order_rcm(h);

  //
  // According to renumbered vertex order, generate rowPtr 
  //  
  vo_mp.resize(m);
  for (ul i=0; i<vo_to_rcm.size(); ++i){
      ul v = vo_to_rcm[i];
      vo_mp[v] = i;
  }
  rowPtr.push_back( 0 );
  for (auto v:vo_mp){
    rowPtr.push_back(rowPtr.back()+dl.rowPtr[v+1]-dl.rowPtr[v]);
  }
  //
  // Copy destinations (col) and edge weights (vals) from dl to this object.
  //
  for ( auto src_vo: views::iota(size_t(0),n) )
    {
      const auto src_rcm = vo_to_rcm[src_vo];
      const int d = dl.rowPtr[src_vo+1] - dl.rowPtr[src_vo];
      assert( rowPtr[src_rcm] + d == rowPtr[src_rcm+1] );

      // Sort destinations.  Tiling algorithm needs dests sorted.
      vector< pair<float,uint> > perm;  perm.reserve(d);
      const auto e_idx_vo = dl.rowPtr[ src_vo ];
      for ( auto e: views::iota( e_idx_vo, e_idx_vo + d ) )
        perm.emplace_back( dl.vals[ e ], vo_to_rcm[ dl.col[ e ] ] );
      ranges::sort(perm, ranges::less(), [](auto& v) { return v.second; } );

      uint e_idx_rcm_i = rowPtr[src_rcm];
      for ( auto& [val, dst_new]: perm )
        {
          col[ e_idx_rcm_i ] = dst_new;
          vals[ e_idx_rcm_i++ ] = val;
        }
    }

  //
  // Perform a rough test of whether the two graphs match.
  //

  cuda_alloc_cpy();
}

DataLoaderGorder::DataLoaderGorder(const DataLoader& dl):DataLoader(dl)
{
  //  for ( auto bb:dl.cpuX ){
  //      cpuX.push_back(bb);
  //  }
  gpuX = dl.gpuX;
  vertex_order_abbr = "GOR";

  assert( dl.rowPtr.size() == n + 1 );

  vector<ul> vo_to_gorder;  // original Vertex Order to Gorder order

  col.resize( dl.col.size() );
  vals.resize( dl.col.size() );

  //
  // Convert CSR to edge lists
  //
  ul window_sz= 3;
  Edgelist h(dl);
  vo_to_gorder.reserve(m);
  vo_to_gorder = complete_gorder(h, window_sz);

  //
  // According to renumbered vertex order, generate rowPtr 
  //  
  vo_mp.resize(m);
  for (ul i=0; i<vo_to_gorder.size(); ++i){
      ul v = vo_to_gorder[i];
      vo_mp[v] = i;
  }
  rowPtr.push_back( 0 );
  for (auto v:vo_mp){
    rowPtr.push_back(rowPtr.back()+dl.rowPtr[v+1]-dl.rowPtr[v]);
  }
  //
  // Copy destinations (col) and edge weights (vals) from dl to this object.
  //
  for ( auto src_vo: views::iota(size_t(0),n) )
    {
      const auto src_gorder = vo_to_gorder[src_vo];
      const int d = dl.rowPtr[src_vo+1] - dl.rowPtr[src_vo];
      assert( rowPtr[src_gorder] + d == rowPtr[src_gorder+1] );

      // Sort destinations.  Tiling algorithm needs dests sorted.
      vector< pair<float,uint> > perm;  perm.reserve(d);
      const auto e_idx_vo = dl.rowPtr[ src_vo ];
      for ( auto e: views::iota( e_idx_vo, e_idx_vo + d ) )
        perm.emplace_back( dl.vals[ e ], vo_to_gorder[ dl.col[ e ] ] );
      ranges::sort(perm, ranges::less(), [](auto& v) { return v.second; } );

      uint e_idx_gorder_i = rowPtr[src_gorder];
      for ( auto& [val, dst_new]: perm )
        {
          col[ e_idx_gorder_i ] = dst_new;
          vals[ e_idx_gorder_i++ ] = val;
        }
    }

  //
  // Perform a rough test of whether the two graphs match.
  //

  cuda_alloc_cpy();
}

bool DataLoader::compare(){
    for (size_t i=0; i<m*c; ++i){
        if (abs(cpuRef1[i]-cpuRef2[i])>=0.1){
            std::cout<<"Ref1["<<i<<"]="<<std::setprecision(12)<<cpuRef1[i]<<" / "<<"Ref2["<<i<<"]="<<std::setprecision(12)<<cpuRef2[i]<<std::endl;
            return false;
        }
        //if (cpuC[i] != cpuRef2[i])  return false;
    }
    std::cout<<"The results are correct.. "<<std::endl;
    return true;
}
void DataLoader::print_data(){
    //LOG(INFO) << "print start.";
    std::cout<<"The first 5 elements of rowptr: ";
    for(auto it=rowPtr.begin(); it<rowPtr.begin()+5; it++)
        std::cout<<(*it)<<" ";
    std::cout<<std::endl;

    std::cout<<"The last 5 elements of rowptr: ";
    for(auto it=rowPtr.end()-5; it!=rowPtr.end() ; it++)
        std::cout<<(*it)<<" ";
    std::cout<<std::endl;

    std::cout<<"The first 5 elements of indies: ";
    for(auto it=col.begin(); it<col.begin()+5 ; it++)
        std::cout<<(*it)<<" ";
    std::cout<<std::endl;

    std::cout<<"The last 5 elements of indies: ";
    for(auto it=col.end()-5; it!=col.end() ; it++)
        std::cout<<(*it)<<" ";
    std::cout<<std::endl;

    std::cout<<"The first 5 elements of vals: ";
    for(auto it=vals.begin(); it<vals.begin()+5 ; it++)
        std::cout<<(*it)<<" ";
    std::cout<<std::endl;

    std::cout<<"The last 5 elements of vals: ";
    for(auto it=vals.end()-5; it!=vals.end() ; it++)
        std::cout<<(*it)<<" ";
    std::cout<<std::endl;
    
    std::cout<<"The first 5 elements of X: ";
    for(auto it=0; it<5 ; it++)
        std::cout<<cpuX[it]<<" ";
    std::cout<<std::endl;
    
    //std::cout<<"The first 5 elements of W: ";
    //for(auto it=0; it<5 ; it++)
    //    std::cout<<cpuW[it]<<" ";
    //std::cout<<std::endl;
    
    std::cout<<std::endl;
    //std::cout<<"The number of nodes: "<< get_nodes()<<"   Rowptr: "<<data.at(0).size()<<"   Pointer: "<<data.at(1).size()<<std::endl;
    //std::cout<<"The size of a node feature: "<<get_feature_size()<<std::endl;
}
