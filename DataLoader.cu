#include "DataLoader.cuh"
#include "edgelist.cuh"
#include "flex.cuh"

#include <ranges>

constexpr bool opt_debug = false;

DataLoader::DataLoader(const std::string& data_path, const int di)
  :dl_original(this),dim(di){
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
            //unsigned int temp_v = (rand()<<16)|rand();
            //temp_v = (temp_v&0x7fffff) | 0x40000000; 
            //vals.push_back( *((float*)&temp_v) - 3.0f );
            vals.push_back( 2*(float)rand()/(float)RAND_MAX - 1.0f );
        }
    }else{
        std::getline(fin,line);
        std::stringstream ss3(line);
        while(std::getline(ss3,word,',')){
            vals.push_back( opt_debug ? 1 : std::stof(word) );
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

    vector< map<int,float> > e_inv(m);
    n_edges_one_way = 0;
    n_edges_asymmetric = 0;
    n_nodes_z_out = 0;
    n_nodes_z_in = 0;
    n_nodes_z_deg = 0;

    for ( int r: views::iota(size_t(0),m) )
      for ( int e: views::iota(rowPtr[r],rowPtr[r+1]) )
        {
          auto dst = col[e];
          assert( e_inv[dst].count(r) == 0 );
          e_inv[dst][r] = vals[e];
        }

    for ( int r: views::iota(size_t(0),m) )
      for ( int e: views::iota(rowPtr[r],rowPtr[r+1]) )
        if ( e_inv[r].count(col[e]) == 0 ) n_edges_one_way++;
        else if ( e_inv[r][col[e]] != vals[e] ) n_edges_asymmetric++;

    for ( int r: views::iota(size_t(0),m) )
      {
        const bool z_out = rowPtr[r] == rowPtr[r+1];
        if ( z_out ) n_nodes_z_out++;
        const bool z_in = e_inv[r].empty();
        if ( z_in ) n_nodes_z_in++;
        if ( z_in && z_out ) n_nodes_z_deg++;
      }

    is_directed = n_edges_one_way;

    vo_mp.resize(m);
    std::iota(vo_mp.begin(), vo_mp.end(), 0);
    cuda_alloc_cpy();

    if (false){
        getDegDist();
    }
}

void
DataLoader::getDegDist(){
    vector<int> deg(5,0);

    for (int i=0; i<rowPtr.size()-1; ++i){
        int nbs = rowPtr[i+1]-rowPtr[i];
        if ( nbs<=8 )   deg[0]++;
        else if ( nbs <=16 )  deg[1]++;
        else if ( nbs <=32 )  deg[2]++;
        else if ( nbs <=256 )  deg[3]++;
        else  deg[4]++;
    }

    printf("( 0, 8]: %f\n",deg[0]*1.0/m);
    printf("( 8, 16]: %f\n",deg[1]*1.0/m);
    printf("( 16, 32]: %f\n",deg[2]*1.0/m);
    printf("( 32, 256]: %f\n",deg[3]*1.0/m);
    printf("( 256, +OO): %f\n",deg[4]*1.0/m);
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
    CUDA_CHECK(cudaMemcpy(rowPtr_dev, rowPtr.data(), sizeof(unsigned int)*(m+1), cudaMemcpyHostToDevice));
    
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

              if ( opt_debug )
                cpuX.push_back(1);
              else
                cpuX.push_back( 2*(float)rand()/(float)RAND_MAX - 1.0f );
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


DataLoader::DataLoader(const DataLoader& dl):dl_original(&dl)
{
  #define CPY(m) m = dl.m
  CPY(m); CPY(n); CPY(dim); CPY(c); CPY(nnz); CPY(graph_name);
  #undef CPY
}


void
DataLoader::perm_apply(const DataLoader& dl)
{
  assert( rowPtr.empty() );

  rowPtr.reserve( n+1 );
  rowPtr.push_back( 0 );
  vector<unsigned int> vold_to_new(n,n);

  for ( auto v_new: views::iota(0ul,n) )
    {
      const int v_old = vo_mp[ v_new ];
      assert( vold_to_new[ v_old ] == n );
      vold_to_new[ v_old ] = v_new;
      rowPtr.push_back( rowPtr.back() + dl.rowPtr[v_old+1] - dl.rowPtr[v_old] );
    }

  col.resize( dl.col.size() );
  vals.resize( dl.col.size() );

  //
  // Copy destinations (col) and edge weights (vals) from dl to this object.
  //
  for ( auto v_old: views::iota(0ul,n) )
    {
      const auto v_new = vold_to_new[v_old];
      const int d = dl.rowPtr[v_old+1] - dl.rowPtr[v_old];

      // Sort destinations.  Tiling algorithm needs dests sorted.
      vector< pair<float,uint> > perm;  perm.reserve(d);
      const auto e_idx_old = dl.rowPtr[ v_old ];
      for ( auto e: views::iota( e_idx_old, e_idx_old + d ) )
        perm.emplace_back( dl.vals[ e ], vold_to_new[ dl.col[ e ] ] );
      ranges::sort(perm, ranges::less(), [](auto& v) { return v.second; } );

      uint e_idx_new_i = rowPtr[v_new];
      for ( auto [val, dst_new]: perm )
        {
          col[ e_idx_new_i ] = dst_new;
          vals[ e_idx_new_i++ ] = val;
        }
    }

    if ( false ){
        print_ord(this->vertex_order_abbr, vold_to_new, this->rowPtr, col);
    }
  //
  // Perform a rough test of whether the two graphs match.
  //

  vector<int64_t> check_old(n);
  vector<int64_t> check_new(n);
  vector<double> checkw_old(n);
  vector<double> checkw_new(n);

  for ( uint v_old: views::iota(0ul,n) )
    {
      const auto v_new = vold_to_new[v_old];
      const int d = dl.rowPtr[v_old+1] - dl.rowPtr[v_old];
      assert( rowPtr[v_new] + d == rowPtr[v_new+1] );
      const int inc = v_old & 0xf;
      for ( auto n_idx: views::iota(0,d) )
        {
          const uint e_idx_old = dl.rowPtr[v_old] + n_idx;
          const uint e_idx_new = rowPtr[v_new] + n_idx;
          check_old[ dl.col[ e_idx_old ] ] += inc;
          check_new[ col[ e_idx_new ] ] += inc;
          checkw_old[ dl.col[ e_idx_old ] ] += dl.vals[ e_idx_old ];
          checkw_new[ col[ e_idx_new ] ] += vals[ e_idx_new ];
        }
    }

  for ( uint v_old: views::iota(0ul,n) )
    {
      assert( check_old[v_old] == check_new[ vold_to_new[v_old] ] );
      assert( checkw_old[v_old] == checkw_new[ vold_to_new[v_old] ] );
    }
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
  // Perform Depth-First Search (DFS) on Each Component
  //

  rowPtr.reserve( n+1 );
  rowPtr.push_back( 0 );

  for ( int dfs_root_vo_idx = 0; dfs_root_vo_idx < n; )
    {
      auto root = dst_iter_make(dfs_root_vo_idx);
      vector< decltype(root) > stack { root };
      if ( dfs_root_vo_idx ) vo_to_dfs[ dfs_root_vo_idx ] = rowPtr.size() - 1;
      rowPtr.push_back( rowPtr.back() + root.size() );

      while ( !stack.empty() )
        {
          auto& dst_iter = stack.back();
          while ( dst_iter && vo_to_dfs[ dst_iter.front() ] )
            dst_iter.advance(1);
          if ( !dst_iter ) { stack.pop_back();  continue; }

          const uint dst_vo  = dst_iter.front();  dst_iter.advance(1);
          const uint dst_dfs = rowPtr.size() - 1;
          vo_to_dfs[ dst_vo ] = dst_dfs;
          auto dst_node_iterator = dst_iter_make( dst_vo );
          stack.push_back( dst_node_iterator );
          // Update edge list pointer. (Row Number to vals/col array index.)
          rowPtr.push_back( rowPtr.back() + dst_node_iterator.size() );
        }

      if ( rowPtr.size() > n ) break;

      // Find a vertex that has not been searched.
      while ( ++dfs_root_vo_idx < n && vo_to_dfs[dfs_root_vo_idx] );
      assert( dfs_root_vo_idx < n );
    }

  assert( rowPtr.size() == n + 1 );

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
      assert( rowPtr[src_dfs] + d == rowPtr[src_dfs+1] );

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
    if ( false ){
        print_ord(vertex_order_abbr, vo_to_dfs, rowPtr, col);
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

DataLoaderRabbit::DataLoaderRabbit(const DataLoader& dl):DataLoader(dl)
{
  /// Order vertices based on modularity, implementing several variations.
  //
  // Original Iterative Serial Algorithm: 
  //   Shiokawa 13 AAAI "Fast algorithm for modularity based clustering."
  //   https://aaai.org/papers/455-fast-algorithm-for-modularity-based- graph-clustering/
  //   Set opt_iterative = true;
  //
  // Parallel Implementation. Rabbit properly refers to the parallel version.
  //   Arai 16 IPDPS
  //   https://ieeexplore.ieee.org/document/7515998
  //
  // Idea for Hub Grouping and Sorting
  //   Balaji 23 ISPASS
  //   https://ieeexplore.ieee.org/document/10158154
  //   Set opt_h

  vertex_order_abbr = "RBT";
  gpuX = dl.gpuX;

  // Variations from Balaji 23 ISPASS 
  //
  const bool opt_hub_group = false;
  const bool opt_hub_sort = false;

  // When true, operate in degree order of current set of vertices.
  // When true, closer to Shiokawa 17 AAAI.
  const bool opt_iterative = true;

  assert( dl.rowPtr.size() == n + 1 );
  auto dst_iter_make = [&](uint s)
  { return ranges::subrange(&dl.col[dl.rowPtr[s]],&dl.col[dl.rowPtr[s+1]]); };

  struct Tree_Node {
    Tree_Node(Tree_Node *a, Tree_Node *b):lchild(a),rchild(b),v_idx(-1){}
    Tree_Node(int v):lchild(nullptr),rchild(nullptr),v_idx(v){}
    Tree_Node(){}
    Tree_Node *lchild, *rchild;
    int v_idx;
    void leaves_apply( vector<int>& perm ) {
      if ( lchild ) { lchild->leaves_apply(perm); rchild->leaves_apply(perm); }
      else          { perm.push_back( v_idx ); } }
  };

  struct Vertex {
    map<int,int> dst_wht; // NOT the graphs original edge weight.
    Tree_Node leaf_node, cluster_node, *tree_node;
    int deg, deg_orig, round;
  };

  vector<uint> v_this_round(n);
  vector<Vertex> mgraph(n);
  int n_edges = 0;

  // If true, perform clustering on a directed version of the graph.
  const bool force_undirected = dl.is_directed;

  // Prepare structure used for Rabbit's community detection.
  //
  for ( auto v: views::iota(0ul,n) )
    {
      Vertex& vo = mgraph[v];
      // This edge weight is used only for computing modularity. 
      for ( auto d: dst_iter_make(v) ) 
        if ( d != v )
          {
            vo.dst_wht[d] = 1;
            if ( force_undirected ) mgraph[d].dst_wht[v] = 1;
          }
      vo.deg_orig = vo.deg = vo.dst_wht.size();
      n_edges += vo.deg;
      vo.leaf_node = Tree_Node(v);
      vo.tree_node = &vo.leaf_node;
      v_this_round[v] = v;
      vo.round = 0;
    }

  // Note: cluster_shyness = 1 is the value used in Arai 16.
  const double opt_cluster_shyness = 1;
  const double two_m_inv = opt_cluster_shyness / double( 2 * n_edges );

  vector<uint> v_next_round;

  for ( int round = 1; !v_this_round.empty(); round++ )
    {
      ranges::sort
        ( v_this_round, ranges::less(), [&](auto i){ return mgraph[i].deg; });

      if ( opt_iterative )
        printf("Rabbit round %2d, n elts %zd\n",round,v_this_round.size());

      for ( auto u: v_this_round )
        {
          Vertex& uo = mgraph[u];
          if ( opt_iterative && uo.round == round ) continue;

          // Find neighbor of u with the largest change in modularity. (Delta Q)
          double dQ_max = -1;
          int v = -1;
          const double dv_2m = uo.deg * two_m_inv;
          for ( auto [d,w]: uo.dst_wht )
            if ( set_max( dQ_max, w - mgraph[d].deg * dv_2m ) ) v = d;
          if ( dQ_max <= 0 ) continue;

          // Modularity improves, so u is merged into v.
          //
          Vertex& vo = mgraph[v];
          vo.deg += uo.deg;

          // Update links affected by "removal" of u.
          for ( auto [d,w]: uo.dst_wht )
            {
              if ( d == v ) continue;
              vo.dst_wht[d] += w;
              auto& dodw = mgraph[d].dst_wht;
              if ( !dodw.contains(u) ) continue;
              dodw[v] += dodw[u];
              dodw.erase(u);
            }
          vo.dst_wht.erase(u);

          // Add to dendrogram for this cluster.
          uo.cluster_node = Tree_Node(vo.tree_node,uo.tree_node);
          uo.tree_node = nullptr;
          vo.tree_node = &uo.cluster_node;

          if ( !opt_iterative || vo.round == round ) continue;
          vo.round = round;
          v_next_round.push_back(v);
        }

      if ( !opt_iterative ) break;
      assert( v_next_round.size() < v_this_round.size() );
      swap(v_this_round,v_next_round);
      v_next_round.clear();
    }

  // Sanity Check
  int deg_sum = 0;
  for ( auto& vo: mgraph ) if ( vo.tree_node ) deg_sum += vo.deg;
  assert( deg_sum == n_edges );

  int n_communities = 0;
  int n_hub_edges = 0;
  int n_hub_vertices = 0;
  vector<int> vo_to_community(n);
  vector<int> perm_rbt; perm_rbt.reserve(n);

  // Compute Modularity. This time keep the 1/(2m) factor. (m is n_edges)
  double q = 0;
  const double twom_inv = 1.0 / ( 2 * n_edges );
  const double twom_inv_sq = twom_inv * twom_inv;

  // Traverse dendrograms.
  for ( auto& vo: mgraph )
    if ( vo.tree_node )
      {
        int w_total = 0;
        for ( auto [_,w]: vo.dst_wht ) w_total += w;
        q += ( vo.deg - w_total ) * twom_inv - vo.deg * vo.deg * twom_inv_sq;
        const int c_idx = ++n_communities;
        const auto c_start = perm_rbt.size();
        vo.tree_node->leaves_apply(perm_rbt); // Append perm_rbt with leaves.
        const auto c_end = perm_rbt.size();
        for ( auto v_new: views::iota(c_start,c_end) )
          vo_to_community[ perm_rbt[v_new] ] = c_idx;
      }

  // Optionally apply special placement for hub (inter-community) nodes.
  //
  vo_mp.reserve(n);
  vector<int> v_old_hub; v_old_hub.reserve(n/2);
  for ( auto v_new: views::iota(0ul,n) )
    {
      const auto v_old = perm_rbt[v_new];
      const auto c_idx = vo_to_community[ v_old ];
      int n_hub_edges_here = 0;
      for ( auto d: dst_iter_make(v_old) )
        if ( vo_to_community[d] != c_idx ) n_hub_edges_here++;
      n_hub_edges += n_hub_edges_here;
      if ( n_hub_edges_here ) n_hub_vertices++;
      if ( opt_hub_group && n_hub_edges_here ) v_old_hub.push_back( v_old );
      else                                     vo_mp.push_back( v_old );
    }
  if ( opt_hub_sort && opt_hub_group )
    ranges::sort
      ( v_old_hub, ranges::less(), [&](auto i){ return mgraph[i].deg_orig; });

  for ( auto v: v_old_hub ) vo_mp.push_back( v );

  printf("Shyness %.1f. Iter %d  GH %d GS %d "
         "Rabbit found %d communities, %d hubs %.3f%%, edges %d. Mod %f\n",
         opt_cluster_shyness, opt_iterative, opt_hub_group, opt_hub_sort,
         n_communities, n_hub_vertices,
         100.0 * n_hub_vertices / n, n_hub_edges,q);

  perm_apply(dl);
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

  vector<unsigned int> vo_to_gorder;  // original Vertex Order to Gorder order

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

    if ( false ){
        print_ord(vertex_order_abbr, vo_to_gorder, rowPtr, col);
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
