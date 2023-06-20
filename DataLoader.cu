#include "DataLoader.cuh"
DataLoader::DataLoader(const std::string& data_path, const int di):dim(di){
    std::string data_name = data_path.substr(data_path.find_last_of("/")+1);
    graph_name = data_name.substr(0, data_name.find(".")); 
    //cpuA = std::make_unique<CSR>();
    std::fstream fin;
    fin.open(data_path,std::ios::in);
    //std::cout<<this->data_path<<std::endl;
    //std::cout<<name0<<std::endl;
    //std::cout<<this->data_path+"\/"+"n_"+name0+".csv"<<std::endl;
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
            vals.push_back((float)rand()/RAND_MAX/100);        
        }
    }else{
        std::getline(fin,line);
        std::stringstream ss3(line);
        while(std::getline(ss3,word,',')){
            vals.push_back(std::stof(word));        
        }
        fin.close(); 
    }
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
    //gpuA = std::make_unique<dCSR>();
#ifdef AXW
    //LOG(INFO) << "Initialize X & W ...";
    //printf("%d of %s, Initialize X & W ...",__LINE__,__FILE__);
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
    
    CUDA_CHECK(cudaMalloc(&gpuC, sizeof(float) * dim * m));
    CUDA_CHECK(cudaMemset(gpuC, 0, sizeof(float)*m*dim));
    
    for (int i=0; i<n*dim; ++i){
        //cpuX[i] = (float)rand()/RAND_MAX;
        cpuX.push_back(1.0);
    }
    CUDA_CHECK(cudaMalloc(&gpuX, sizeof(float) * n * dim));
    CUDA_CHECK(cudaMemcpy(gpuX, cpuX.data(), sizeof(float)*n*dim, cudaMemcpyHostToDevice));
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
