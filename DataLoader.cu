#include "DataLoader.cuh"
DataLoader::DataLoader(const std::string& data_path, const int di, bool genXW):dim(di){
    std::string data_name = data_path.substr(data_path.find_last_of("/")+1,-1);
    graph_name = data_name.substr(0, data_name.find(".")); 
    cpuA = std::make_unique<CSR>();
    std::fstream fin;
    fin.open(data_path,std::ios::in);
    //std::cout<<this->data_path<<std::endl;
    //std::cout<<name0<<std::endl;
    //std::cout<<this->data_path+"\/"+"n_"+name0+".csv"<<std::endl;
    std::string line, word;
    
    std::getline(fin,line);
    std::stringstream ss1(line);
    while(std::getline(ss1,word,',')){
        cpuA->row.push_back(std::stoi(word));        
    }
    
    std::getline(fin,line);
    std::stringstream ss2(line);
    while(std::getline(ss2,word,',')){
        cpuA->col.push_back(std::stoi(word));        
    }
    
    if (data_name == "amazon.csv"){
        // amazon.csv only contains row offset and col indice
        fin.close(); 
        std::cout<<"Amazon n = "<<cpuA->row.size()-1<<std::endl;
        std::cout<<"Amazon nnz = "<<cpuA->col.size()<<std::endl;
        for (size_t i=0; i<cpuA->col.size(); ++i){
            cpuA->vals.push_back((float)rand()/RAND_MAX/100);        
        }
    }else{
        std::getline(fin,line);
        std::stringstream ss3(line);
        while(std::getline(ss3,word,',')){
            cpuA->vals.push_back(std::stof(word));        
        }
        fin.close(); 
    }
    assert(cpuA->col.size()==cpuA->vals.size());
    n = cpuA->row.size()-1; 
    cpuA->r = cpuA->row.size()-1; 
    cpuA->c = cpuA->row.size()-1; 
    cpuA->nnz = cpuA->col.size();

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
    gpuA = std::make_unique<dCSR>();
    if (genXW){
        if (alloc()){
            //LOG(INFO) << "Initialize X & W ...";
            printf("%d of %s, Initialize X & W ...",__LINE__,__FILE__);
            for (int i=0; i<n*dim; ++i){
                //cpuX[i] = (float)rand()/RAND_MAX;
                cpuX[i] = 1;
            }
            for (int i=0; i<c*dim; ++i){
                cpuW[i] = (float)rand()/RAND_MAX;
            }
            //LOG(INFO) << "X & W initialized ...";
            //print_data();
            transfer();
        } 
    }
}

bool DataLoader::transfer(){
    //LOG(INFO) << "Transfer A, X & W to gpu ...";
    CUDA_CHECK(cudaMemcpy(gpuA->row, cpuA->row.data(), sizeof(unsigned int)*(cpuA->r+1), cudaMemcpyHostToDevice));
    //LOG(INFO) << "Transfer A row ...";
    CUDA_CHECK(cudaMemcpy(gpuA->col, cpuA->col.data(), sizeof(unsigned int)*cpuA->nnz, cudaMemcpyHostToDevice));
    //LOG(INFO) << "Transfer A, col ...";
    CUDA_CHECK(cudaMemcpy(gpuA->vals, cpuA->vals.data(), sizeof(T)*cpuA->nnz, cudaMemcpyHostToDevice));
    
    //LOG(INFO) << "Transfer A, X & W to gpu ...";
    CUDA_CHECK(cudaMemcpy(gpuX, &cpuX[0], sizeof(T)*n*dim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuW, &cpuW[0], sizeof(T)*dim*c, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(gpuC, 0, sizeof(T)*n*c));
    //CUDA_CHECK(cudaMemset(gpuRef1, 0, sizeof(T)*n*c));
    CUDA_CHECK(cudaMemset(gpuRef1, 0, sizeof(T)*n*dim));
    CUDA_CHECK(cudaMemset(gpuRef2, 0, sizeof(T)*n*c));
    //LOG(INFO) << "A, X & W have transfered to gpu ...";
    return true;
}

bool DataLoader::alloc(){
    cpuX = std::make_unique<T []>(n*dim);
    cpuW = std::make_unique<T []>(c*dim);
    cpuC = std::make_unique<T []>(n*c);
    cpuRef1 = std::make_unique<T []>(n*c);
    cpuRef2 = std::make_unique<T []>(n*c);
    //memset(cpuC, 0, sizeof(T)*n*c);

    CUDA_CHECK(cudaMalloc(&(gpuA->row), sizeof(unsigned int) * (cpuA->r+1)));
    CUDA_CHECK(cudaMalloc(&(gpuA->col), sizeof(unsigned int) * (cpuA->nnz)));
    CUDA_CHECK(cudaMalloc(&(gpuA->vals), sizeof(T) * (cpuA->nnz)));
    
    CUDA_CHECK(cudaMalloc(&gpuX, sizeof(T) * n * dim));
    CUDA_CHECK(cudaMalloc(&gpuW, sizeof(T) * c * dim));
    CUDA_CHECK(cudaMalloc(&gpuC, sizeof(T) * c * n));
    //CUDA_CHECK(cudaMalloc(&gpuRef1, sizeof(T) * c * n));
    CUDA_CHECK(cudaMalloc(&gpuRef1, sizeof(T) * dim * n));
    CUDA_CHECK(cudaMalloc(&gpuRef2, sizeof(T) * c * n));
    return true;
}
bool DataLoader::compare(){
    for (size_t i=0; i<n*c; ++i){
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
    for(auto it=cpuA->row.begin(); it<cpuA->row.begin()+5; it++)
        std::cout<<(*it)<<" ";
    std::cout<<std::endl;

    std::cout<<"The last 5 elements of rowptr: ";
    for(auto it=cpuA->row.end()-5; it!=cpuA->row.end() ; it++)
        std::cout<<(*it)<<" ";
    std::cout<<std::endl;

    std::cout<<"The first 5 elements of indies: ";
    for(auto it=cpuA->col.begin(); it<cpuA->col.begin()+5 ; it++)
        std::cout<<(*it)<<" ";
    std::cout<<std::endl;

    std::cout<<"The last 5 elements of indies: ";
    for(auto it=cpuA->col.end()-5; it!=cpuA->col.end() ; it++)
        std::cout<<(*it)<<" ";
    std::cout<<std::endl;

    std::cout<<"The first 5 elements of vals: ";
    for(auto it=cpuA->vals.begin(); it<cpuA->vals.begin()+5 ; it++)
        std::cout<<(*it)<<" ";
    std::cout<<std::endl;

    std::cout<<"The last 5 elements of vals: ";
    for(auto it=cpuA->vals.end()-5; it!=cpuA->vals.end() ; it++)
        std::cout<<(*it)<<" ";
    std::cout<<std::endl;
    
    std::cout<<"The first 5 elements of X: ";
    for(auto it=0; it<5 ; it++)
        std::cout<<cpuX[it]<<" ";
    std::cout<<std::endl;
    
    std::cout<<"The first 5 elements of W: ";
    for(auto it=0; it<5 ; it++)
        std::cout<<cpuW[it]<<" ";
    std::cout<<std::endl;
    
    std::cout<<std::endl;
    //std::cout<<"The number of nodes: "<< get_nodes()<<"   Rowptr: "<<data.at(0).size()<<"   Pointer: "<<data.at(1).size()<<std::endl;
    //std::cout<<"The size of a node feature: "<<get_feature_size()<<std::endl;
}
