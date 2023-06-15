#ifndef DATALOADER_H
#define DATALOADER_H 
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>
#include <string>
#include <cstdlib>
#include <assert.h>
#include <iomanip>
#include "common.h"
class DataLoader{
public:
    DataLoader(const std::string& st, const int di);
    
    std::vector<unsigned int> rowPtr;
    std::vector<unsigned int> col;
    std::vector<float> vals;
    
    bool compare();
    void print_data();
    
    std::vector<float> cpuX; // n * dim
    std::vector<float> cpuW; // dim * c
	std::vector<float> cpuC; // n * c
	std::vector<float> cpuRef1; // n * c
	std::vector<float> cpuRef2; // n * c
    
    unsigned int *rowPtr_dev = nullptr;
    unsigned int *col_dev = nullptr;
    float *vals_dev = nullptr;
    
    float *gpuX = nullptr;
    float *gpuW = nullptr;
    float *gpuC = nullptr;
    float *gpuRef1 = nullptr;
    float *gpuRef2 = nullptr;

    size_t m, n, dim, c, nnz;
    std::string graph_name;
    void freemem(){
        cudaFree(rowPtr_dev);
        cudaFree(col_dev);
        cudaFree(vals_dev);
        cudaFree(gpuX); 
        cudaFree(gpuC); // C = AX
#ifdef AXW
        cudaFree(gpuW);
        cudaFree(gpuRef1);
        cudaFree(gpuRef2);
#endif
    }
};
#endif /* DATALOADER_H */
