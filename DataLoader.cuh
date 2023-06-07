#ifndef DATALOADER_H
#define DATALOADER_H 
//#include <glog/logging.h>
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
//#define T float
class CSR
{
public:
    std::vector<unsigned int> row;
    std::vector<unsigned int> col;
    std::vector<float> vals;
    size_t nnz, r, c;
};

class dCSR
{
public:
    unsigned int *row = nullptr;
    unsigned int *col = nullptr;
    float *vals = nullptr;
    size_t nnz, r, c;
};

class DataLoader{
public:
    DataLoader(const std::string& st, const int di, bool genXW = true);
    
    bool transfer();
    bool alloc();
    bool compare();
    void print_data();
    
    std::unique_ptr<CSR> cpuA; // n * n 
	std::unique_ptr<float[]> cpuX; // n * dim
	std::unique_ptr<float[]> cpuW; // dim * c
	std::unique_ptr<float[]> cpuC; // n * c
	std::unique_ptr<float[]> cpuRef1; // n * c
	std::unique_ptr<float[]> cpuRef2; // n * c
    
    std::unique_ptr<dCSR> gpuA;
    float *gpuX = nullptr;
    float *gpuW = nullptr;
    float *gpuC = nullptr;
    float *gpuRef1 = nullptr;
    float *gpuRef2 = nullptr;

    size_t n, dim, c;
    std::string graph_name;
};
#endif /* DATALOADER_H */
