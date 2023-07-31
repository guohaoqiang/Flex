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
#include <numeric>
#include "common.h"
#include "order_deg.cuh"
#include "order_rcm.cuh"
#include "order_gorder.cuh"

#define COL_MAJ_TILE
#define VO_RECOVER

class DataLoader{
public:
    DataLoader(const std::string& st, const int di);
    DataLoader(const DataLoader& dl);
    ~DataLoader() { freeAll(); }
    void cuda_alloc_cpy();
    void c_cuSpmm_run(Perfs& perfRes);
    void gpuC_zero();
    
    std::vector<unsigned int> rowPtr;
    std::vector<unsigned int> col;
    std::vector<float> vals;

    std::vector<int> vo_mp;
    
    bool compare();
    void print_data();
    
    std::vector<float> cpuX; // n * dim
    std::vector<float> cpuW; // dim * c
	std::vector<float> cpuC; // n * c
	std::vector<float> cpuRef1; // n * c
	std::vector<float> cpuRef2; // n * c
    std::vector<float> h_ref_c; // Computed using cuSpmm;
    
    std::string vertex_order_abbr; // Short abbreviation for vertex ordering.
    unsigned int *rowPtr_dev = nullptr;
    unsigned int *col_dev = nullptr;
    float *vals_dev = nullptr;
    
    int64_t gpuX_bytes, C_elts, gpuC_bytes;
    float *gpuX = nullptr;
    float *gpuW = nullptr;
    float *gpuC = nullptr;
    float *gpuRef1 = nullptr;
    float *gpuRef2 = nullptr;

    size_t m, n, dim, c, nnz;
    std::string graph_name;

    void freeA(){
        cuda_freez(rowPtr_dev);
        cuda_freez(col_dev);
        cuda_freez(vals_dev);
#ifdef AXW
        cuda_freez(gpuW);
        cuda_freez(gpuRef1);
        cuda_freez(gpuRef2);
#endif
    }
    void freeAll(){
        cuda_freez(rowPtr_dev);
        cuda_freez(col_dev);
        cuda_freez(vals_dev);
        if (vertex_order_abbr=="OVO") cuda_freez(gpuX);
        cuda_freez(gpuC);
#ifdef AXW
        cuda_freez(gpuW);
        cuda_freez(gpuRef1);
        cuda_freez(gpuRef2);
#endif
    }
};

class DataLoaderDFS : public DataLoader
{
public:
  DataLoaderDFS(const DataLoader& dl);
};

class DataLoaderDeg : public DataLoader
{
public:
  DataLoaderDeg(const DataLoader& dl);
};

class DataLoaderRcm : public DataLoader
{
public:
  DataLoaderRcm(const DataLoader& dl);
};

class DataLoaderGorder : public DataLoader
{
public:
  DataLoaderGorder(const DataLoader& dl);
};
#endif /* DATALOADER_H */
