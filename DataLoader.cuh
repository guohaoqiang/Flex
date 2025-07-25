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
    

    const DataLoader* const dl_original;
    std::vector<unsigned int> rowPtr;
    std::vector<unsigned int> col;
    std::vector<float> vals;

    // Use vo_mp set this to dl permuted.
    void perm_apply(const DataLoader& dl);
    std::vector<int> vo_mp;
    
    bool compare();
    void print_data();
    void print4(int, bool);
    void getDegDist();
    
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
    int64_t uni_nb;
    float *gpuX = nullptr;
    float *gpuW = nullptr;
    float *gpuC = nullptr;
    float *gpuRef1 = nullptr;
    float *gpuRef2 = nullptr;

    bool is_directed;
    // Number of nodes with zero in, zero out edges, or no edges at all.
    int n_nodes_z_out, n_nodes_z_in, n_nodes_z_deg;
    size_t n_edges_one_way;
    size_t n_edges_asymmetric; // Have (u,v) & (v,u) but wht(u,v) != wht(v,u)
    size_t m, n, dim, c, nnz;
    std::string graph_name;

    void print_ord(std::string& vertex_order_abbr, std::vector<unsigned int>& vo_to_dfs, 
            std::vector<unsigned int>& rowPtr, std::vector<unsigned int>& col){
        std::cout<<vertex_order_abbr<<":"<<std::endl;
        std::cout<<"    vo:"<<std::endl;
        for (int v:vo_to_dfs){
            std::cout<<v<<",";
        }
        std::cout<<std::endl<<"    rowPtr:"<<std::endl;
        for (int r:rowPtr){
            std::cout<<r<<",";
        }
        std::cout<<std::endl<<"    col:"<<std::endl;
        for (int c:col){
            std::cout<<c<<",";
        }
        std::cout<<std::endl;
    }
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

class DataLoaderRabbit : public DataLoader
{
public:
  int64_t uni_nb;
  DataLoaderRabbit(const DataLoader& dl);
};

class DataLoaderDFS : public DataLoader
{
public:
  int64_t uni_nb;
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
  int64_t uni_nb;
  DataLoaderGorder(const DataLoader& dl);
};
#endif /* DATALOADER_H */
