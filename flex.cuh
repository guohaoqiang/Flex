#ifndef FLEX_H
#define FLEX_H 
#include <iostream>
#include <fstream>
#include "common.h"
#include "DataLoader.cuh"
#include "mat.h"
#include "flex_spmm.cuh"
#define OUTPUTCSV


#define CUBE4X4
#define RECT8X4
#define RECT16X4
#define RECT32X4
#define RECT64X4
#define RECT128X4
#define RECT256X4

#define RECT4X8
#define CUBE8X8
#define RECT16X8
#define RECT32X8
#define RECT64X8
#define RECT128X8
#define RECT256X8

#define RECT4X16
#define RECT8X16
#define CUBE16X16
#define RECT32X16
#define RECT64X16
#define RECT128X16
#define RECT256X16

#define RECT4X32
#define RECT8X32
#define RECT16X32
#define CUBE32X32
#define RECT64X32
#define RECT128X32
#define RECT256X32


/*
void run_test(float* h_res_c, 
                DataLoader& input, 
                const float* mat_b, 
                const vector<vector<int>>& mnk, 
                int idx, 
                int warmup, 
                int runs, 
                Perfs& perfRes);
*/
void run(DataLoader& input);
void cuSpgemm(DataLoader& input, Perfs& perfRes);

template<typename MT, int tm, int tn>
void flexspgemm(float* h_res_c, MT& data, const float* mat_b, Perfs& perfRes);


template<typename TP>
void print(vector<TP>& arr){
    for (size_t i=0; i<arr.size(); ++i)
        std::cout << arr[i] << " ";
    std::cout<<std::endl;
}
#endif /* FLEX_H */
