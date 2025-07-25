#!/bin/bash

cd ASpT_SpMM_GPU
nvcc -std=c++11 -O3 -gencode arch=compute_89,code=sm_89 sspmm_32.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o sspmm_32
nvcc -std=c++11 -O3 -gencode arch=compute_89,code=sm_89 sspmm_128.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o sspmm_128
cd ..

