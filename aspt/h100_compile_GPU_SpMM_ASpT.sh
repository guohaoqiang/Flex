#!/bin/bash

cd ASpT_SpMM_GPU
#nvcc -std=c++11 -O3 -gencode arch=compute_90a,code=sm_90a sspmm_32.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o sspmm_32
nvcc -std=c++11 -O3 -arch native sspmm_32.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o sspmm_32
#nvcc -std=c++11 -O3 -gencode arch=compute_90a,code=sm_90a sspmm_128.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o sspmm_128
nvcc -std=c++11 -O3 -arch native sspmm_128.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o sspmm_128
cd ..

