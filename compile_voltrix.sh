#!/bin/bash
set -e
nvcc -O3 -std=c++17 -arch=sm_90 -Xcompiler=-fopenmp \
     -o voltrix_spmm voltrix_spmm.cu -lcusparse
echo "Build OK: voltrix_spmm"
