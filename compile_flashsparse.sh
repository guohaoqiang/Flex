#!/bin/bash
set -e

echo "Compiling flashsparse_spmm.cu ..."
nvcc -O3 -std=c++17 -arch native -o flashsparse_spmm flashsparse_spmm.cu -lcusparse

echo "Done. Binary: ./flashsparse_spmm"