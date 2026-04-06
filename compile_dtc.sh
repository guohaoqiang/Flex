#!/bin/bash
# Compile DTC-SpMM as a standalone binary (no Flex dependency).
# Usage:  ./compile_dtc.sh
# Run:    ./dtc_spmm ./data/flickr.csv 64

set -e

NVCC=/usr/local/cuda/bin/nvcc
FLAGS="-O3 -std=c++17 -arch native"
LIBS="-lcusparse"

echo "Compiling dtc_spmm.cu ..."
$NVCC $FLAGS -o dtc_spmm dtc_spmm.cu $LIBS
echo "Done.  Binary: ./dtc_spmm"
