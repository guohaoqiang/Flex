#!/bin/bash
k=${1:-64}
./voltrix_spmm ./data/pubmed.csv $k ./data/flickr.csv
