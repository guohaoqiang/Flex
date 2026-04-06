#!/bin/bash
set -e

k=64
./flashsparse_spmm ./data/pubmed.csv $k \
    ./data/flickr.csv
# Add more datasets after the first CSV:
#   ./data/reddit.csv ./data/ppi.csv ./data/amazon.csv ./data/yelp.csv