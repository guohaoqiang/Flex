#!/bin/bash

rm flex-tile-nperf.csv
#rm ge_spmm_roofline.csv

k=8
./flex ./data/pubmed.csv $k
#./flex ./data/flickr.csv $k
#./flex ./data/reddit.csv $k
#./flex ./data/ppi.csv $k
#./flex ./data/amazon.csv $k
#./flex ./data/yelp.csv $k
#./flex ./data/soc-sign-epinions.csv $k
#./flex ./data/wiki-Vote.csv $k
#./flex ./data/b_mat.csv $k


