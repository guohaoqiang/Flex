#!/bin/bash

rm flex-tile-nperf.csv
#rm ge_spmm_roofline.csv

k=4
./flex ./data/pubmed.csv $k
./flex ./data/flickr.csv $k
./flex ./data/reddit.csv $k
./flex ./data/ppi.csv $k
#./flex ./data/amazon.csv $k
./flex ./data/yelp.csv $k
#./flex ./data/soc-sign-epinions.csv $k
#./flex ./data/wiki-Vote.csv $k
#./flex ./data/a_mat.csv 128


