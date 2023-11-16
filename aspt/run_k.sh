#!/bin/bash
k=16
./sspmm_32 ../data/pubmed.csv $k
./sspmm_32 ../data/flickr.csv $k
./sspmm_32 ../data/reddit.csv $k
./sspmm_32 ../data/ppi.csv $k
./sspmm_32 ../data/amazon.csv $k
./sspmm_32 ../data/yelp.csv $k 
