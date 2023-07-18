#!/bin/bash

wget https://sparse.tamu.edu/MM/SNAP/soc-sign-epinions.tar.gz
#wget https://sparse.tamu.edu/MM/SNAP/web-NotreDame.tar.gz
wget https://sparse.tamu.edu/MM/SNAP/wiki-Vote.tar.gz



find . -name '*.tar.gz' -exec tar xvf {} \;
rm *.tar.gz


g++  mtx2csr.cc -o conv

for i in `ls -d */`
do
    cd ${i}
    ii=${i/\//}
    ../conv ${ii}.mtx ${ii}.csv
    mv ${ii}.csv ../..
    cd ..
    rm -rf ${ii}
done

rm conv
