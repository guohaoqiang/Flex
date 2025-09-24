#!/bin/bash

make clean
# uniform diagonal tiling, no wing enablement
make flex NON_UNIFORM_TILE= WING_ENABLED=
./run.sh

make clean
# non-uniform diagonal tiling, no wing enablement
make flex NON_UNIFORM_TILE=1 WING_ENABLED=
./run.sh

make clean
# uniform diagonal tiling, wing enablement
make flex NON_UNIFORM_TILE= WING_ENABLED=1
./run.sh

make clean
# non-uniform diagonal tiling, wing enablement
make flex NON_UNIFORM_TILE=1 WING_ENABLED=1
./run.sh