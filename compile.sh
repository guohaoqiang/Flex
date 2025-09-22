#!/bin/bash

make clean
# uniform diagonal tiling, no wing enablement
# make flex NON_UNIFORM_TILE= WING_ENABLED=

# non-uniform diagonal tiling, no wing enablement
make flex NON_UNIFORM_TILE=1 WING_ENABLED=

# uniform diagonal tiling, wing enablement
# make flex NON_UNIFORM_TILE= WING_ENABLED=1

# non-uniform diagonal tiling, wing enablement
# make flex NON_UNIFORM_TILE=1 WING_ENABLED=1
