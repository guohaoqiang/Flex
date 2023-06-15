GP_ROOT ?= ../../gp
# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

NP_ROOT = $(GP_ROOT)/lib

DEFAULT: flex

SRC_FILES = mat.cu main.cu flex.cu DataLoader.cu
OBJ_FILES = $(addsuffix .o,$(basename $(SRC_FILES)))

CXX = g++
NVXX = nvcc

# Recognized both by nvcc and g++
#
IDENTICAL_FLAGS = -O3 -g
IDENTICAL_FLAGS += -I$(CUDA_ROOT_DIR)/include
IDENTICAL_FLAGS += -I$(GP_ROOT)/include

# Used both by nvcc and g++, but requires an -Xcompiler prefix for nvcc
#
COMMON_FLAGS = -Wall -Wno-parentheses -Wno-sign-compare -march=native

# Used only by nvcc.
#
CUCC_ONLY_FLAGS = -std c++14 -arch native
# Turing:2080 super: -arch sm_75
# Ampere:3080,3090: -arch sm_86
# Ada: 4090: -arch sm_89
# Hopper: H100: -arch sm_90

# Used only by g++
#
CXX_ONLY_FLAGS = -std=c++14

LINKFLAGS = -lpthread -lcudart -lcusparse -lcublas

ifneq ($(MAKECMDGOALS),clean)
  DUMMY:= $(shell $(MAKE) -C $(NP_ROOT))
  LINKFLAGS += $(shell $(NP_ROOT)/ld-flags)
  COMMON_FLAGS += $(shell $(NP_ROOT)/cc-flags)
endif


COMMON_FLAGS_PREFIXED = $(COMMON_FLAGS:%=-Xcompiler %)
CXXFLAGS = $(IDENTICAL_FLAGS) $(COMMON_FLAGS) $(CXX_ONLY_FLAGS)
NVXXFLAGS = $(IDENTICAL_FLAGS) $(COMMON_FLAGS_PREFIXED) $(CUCC_ONLY_FLAGS)


test:
	@echo SRC_FILES: $(SRC_FILES)
	@echo OBJ_FILES: ${OBJ_FILES}

# Include dependencies that were created by %.d rules.
#
DHASH = d
ifneq ($(MAKECMDGOALS),clean)
-include $(SRC_FILES:=.$(DHASH))
endif

# Prepare file holding dependencies, to be included in this file.
#
%.cc.$(DHASH): %.cc
	@set -e; rm -f $@; \
	 $(CXX) -M $(CXXFLAGS) $< > $@.$$$$; \
	 sed 's,\($*\)\.o[ :]*,\1-debug.o \1.o $@ : ,g' < $@.$$$$ > $@; \
	 rm -f $@.$$$$

%.cu.$(DHASH): %.cu
	@set -e; rm -f $@; \
	 $(NVXX) -M $(NVXXFLAGS) $< > $@.$$$$; \
	 sed 's,\($*\)\.o[ :]*,\1-debug.o \1.o $@ : ,g' < $@.$$$$ > $@; \
	 rm -f $@.$$$$


%.o: %.cc Makefile
	$(CXX) -c $< -o $@ $(CXXFLAGS)

%.o: %.cu Makefile
	$(NVXX) -c $< -o $@ $(NVXXFLAGS) -rdc=true

flex: $(OBJ_FILES)
	$(NVXX) -o $@ $^ $(NVXXFLAGS) $(LINKFLAGS)


.PHONY: clean

clean:
	/bin/rm -f $(OBJ_FILES) flex \
	*.d *.d.[0-9][0-9][0-9][0-9][0-9] \
	*.d.[0-9][0-9][0-9][0-9][0-9][0-9] \
	*.d.[0-9][0-9][0-9][0-9][0-9][0-9][0-9]
