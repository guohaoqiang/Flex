GP_ROOT = ../../gp
NP_ROOT = $(GP_ROOT)/lib

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda
# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(GP_ROOT)/include -I$(CUDA_ROOT_DIR)/include
CUDA_INC_DIR+= -I$(GP_ROOT)/lib
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart -lcusparse -lcublas

LIBRARIES =
ifneq ($(MAKECMDGOALS),clean)
  DUMMY:= $(shell $(MAKE) -C $(NP_ROOT))
  LIBRARIES += $(shell $(NP_ROOT)/ld-flags)
endif

CXX = g++
#INC_FLAGS := -I/home/grads/ghaoqi1/opt/glog/build -I/home/grads/ghaoqi1/opt/glog/src -I/home/grads/ghaoqi1/opt/gflags/build/include
#LIB_FLAGS := -L/home/grads/ghaoqi1/opt/glog/build -L/home/grads/ghaoqi1/opt/gflags/build/lib 
#INC_FLAGS := -I./glog/build -I./glog/src -I./gflags/build/include
#LIB_FLAGS := -L./glog/build -L./gflags/build/lib 
CXXFLAGS = -std=c++14 -O3
CXX_LINK_FLAGS = -lpthread
WARN_FLAGS = -g -Wall -Wextra -Wabi -Wctor-dtor-privacy -Wnon-virtual-dtor -Wreorder -Wstrict-null-sentinel -Woverloaded-virtual -Wshadow -Wcast-align -Wpointer-arith -Wwrite-strings -Wundef -Wredundant-decls

NVXX = nvcc
# Turing:2080 super
#NVXXFLAGS = -arch sm_75
# Ampere:3080,3090
#NVXXFLAGS = -arch sm_86
# Ada: 4090
#NVXXFLAGS = -arch sm_89
# Hopper: H100
NVXXFLAGS = -arch sm_90
#CUFLAGS = $(shell $(NP_ROOT)/cc-flags)
#NVXX_LINK_FLAGS = -lcusparse -lcublas -lcudart

#SASS_FILTER = $(GP_ROOT)/util/sass-filter --show-source-code

SRC = .
CPP_SRC = ${SRC}/cpp
CUDA_SRC = ${SRC}/cuda
INC = .
OBJ = ./obj

CC_SOURCE = $(wildcard ${SRC}/*.cc)
CU_SOURCE = $(wildcard ${SRC}/*.cu)

CC_OBJECTS = $(patsubst %.cc,$(OBJ)/%.cc.o,$(notdir ${CC_SOURCE}))
CU_OBJECTS = $(patsubst %.cu,$(OBJ)/%.cu.o,$(notdir ${CU_SOURCE}))
#OBJECTS = main.o graphdata.o OptionParser.o acc.o pec.o analysis.o 

test:
	@echo ${SRC}
	@echo ${CC_OBJECTS}
	@echo ${CU_OBJECTS}

flex: $(CC_OBJECTS) $(CU_OBJECTS)
	$(CXX) -o $@ $^ $(CXX_LINK_FLAGS) $(WARN_FLAGS) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(LIBRARIES)
	#$(CXX) -o $@ $^ $(CXX_LINK_FLAGS) $(NVXXFLAGS) $(WARN_FLAGS) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(LIB_FLAGS)

$(OBJ)/%.cc.o: $(SRC)/%.cc $(INC)/%.h
	$(CXX) -c $< -o $@ $(WARN_FLAGS) $(CXXFLAGS) $(CUDA_INC_DIR)
	#$(CXX) -c $< -o $@ $(WARN_FLAGS) $(CXXFLAGS) $(CXX_LINK_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(CUDA_INC_DIR) $(CUDA_LIB_DIR)

$(OBJ)/%.cu.o: $(SRC)/%.cu $(INC)/%.cuh
	$(NVXX) -c $< -o $@ $(CUDA_INC_DIR) $(NVXXFLAGS) $(LIBRARIES)
	#$(NVXX) -c $< -o $@ $(NVXXFLAGS) $(NVXX_LINK_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(CXX_LINK_FLAGS) $(CUDA_LIB_DIR) $(CUDA_INC_DIR)


.PHONY: clean

clean:
	find ${OBJ} -name *.o -exec rm -r {} \;
	rm -f flex
