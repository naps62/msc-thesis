ROOTSRC_DIR			:= src
CUDA_COMPILATION	:= yes
CUDA_INSTALL_PATH 	:= /usr/local/cuda/
APPNAME				:= photonmapping
BUILD_MODE 			:= opt


ifeq ($(BUILD_MODE),opt)
CFLAGS 		:= -O3
SYMBOLS := -D__RELEASE
else
ifeq ($(BUILD_MODE),debug)
CFLAGS	 	:= -G -g -O0
SYMBOLS := -D__DEBUG
endif
endif

NVFLAGS		:=
LDFLAGS		:=
LD_LIBS		:= gomp cublas cudart GLEW GL glut GLU GL cudpp_x86_64 boost_system freeimage boost_thread
INCLUDE_DIR	:= include /usr/local/include /usr/include/GL /home/rr/apps/cudpp_src_1.1.1/cudpp/include
LD_LIBS_DIR	:= /usr/local/lib /opt/local/lib $(CUDA_INSTALL_PATH)/samples/common/lib ../lib
#LD_LIBS_DIR	:= /home/rr/apps/cudpp_src_1.1.1/lib
BOOST_INCLUDE	:= /opt/local/include
BOOST_LIBS_DIR	:= /usr/local/lib
CUDA_INCLUDE	:= samples/common/inc
CUDA_LIBS_DIR	:= samples/common/lib

OTHER_FLAGS	:=


ifeq ($(CUDA_COMPILATION),yes)
CC        	:= nvcc -x c
CXX		    := nvcc -x c++
NVCC	  	:= nvcc -x cu
LD        	:= nvcc -link
else
CC        	:= gcc
CXX		    := g++
NVCC	  	:= nvcc
LD        	:= gcc
endif


#
# Do not change
#
ifdef (DEFINES)
ifeq ($(CUDA_COMPILATION),yes)
DEFINES   := -Xcompiler
endif
endif

ARCH 	  := $(shell getconf LONG_BIT)
OMP	      := openmp

MODULES   := ${shell find ${ROOTSRC_DIR} -type d -print | sed 1d | cut -d'/' -f2,3,4,5 | sort -u}
SRC_DIR   := $(ROOTSRC_DIR) $(addprefix src/,$(MODULES))
BUILD_DIR := $(addprefix build/,$(MODULES)) build

SRC_CPP   := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
SRC_C     += $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.c))
SRC_CU    += $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cu))
SRC 	  := $(SRC_CU) $(SRC_CPP) $(SRC_C)
OBJ   	  := $(patsubst src/%.cpp,build/%.o,$(SRC_CPP))
OBJ   	  += $(patsubst src/%.c,build/%.o,$(SRC_C))
OBJ   	  += $(patsubst src/%.cu,build/%.o,$(SRC_CU))
DEPS   	  += $(patsubst build/%.o,build/%.d,$(OBJ))

CUDA_INCLUDE	:= $(addprefix $(CUDA_INSTALL_PATH),$(CUDA_INCLUDE))

INCLUDES  := $(addprefix -I,$(ROOTSRC_DIR)/)
INCLUDES  += $(addprefix -I,$(CUDA_INCLUDE)/)
INCLUDES  += $(addprefix -I,$(BOOST_INCLUDE)/)
INCLUDES  += $(addprefix -I,$(LAPACK_INCLUDE)/)
INCLUDES  += $(addprefix -I,$(INCLUDE_DIR)/)


CFLAGS	  := -m$(ARCH) $(CFLAGS)
LDFLAGS	  := -m$(ARCH) $(LDFLAGS)

LD_LIBS	     := $(addprefix -l,$(LD_LIBS))
LD_LIBS_DIR  := $(addprefix -L,$(LD_LIBS_DIR))
LD_LIBS_DIR  += $(addprefix -L$(CUDA_INSTALL_PATH),$(CUDA_LIBS_DIR))
LD_LIBS_DIR  += $(addprefix -L,$(BOOST_LIBS_DIR))
LD_LIBS_DIR  += $(addprefix -L,$(LAPACK_LIBS_DIR))

ifeq ($(CUDA_COMPILATION),yes)
OMP := $(addprefix -Xcompiler -f,$(OMP))
else
OMP := $(addprefix -f,$(OMP))
endif

ifeq ($(CUDA_COMPILATION),yes)
NVFLAGS	  := $(NVFLAGS) $(OTHER_FLAGS) -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30
LDFLAGS	  := $(LDFLAGS) -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30

vpath %.cu $(SRC_DIR)
endif

vpath %.cpp $(SRC_DIR)
vpath %.c $(SRC_DIR)


define make-cuda
$1/%.d: %.cu
	$(NVCC) -M $(CFLAGS) $(DEFINES) $(OMP) $(SYMBOLS) $(NVFLAGS) $(INCLUDES) $$< -o $$@

$1/%.d: %.cpp
	$(NVCC) -M $(CFLAGS) $(DEFINES) $(OMP) $(SYMBOLS) $(NVFLAGS) $(INCLUDES) $$< -o $$@

$1/%.d: %.c
	$(NVCC) -M $(CFLAGS) $(DEFINES) $(OMP) $(SYMBOLS) $(NVFLAGS) $(INCLUDES) $$< -o $$@

$1/%.o: %.cu
	@echo "Compile "$$<
	$(NVCC) -c $(CFLAGS) $(DEFINES) $(OMP) $(SYMBOLS) $(NVFLAGS) $(INCLUDES) $$< -o $$@

$1/%.o: %.cpp
	@echo "Compile "$$<
	$(CXX) -c $(CFLAGS) $(DEFINES) $(OMP) $(SYMBOLS) $(NVFLAGS) $(INCLUDES) $$< -o $$@

$1/%.o: %.c
	@echo "Compile "$$<
	$(CC) -c $(CFLAGS)  $(DEFINES) $(OMP) $(SYMBOLS) $(NVFLAGS) $(INCLUDES) $$< -o $$@
endef


define make-ncuda
$1/%.d: %.cu

$1/%.d: %.cpp

$1/%.d: %.c

$1/%.o: %.cu
	@echo "Compile "$$<
	$(NVCC) -c $(CFLAGS) $(DEFINES) $(OMP) $(SYMBOLS) $(NVFLAGS) $(INCLUDES) $$< -o $$@

$1/%.o: %.cpp
	@echo "Compile "$$<
	$(CXX) -c $(CFLAGS) $(DEFINES) $(OMP) $(SYMBOLS) $(NVFLAGS) $(INCLUDES) $$< -o $$@

$1/%.o: %.c
	@echo "Compile "$$<
	$(CC) -C $(CFLAGS) $(DEFINES) $(OMP) $(SYMBOLS) $(NVFLAGS) $(INCLUDES) $$< -o $$@
endef

.PHONY: all checkdirs clean

all: checkdirs bin/$(APPNAME)

bin/$(APPNAME): $(DEPS) $(OBJ)
	@echo "Creating build tree"
	@mkdir -p bin
	@mkdir -p build
	$(LD) $(LDFLAGS) $(OMP) $(SYMBOLS) $(LD_LIBS_DIR) $(LD_LIBS) $(OBJ) -o $@

checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
	rm -rf bin/$(APPNAME)

ifeq ($(CUDA_COMPILATION),yes)
$(foreach bdir,$(BUILD_DIR),$(eval $(call make-cuda,$(bdir))))
else
$(foreach bdir,$(BUILD_DIR),$(eval $(call make-ncuda,$(bdir))))
endif
