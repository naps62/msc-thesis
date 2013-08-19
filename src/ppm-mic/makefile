default: bin
BINNAME := ppm-rr
UNAME   := $(shell uname)

SRC_DIR           := src
CUDA_SUPPORT      := no
CUDA_INSTALL_PATH := /usr/local/cuda-5.0

EXCLUDE_DIR :=

BEAST_ROOT := $(HOME)/projects/beast
FREEIMAGE_ROOT := /share/edu-mei/freeimage/3.15.4

#ARCH := $(shell getconf LONG_BIT)
ARCH := mic
OMP  := openmp

CFLAGS  := -m$(ARCH) -O3 -Wall -Wextra -std=c++11
NVFLAGS := -m$(ARCH) -O3 -std=c++11
LDFLAGS := -m$(ARCH)
ARFLAGS :=
LD_LIBS       := boost_program_options boost_system boost_thread boost_chrono freeimage
INCLUDE_DIR   := include /usr/local/include /usr/include /usr/include/GL $(CUDA_INSTALL_PATH)/include $(BOOST_ROOT)/include $(FREEIMAGE_ROOT)/include
LD_LIBS_DIR   := /home/cpd19808/projects/msc-thesis/src/ppm-mic /usr/lib /usr/local/lib /opt/local/lib $(CUDA_INSTALL_PATH)/lib $(BOOST_ROOT)/lib
CUDA_LIBS_DIR := $(CUDA_INSTALL_PATH)/lib64 $(CUDA_INSTALL_PATH)/lib


#
# Do not change
#
ifdef (DEFINES)
ifeq ($(CUDA_SUPPORT),yes)
DEFINES := -Xcompiler "$(CFLAGS)"
endif
endif

MODULES   := ${shell find $(SRC_DIR) -type d -print | sed 1d | cut -d'/' -f2,3,4,5 }
SRC_DIR   := $(SRC_DIR) $(addprefix src/,$(MODULES))
BUILD_DIR := build      $(addprefix build/,$(MODULES))

SRC_CPP := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
SRC_C   += $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.c))
SRC_CU  += $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cu))
SRC     := $(SRC_CPP) $(SRC_C)
OBJ     := $(patsubst src/%.cpp,build/%.o,$(SRC_CPP))
OBJ     += $(patsubst src/%.c,build/%.o,$(SRC_C))
OBJ     += $(patsubst src/%.cu,build/%.o,$(SRC_CU))
DEPS    += $(patsubst build/%.o,build/%.d,$(OBJ))

INCLUDES    += $(addprefix -I,$(INCLUDE_DIR)/)
LD_LIBS     := $(addprefix -l,$(LD_LIBS))
LD_LIBS_DIR := $(addprefix -L,$(LD_LIBS_DIR))
ifeq ($(CUDA_SUPPORT),yes)
LD_LIBS_DIR += $(addprefix -L,$(CUDA_LIBS_DIR))
endif

ifeq ($(CUDA_SUPPORT),yes)
OMP := $(addprefix -Xcompiler -f,$(OMP))
else
OMP := $(addprefix -f,$(OMP))
endif

vpath %.cpp $(SRC_DIR)
vpath %.c $(SRC_DIR)

ifeq ($(CUDA_SUPPORT),yes)
NVFLAGS := $(NVFLAGS) -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30
LDFLAGS := $(LDFLAGS) -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30
vpath %.cu $(SRC_DIR)

CC   := nvcc -x c   $(NVFLAGS) -Xcompiler "$(CFLAGS)"
CXX  := nvcc -x c++ $(NVFLAGS) -Xcompiler "$(CFLAGS)"
NVCC := nvcc -x cu  $(NVFLAGS) -Xcompiler "$(CFLAGS)"
LD   := nvcc -link  $(LDFLAGS)
AR   := ar
else
CC   := icc  $(CFLAGS)
CXX  := icc  $(CFLAGS)
NVCC := nvcc $(NVFLAGS) -Xcompiler "$(CFLAGS)"
LD   := icc  $(LDFLAGS)
endif

define make-files

$1/%.d: %.cpp
	@echo " DEPS   $$@"
	@$(CXX) -M $(DEFINES) $(OMP) $(INCLUDES) $$< -o $$@

$1/%.d: %.c
	@echo " DEPS   $$@"
	@$(CC) -M $(DEFINES) $(OMP) $(INCLUDES) $$< -o $$@

$1/%.o: %.cpp
	@echo " CXX    $$<"
	@$(CXX) -c $(DEFINES) $(OMP) $(INCLUDES) $$< -o $$@

$1/%.o: %.c
	@echo " CC     $$<"
	@$(CC) -c $(DEFINES) $(OMP) $(INCLUDES) $$< -o $$@
endef

.PHONY: all checkdirs clean asd


bin/$(BINNAME): $(DEPS) $(OBJ)
	@echo " LD    $(BINNAME)"
	$(LD) $(OMP) $(LD_LIBS_DIR) $(LD_LIBS) $(OBJ) -o $@

lib/$(LIBNAME): $(DEPS) $(OBJ)
	@echo " AR     $(LIBNAME)"
	@$(AR) $(ARFLAGS) -r "lib/$(LIBNAME)" $(OBJ) 2> /dev/null

checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
	rm -rf bin/$(BINNAME)
	rm -rf lib/$(LIBNAME)

bin: checkdirs bin/$(BINNAME)
lib: checkdirs lib/$(LIBNAME)

$(foreach bdir,$(BUILD_DIR),$(eval $(call make-files,$(bdir))))
