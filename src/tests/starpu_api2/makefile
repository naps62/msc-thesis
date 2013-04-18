default: bin
BINNAME := starpu_hello_world2
STARPU_MODE := lib
UNAME   := $(shell uname)

SRC_DIR           := src
CUDA_SUPPORT      := yes
CUDA_INSTALL_PATH := /usr/local/cuda-5.0


ARCH := $(shell getconf LONG_BIT)
OMP  := openmp
OMP := $(addprefix -f,$(OMP))

ifeq ($(STARPU_MODE),lib)
$(info using StarPU lib)
STARPU_CFLAGS := ${shell pkg-config --cflags starpu-1.0}
STARPU_LIBS   := $(shell pkg-config --libs   starpu-1.0)
else
$(info using StarPU plugin)
STARPU_CFLAGS := $(shell pkg-config starpu-1.0 --variable=gccplugin)
STARPU_LIBS   := $(shell pkg-config starpu-1.0 --variable=gccplugin)
endif

BEAST_ROOT := $(HOME)/projects/beast
FREEIMAGE_ROOT := /share/edu-mei/freeimage/3.15.4

CFLAGS  := -m$(ARCH) -O3 -Wall -Wno-unused-result $(STARPU_CFLAGS)
NVFLAGS := -m$(ARCH) -O3 $(STARPU_CFLAGS)
LDFLAGS := -m$(ARCH) $(STARPU_LIBS)
ARFLAGS :=
LD_LIBS       := beast glut GLU GL boost_program_options boost_system boost_thread-mt boost_chrono freeimage
INCLUDE_DIR   := include /usr/local/include /usr/include/GL $(CUDA_INSTALL_PATH)/include $(BOOST_ROOT)/include $(BEAST_ROOT)/include $(FREEIMAGE_ROOT)/include
LD_LIBS_DIR   := /usr/lib /usr/local/lib /opt/local/lib $(CUDA_INSTALL_PATH)/lib $(BOOST_ROOT)/lib $(BEAST_ROOT)/lib $(FREEIMAGE_ROOT)/lib/linux $(FREEIMAGE_ROOT)/lib
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
SRC     := $(SRC_CU) $(SRC_CPP) $(SRC_C)
OBJ     := $(patsubst src/%.cpp,build/%.o,$(SRC_CPP))
OBJ     += $(patsubst src/%.c,build/%.o,$(SRC_C))
OBJ     += $(patsubst src/%.cu,build/%.o,$(SRC_CU))
DEPS    += $(patsubst build/%.o,build/%.d,$(OBJ))

INCLUDES    += $(addprefix -I,$(INCLUDE_DIR)/)
LD_LIBS     := $(addprefix -l,$(LD_LIBS))
LD_LIBS_DIR := $(addprefix -L,$(LD_LIBS_DIR))
LD_LIBS_DIR += $(addprefix -L,$(CUDA_LIBS_DIR))

vpath %.cpp $(SRC_DIR)
vpath %.c $(SRC_DIR)


CC   := gcc $(CFLAGS) $(OMP)
CXX  := g++ $(CFLAGS) $(OMP)

ifeq ($(CUDA_SUPPORT),yes)
NVFLAGS := $(NVFLAGS) -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30
LDFLAGS := $(LDFLAGS) -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30
vpath %.cu $(SRC_DIR)
NVCC := nvcc -x cu  $(NVFLAGS) -Xcompiler "$(CFLAGS) $(OMP)"
LD   := nvcc -link  $(LDFLAGS) -Xcompiler "$(OMP)"
AR   := ar
else
NVCC := nvcc $(NVFLAGS) -Xcompiler "$(CFLAGS) $(OMP)"
LD   := gcc  $(LDFLAGS) $(OMP)
endif

define make-files
$1/%.d: %.cu
	@echo " DEPS   $$@"
	@$(NVXX) -M $(DEFINES) $(INCLUDES) $$< -o $$@

$1/%.d: %.cpp
	@echo " DEPS   $$@"
	@$(CXX) -M $(DEFINES) $(INCLUDES) $$< -o $$@

$1/%.d: %.c
	@echo " DEPS   $$@"
	@$(CC) -M $(DEFINES) $(INCLUDES) $$< -o $$@

$1/%.o: %.cu
	@echo " NVCC   $$<"
	@$(NVCC) -c $(DEFINES) $(INCLUDES) $$< -o $$@

$1/%.o: %.cpp
	@echo " CXX    $$<"
	@$(CXX) -c $(DEFINES) $(INCLUDES) $$< -o $$@

$1/%.o: %.c
	@echo " CC     $$<"
	@$(CC) -c $(DEFINES) $(INCLUDES) $$< -o $$@
endef

.PHONY: all checkdirs clean asd


bin/$(BINNAME): $(DEPS) $(OBJ)
	@echo " LD    $(BINNAME)"
	@$(LD) $(LD_LIBS_DIR) $(LD_LIBS) $(OBJ) -o $@

lib/$(LIBNAME): $(DEPS) $(OBJ)
	@echo " AR     $(LIBNAME)"
	@$(AR) $(ARFLAGS) -r "lib/$(LIBNAME)" $(OBJ) 2> /dev/null

$(BUILD_DIR):
	@mkdir -p $@
	@mkdir -p bin

clean:
	rm -rf $(BUILD_DIR)
	rm -rf bin/$(BINNAME)
	rm -rf lib/$(LIBNAME)

bin: $(BUILD_DIR) bin/$(BINNAME)
lib: $(BUILD_DIR) lib/$(LIBNAME)

$(foreach bdir,$(BUILD_DIR),$(eval $(call make-files,$(bdir))))
