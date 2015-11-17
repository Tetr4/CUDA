NVCC		:= /usr/local/cuda/bin/nvcc
LIBRARY		:= ../lib/libcudastereo.so
CCFILES		:=  cuda-jni.cpp
CUFILES		:= ../kernel/kernel.cu

INCLUDES += -I$(JAVA_HOME)/include -I../kernel -I../include
CCFLAGS += --shared --compiler-options '-fPIC'


################################################################################
# Rules and targets

all:    $(LIBRARY)

$(LIBRARY): $(CCFILES) $(CUFILES)
	$(NVCC) $(INCLUDES) $(CUFILES) $(CCFILES) $(CCFLAGS) -o $(LIBRARY)

.PHONY: clean
clean:
	rm $(LIBRARY)
