
################################################################################
#
# Build script for project
#
################################################################################


NVCC		:= /usr/local/cuda/bin/nvcc
# Add source files here
OBJECT		:= kernel.o 
CUFILES		:=  kernel.cu

INCLUDES += -I../include
CCFLAGS += --shared --compiler-options '-fPIC'


################################################################################
# Rules and targets

all:	kernel.o

kernel.o: $(CUFILES)
#	$(NVCC) $(INCLUDES) $(CCFLAGS) --default-stream legacy -o $(OBJECT) $(CUFILES)
	$(NVCC) $(INCLUDES) $(CCFLAGS) -o $(OBJECT) $(CUFILES)

.PHONY: clean
clean:
	rm $(OBJECT)
