#!/usr/bin/make
#

BIN := ../lib/libcputhreaded.so

INCLUDE += -I$(JAVA_HOME)/include

CPPFILES += cputhreaded.cpp

.PHONY: all gen clean

all: $(BIN)

$(BIN):
	gcc $(INCLUDE) -O3 -shared -fPIC $(CPPFILES) -o $(BIN)
clean:
	rm -f *.o $(BIN)
