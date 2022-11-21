# objects = main.o hip_euler_1d.o datawrite.o

# all: $(objects)
# 	hipcc $(objects) -lhdf5 -lhdf5_cpp  -o app

# %.o: %.cpp
# 	hipcc -x cu -lhdf5 -lhdf5_cpp -I. -dc  $< -o $@

# clean:
# 	rm -f *.o app

HIP_PATH ?= $(wildcard /opt/rocm/hip)
HIP_PLATFORM = $(shell $(HIP_PATH)/bin/hipconfig --platform)
HIP_INCLUDE  = -I$(HIP_PATH)/include
BUILD_DIR ?= build 

HIPCC=$(HIP_PATH)/bin/hipcc

ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

CXX=${HIPCC}
CXXFLAGS = -std=c++17 -I. -I/public/apps/hdf5/1.12.0/gcc.7.3.1/include -O3 -march=native
EXTRAFLAGS := 
CXXFLAGS:=$(CXXFLAGS) $(EXTRAFLAGS)
LDFLAGS = -lhdf5 -lhdf5_cpp
LD_LIBRARY_PATH := /public/apps/hdf5/1.12.0/gcc.7.3.1/lib
LIBS=-L$(LD_LIBRARY_PATH) $(LDFLAGS)

ifeq (, ${LD_LIBRARY_PATH})
	LIBS=$(LDFLAGS)
endif


SOURCES=$(wildcard *.cpp)
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=./gpuEuler


$(EXECUTABLE): $(OBJECTS)
		$(CXX) $(CXXFLAGS) $(LIBS) $(OBJECTS) -o $@

# %.o: %.cpp
# 		$(CXX) $(CXXFLAGS) $(LDFLAGS) -dc $< -o $@	
clean:
	rm -f $(EXECUTABLE) 
	rm -f $(OBJECTS) 