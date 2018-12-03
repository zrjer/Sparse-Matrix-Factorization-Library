MAKE = make
CC = gcc
CXX = g++
NVCC = nvcc
CFLAGS = -O3 -fPIC
CXXFLAGS = -O3 -fPIC
NVCCFLAGS = -O3 \
    -Xcompiler -fPIC \
	-gencode arch=compute_30,code=sm_30 \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50 \
	-gencode arch=compute_60,code=sm_60
CFOPENMP = -fopenmp
CFPTHREADS = -pthread
