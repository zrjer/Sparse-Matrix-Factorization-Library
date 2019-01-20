MAKE = make

CC = gcc
CXX = g++
NVCC = nvcc

CFOPENMP = -fopenmp
CFPTHREADS = -pthread

CFLAGS = -O0 -fPIC -Wall $(CFOPENMP)
CXXFLAGS = -O0 -fPIC -Wall $(CFOPENMP)
NVCCFLAGS = -O0 \
    -Xcompiler -fPIC \
	-gencode arch=compute_30,code=sm_30 \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50 \
	-gencode arch=compute_60,code=sm_60

LBLAS = -lopenblas
LLAPACK = -llapack
LCUDA = -lcudart -lcublas -lcusolver
LDLIBS = -lgfortran -lm -lrt # $(LBLAS) $(LLAPACK) $(LCUDA)

SOFLAGS = -shared
LDFLAGS = $(CFOPENMP)
