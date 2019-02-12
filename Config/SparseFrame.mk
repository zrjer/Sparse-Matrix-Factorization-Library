MAKE = make

ifneq ($(shell which icc 2> /dev/null),)
	CC = icc -D_GNU_SOURCE
	CXX = $(CC)
	CFOPENMP = -qopenmp
	LDLIBS = -lm -lirc
else
	CC = gcc
	CXX = g++
	CFOPENMP = -fopenmp
endif

ifneq ($(shell which ifort 2> /dev/null),)
	F77 = ifort
else
	F77 = gfortran
endif

NVCC = nvcc

CFPTHREADS = -pthread

CFLAGS = -O0 -fPIC -Wall $(CFOPENMP)
CXXFLAGS = $(CFLAGS)
NVCCFLAGS = -O0 \
			-Xcompiler -fPIC \
			-gencode arch=compute_30,code=sm_30 \
			-gencode arch=compute_35,code=sm_35 \
			-gencode arch=compute_50,code=sm_50 \
			-gencode arch=compute_60,code=sm_60

LBLAS = -lopenblas
LCUDA = -lcudart -lcublas -lcusolver
LDLIBS += -lgfortran -lm -lrt -lamd -lcamd $(LBLAS) $(LCUDA)

SOFLAGS = -shared
LDFLAGS = $(CFOPENMP)
