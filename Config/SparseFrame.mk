MAKE = make

ifneq ($(shell which icc 2> /dev/null),)
	CC = icc -D_GNU_SOURCE
	CXX = $(CC)
	CFOPENMP = -qopenmp
	LDLIBS = -lirc
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

CFLAGS = -O3 -fexceptions -fPIC -Wall -Werror $(CFOPENMP)
CXXFLAGS = $(CFLAGS)
NVCCFLAGS = -Xcompiler -O3,-fexceptions,-fPIC,-Wall \
			-gencode arch=compute_35,code=sm_35 \
			-gencode arch=compute_37,code=sm_37 \
			-gencode arch=compute_50,code=sm_50 \
			-gencode arch=compute_52,code=sm_52 \
			-gencode arch=compute_53,code=sm_53 \
			-gencode arch=compute_60,code=sm_60 \
			-gencode arch=compute_61,code=sm_61 \
			-gencode arch=compute_70,code=sm_70 \
			-gencode arch=compute_72,code=sm_72

LBLAS = -lopenblas
LCUDA = -lcublas -lcusolver -lcudart -lcudadevrt
LDLIBS += -lgfortran -lm -lrt -lamd -lcamd -lcolamd -lccolamd -lmetis $(LBLAS) $(LCUDA)

SOFLAGS = -shared -Wl,--no-undefined
LDFLAGS = $(CFOPENMP)
