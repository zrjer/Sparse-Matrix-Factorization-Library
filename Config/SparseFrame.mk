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

CFLAGS = -O3 -fPIC -Wall $(CFOPENMP)
CXXFLAGS = $(CFLAGS)
NVCCFLAGS = -O3 \
			-Xcompiler -fPIC \
			-gencode arch=compute_30,code=sm_30 \
			-gencode arch=compute_35,code=sm_35 \
			-gencode arch=compute_50,code=sm_50 \
			-gencode arch=compute_60,code=sm_60

LBLAS = -lopenblas
LCUDA = -lcublas -lcublas_device -lcusolver -lcudart -lcudadevrt
LDLIBS += -lgfortran -lm -lrt -lamd -lcamd -lcolamd -lccolamd -lmetis $(LBLAS) $(LCUDA)

SOFLAGS = -shared -Wl,--no-undefined
LDFLAGS = $(CFOPENMP)
