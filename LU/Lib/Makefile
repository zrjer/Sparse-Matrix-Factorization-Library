include ../../Config/SparseFrame.mk

LIBRARY = libSparseFrame

I = -I../Include
LIBS = $(LDLIBS)

OBJS = cuda_kernel.o SparseFrame.o

all: library

library: $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(SOFLAGS) $^ -o $(LIBRARY).so $(LIBS)

clean:
	rm -fr *.o *.so

cuda_kernel.o: ../Source/cuda_kernel.cu
	$(NVCC) $(NVCCFLAGS) $(I) -c $< -o $@

SparseFrame.o: ../Source/SparseFrame.c
	$(CC) $(CFLAGS) $(LDFLAGS) $(I) -c $< -o $@
