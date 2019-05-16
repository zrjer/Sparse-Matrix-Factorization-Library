include ../Config/SparseFrame.mk

LIBRARY = libSparseFrame

I = -I../Include
LIBS = $(LDLIBS)

OBJS = cuda_kernel.o cuda_kernel_dlink.o SparseFrame.o

all: library

library: $(OBJS)
	$(CC) $(SOFLAGS) $(LDFLAGS) $^ -o $(LIBRARY).so $(LIBS)

clean:
	rm -fr *.o *.so

cuda_kernel.o: ../Source/cuda_kernel.cu
	$(NVCC) $(NVCCFLAGS) $(I) -rdc=true -c $< -o $@

cuda_kernel_dlink.o: cuda_kernel.o
	$(NVCC) $(NVCCFLAGS) -dlink $< -o $@ $(LIBS)

SparseFrame.o: ../Source/SparseFrame.c
	$(CC) $(CFLAGS) $(I) -c $< -o $@
