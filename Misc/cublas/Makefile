include ../../Config/SparseFrame.mk

RPATH = $(realpath $(CURDIR)/../Lib)
LIBS = $(LDLIBS) \
	   -Wl,-rpath=$(RPATH)

all: cublas_demo

cublas_demo_kernel.o: cublas_demo_kernel.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

cublas_demo.o: cublas_demo.c
	$(CC) $(CFLAGS) $(LDFLAGS) -c $< -o $@

cublas_demo: cublas_demo_kernel.o cublas_demo.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@ $(LIBS)

clean:
	rm -fr *.o *.so cublas_demo
