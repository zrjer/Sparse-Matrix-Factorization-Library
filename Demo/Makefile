include ../Config/SparseFrame.mk

I = -I../Include
RPATH = $(realpath $(CURDIR)/../Lib)
LIBS = $(LDLIBS) \
	   -L../Lib \
	   -lSparseFrame \
	   -Wl,-rpath=$(RPATH)

all: library demo cublas_demo

library:
	( cd ../Lib ; $(MAKE) )

demo: demo.c
	$(CC) $(CFLAGS) $(LDFLAGS) $(I) $< -o $@ $(LIBS)

cublas_demo_kernel.o: cublas_demo_kernel.cu
	$(NVCC) $(NVCCFLAGS) $(I) -c $< -o $@

cublas_demo.o: cublas_demo.c
	$(CC) $(CFLAGS) $(LDFLAGS) $(I) -c $< -o $@

cublas_demo: cublas_demo_kernel.o cublas_demo.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@ $(LIBS)

clean:
	rm -fr *.o *.so demo cublas_demo
