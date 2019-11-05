include ../../Config/SparseFrame.mk

I = -I../Include
RPATH = $(realpath $(CURDIR)/../Lib)
LIBS = $(LDLIBS) \
	   -L../Lib \
	   -lSparseFrame \
	   -Wl,-rpath=$(RPATH)

all: library demo

library:
	( cd ../Lib ; $(MAKE) )

demo: demo.c
	$(CC) $(CFLAGS) $(LDFLAGS) $(I) $< -o $@ $(LIBS)

clean:
	rm -fr *.o *.so demo
