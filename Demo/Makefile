include ../Config/SparseFrame.mk

C = $(CC) $(CFLAGS)
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
	$(C) $(I) $(LDFLAGS) $(LIBS) $< -o $@

clean:
	rm -fr demo
