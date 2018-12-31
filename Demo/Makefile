include ../Config/SparseFrame.mk

C = $(CC) $(CFLAGS)
I = -I../Include
LIBS = $(LDLIBS) \
	   -L../Lib \
	   -lSparseFrame

all: library demo

library:
	( cd ../Lib ; $(MAKE) )

demo: demo.c
	$(C) $(I) $(LDFLAGS) -Wl,-rpath=$(realpath $(CURDIR)/../Lib) $(LIBS) $< -o $@

clean:
	rm -fr demo
