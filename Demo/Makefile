include ../Config/SparseFrame.mk

C = $(CC) $(CFLAGS)
I = -I../Include

all: library demo

library:
	( cd ../Lib ; $(MAKE) )

demo: demo.c
	$(C) -o $@ $(I) $^

clean:
	rm -fr demo
