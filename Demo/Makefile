include ../Config/SparseFrame.mk

C = $(CC) $(CFLAGS)

all: library demo

library:
	( cd ../Lib ; $(MAKE) )

demo: demo.c
	$(C) -o $@ $^
