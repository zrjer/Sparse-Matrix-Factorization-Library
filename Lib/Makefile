include ../Config/SparseFrame.mk

LIBRARY = libSparseFrame

C = $(CC) $(CFLAGS)
I = -I../Include
LIBS = $(LDLIBS)

OBJS = SparseFrame.o

all: library

library: $(OBJS)
	$(CC) $(SOFLAGS) $(LDFLAGS) $^ -o $(LIBRARY).so $(LIBS)

clean:
	rm -fr *.o *.so

SparseFrame.o: ../Source/SparseFrame.c
	$(C) $(I) -c $< -o $@
