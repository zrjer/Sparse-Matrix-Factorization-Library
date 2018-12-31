include ../Config/SparseFrame.mk

LIBRARY = libSparseFrame

C = $(CC) $(CFLAGS)
I = -I../Include
LIBS = $(LDLIBS)

OBJS = SparseFrame.o

all: library

library: $(OBJS)
	$(CC) $(SOFLAGS) $(LDFLAGS) $(LIBS) $^ -o $(LIBRARY).so

clean:
	rm -fr *.o

SparseFrame.o: ../Source/SparseFrame.c
	$(C) $(I) -c $< -o $@
