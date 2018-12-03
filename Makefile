include Config/SparseFrame.mk

all:
	( cd Lib ; $(MAKE) )
	( cd Demo ; $(MAKE) )

library:
	( cd Lib ; $(MAKE) )
