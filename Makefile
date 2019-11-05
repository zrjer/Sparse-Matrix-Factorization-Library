include Config/SparseFrame.mk

all:
	( cd Cholesky ; $(MAKE) )
	( cd LU ; $(MAKE) )
	( cd Misc ; $(MAKE) )

clean:
	( cd Cholesky ; $(MAKE) clean )
	( cd LU ; $(MAKE) clean )
	( cd Misc ; $(MAKE) clean )
