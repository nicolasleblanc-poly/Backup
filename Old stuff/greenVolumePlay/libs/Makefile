#_*_MakeFile_*_
CC = gcc
_CC2 = nvcc
CFLAGS = -Wall -g -c -fPIC
_CLIBFLAGS = -Wall -g -shared -o 
_CC2FLAGS = -arch=sm_86 
# tiger gpu is sm_60, 
# GeForce RTX 3070 (desktop) sm_86

_MAJORV = 1
_MINORV = 1

_COPTS = -std=c99

IDIR = -I. -I/usr/include
_LIBDIR = -L. -L/usr/lib/x86_64-linux-gnu 
_LIBS_CUDA = -lm -lcufft -lcublas -lcusolver

VICu: gMGCu.o 
	$(_CC2) $(_CC2FLAGS) -shared -Xlinker '-soname,libmgcu.so.$(_MAJORV)' -o libMGCu.so.$(_MAJORV).$(_MINORV) gMGCu.o $(_LIBDIR) $(_LIBS_CUDA)
	ln -sf libMGCu.so.$(_MAJORV).$(_MINORV) libMGCu.so.$(_MAJORV)
	ln -sf libMGCu.so.$(_MAJORV) libMGCu.so 

gMGCu.o:
	$(_CC2) $(_CC2FLAGS) -Xcompiler '-fPIC' -dc gMGCu.cu $(IDIR)

.PHONY: cleanLibs

.PHONY: cleanObjs

.PHONY: cleanProgs

cleanLibs: 
	rm -f ./*.so ./*.so.* ./*.o

cleanProgs:
	rm -f ./*.exe ./*.o

cleanObjs:
	rm -f ./*.o