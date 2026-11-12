# Library comparison: libgen9blas (GPU), CLBlast (GPU), OpenBLAS (CPU). Run from the repo root (gen9blas_init("bench")).
# Needs ../../lib built first (make -C lib): links the shared libgen9blas.so.
l3: l3.c ../../lib/gen9blas.h ../../lib/libgen9blas.so
	gcc -O2 -Wall -o $@ l3.c -L../../lib -lgen9blas -Wl,-rpath,$(abspath ../../lib) -lclblast -lOpenCL -l:libopenblas.so.0 -lm
