# Library comparison: libgen9blas (GPU), CLBlast (GPU), OpenBLAS (CPU). Run from the repo root (gen9blas_init("bench")).
# Needs ../../lib built first (make -C lib): links the shared libgen9blas.so.
l3: l3.c ../../lib/gen9blas.h
	gcc -O2 -Wall -I../../lib -o $@ l3.c -L../../build/lib -lgen9blas -Wl,-rpath,$(abspath ../../build/lib) -lclblast -lOpenCL -l:libopenblas.so.0 -lm
