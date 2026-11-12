# Porównanie bibliotek: libgen9blas (GPU), CLBlast (GPU), OpenBLAS (CPU). Uruchamiać z katalogu repo (gen9blas_init("bench")).
l3: l3.c ../../lib/gen9blas.c ../../lib/gen9blas.h
	gcc -O2 -Wall -o $@ l3.c ../../lib/gen9blas.c -lclblast -lOpenCL -l:libopenblas.so.0 -lm
