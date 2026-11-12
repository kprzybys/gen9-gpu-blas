# libgen9blas: BLAS (poziomy 1–3) na UHD 620. Domyślnie bez NEO: kcl (podzbiór OpenCL na i915) + jądra wbudowane
# (kernels.c generowane przez gen_kernels.py z ../bench/*.cl i binarek asm ../bench/gen_*.bin; wymaga ocloc przy budowie).
# libgen9blas_neo.so / libgen9blas_blas_neo.so: dawna ścieżka przez NEO (OpenCL), tylko do porównań.
GEN = third_party/mesa
CFL = -O2 -Wall -shared -fPIC
ZRODLA_CL = ../bench/hgemm.cl ../bench/sgemm.cl ../bench/dgemm.cl ../bench/level12.cl ../bench/gen_h.bin ../bench/gen_hd.bin ../bench/gen_s.bin
all: libgen9blas.so libgen9blas_blas.so libgen9blas_neo.so libgen9blas_blas_neo.so

kernels.c: gen_kernels.py $(ZRODLA_CL)
	python3 gen_kernels.py ../bench .kernels > $@

libgen9blas.so: gen9blas.c gen9blas.h kcl.c kcl.h kcl_desc.h kernels.c $(GEN)/gen9_pack.h
	gcc $(CFL) -I$(GEN) -o $@ gen9blas.c kcl.c kernels.c -lm

libgen9blas_neo.so: gen9blas.c gen9blas.h
	gcc $(CFL) -DGEN9BLAS_NEO -o $@ gen9blas.c -lOpenCL -lm

test: libgen9blas.so
	python3 test_basic.py

clean:
	rm -f libgen9blas.so libgen9blas_neo.so libgen9blas_blas.so libgen9blas_blas_neo.so kernels.c

# Pełny BLAS/CBLAS: funkcje S/D na GPU, reszta przez skoki do OpenBLAS (gen9blas_blas_forward.c generowane z listy symboli).
libgen9blas_blas.so: gen9blas.c gen9blas.h kcl.c kcl.h kcl_desc.h kernels.c gen9blas_blas.c gen9blas_blas_d.inc gen9blas_blas_forward.c
	gcc $(CFL) -I$(GEN) -o $@ gen9blas.c kcl.c kernels.c gen9blas_blas.c gen9blas_blas_forward.c -lm -ldl -lpthread

libgen9blas_blas_neo.so: gen9blas.c gen9blas.h gen9blas_blas.c gen9blas_blas_d.inc gen9blas_blas_forward.c
	gcc $(CFL) -DGEN9BLAS_NEO -o $@ gen9blas.c gen9blas_blas.c gen9blas_blas_forward.c -lOpenCL -lm -ldl -lpthread

gen9blas_blas_forward.c: gen_forward.py blas_symbols.txt gen9blas_blas.c gen9blas_blas_d.inc
	python3 gen_forward.py blas_symbols.txt gen9blas_blas.c gen9blas_blas_d.inc > $@

# Funkcje D warstwy BLAS z kodu funkcji S (float -> double, nazwy s -> d).
gen9blas_blas_d.inc: gen_blas_d.py gen9blas_blas.c
	python3 gen_blas_d.py gen9blas_blas.c > $@

# Kernele FP64 poziomu 3: rdzeń (pakowanie, DGEMM) + kernele pomocnicze z sgemm.cl po podstawieniu typów.
../bench/dgemm.cl: gen_dgemm.py ../bench/sgemm.cl ../bench/dgemm_core.cl
	python3 gen_dgemm.py ../bench/sgemm.cl ../bench/dgemm_core.cl > $@
