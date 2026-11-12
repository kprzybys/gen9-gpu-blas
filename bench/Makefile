# Measurement programs. OpenCL programs need an OpenCL runtime (NEO); overhead links libgen9blas.so (make -C lib first);
# gemm_clblast needs CLBlast and OpenBLAS. Comparisons with other libraries: bench/compare/Makefile.
CFLAGS = -O2 -Wall
CL_PROGRAMS = hgemm sgemm clbench packing l12
all: $(CL_PROGRAMS) overhead
optional: gemm_clblast

$(CL_PROGRAMS): %: %.c
	gcc $(CFLAGS) -o $@ $< -lOpenCL -lm

overhead: overhead.c ../lib/gen9blas.h ../lib/libgen9blas.so
	gcc $(CFLAGS) -o $@ overhead.c -L../lib -lgen9blas -Wl,-rpath,$(abspath ../lib) -lm

gemm_clblast: gemm_clblast.c
	gcc $(CFLAGS) -o $@ gemm_clblast.c -lclblast -lOpenCL -l:libopenblas.so.0 -lm

clean:
	rm -f $(CL_PROGRAMS) overhead gemm_clblast
.PHONY: all optional clean
