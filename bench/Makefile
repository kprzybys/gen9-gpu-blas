# Measurement programs. OpenCL programs need an OpenCL runtime (NEO); overhead links libgen9blas.so (make -C lib first);
# gemm_clblast needs CLBlast and OpenBLAS. Comparisons with other libraries: bench/compare/Makefile.
CFLAGS = -O2 -Wall
CL_PROGRAMS = hgemm sgemm clbench packing l12
all: $(CL_PROGRAMS) overhead
optional: gemm_clblast

# attn_kcl: the attention asm kernels through kcl (no NEO) on userptr memory (docs/attention-gpu.md section 10). Needs
# the whl *_tok binaries (make -C ../asm attn) and ocloc for the kernel table; on a machine without ocloc copy
# attn_kernels.c from one that has it.
ATTN_KCL_BINS = attn_qwen_tok.bin attn_llama_tok.bin attn_qwen_tokc.bin attn_llama_tokc.bin attn_whisper_tokc.bin attn_qwen7b_tokc.bin attn_qwen3_tokc.bin
attn_kernels.c: ../lib/gen_kernels.py $(addprefix ../lib/kernels/,$(ATTN_KCL_BINS))
	python3 ../lib/gen_kernels.py ../lib/kernels .attn_kernels $(ATTN_KCL_BINS) > $@
attn_kcl: attn_kcl.c attn_kernels.c ../lib/libkcl.a ../lib/kcl.h ../lib/kcl_desc.h
	gcc $(CFLAGS) -I../lib -I../lib/third_party/mesa -o $@ attn_kcl.c attn_kernels.c ../lib/libkcl.a -lm -lpthread

# kcl_const: regression test for program constants read through a binding-table slot (lib/kcl.c const_bti)
kcl_const.bin: kcl_const.cl
	ocloc compile -file kcl_const.cl -device whl -options "-cl-std=CL2.0" -output kcl_const -output_no_suffix -q
kcl_const_kernels.c: kcl_const.bin ../lib/gen_kernels.py
	python3 ../lib/gen_kernels.py . .kcl_const kcl_const.bin > $@
kcl_const: kcl_const.c kcl_const_kernels.c ../lib/libkcl.a ../lib/kcl.h ../lib/kcl_desc.h
	gcc $(CFLAGS) -I../lib -I../lib/third_party/mesa -o $@ kcl_const.c kcl_const_kernels.c ../lib/libkcl.a -lm -lpthread

$(CL_PROGRAMS): %: %.c
	gcc $(CFLAGS) -o $@ $< -lOpenCL -lm

overhead: overhead.c ../lib/gen9blas.h ../lib/libgen9blas.so
	gcc $(CFLAGS) -o $@ overhead.c -L../lib -lgen9blas -Wl,-rpath,$(abspath ../lib) -lm

gemm_clblast: gemm_clblast.c
	gcc $(CFLAGS) -o $@ gemm_clblast.c -lclblast -lOpenCL -l:libopenblas.so.0 -lm

clean:
	rm -f $(CL_PROGRAMS) overhead gemm_clblast attn_kcl attn_kernels.c kcl_const kcl_const.bin kcl_const_kernels.c
.PHONY: all optional clean

../lib/libkcl.a: ../lib/kcl.c ../lib/kcl.h ../lib/kcl_desc.h
	$(MAKE) -C ../lib libkcl.a
