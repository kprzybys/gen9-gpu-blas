# Reproducible build of the six hand-assembled GEMM binaries (no GPU needed):
#   make -C asm          rebuild all binaries into ../bench from the frozen templates (*.cl) and tracked maps (*_map.txt)
#   make -C asm check    rebuild into a temporary directory (from scratch) and cmp with the binaries in ../bench
#   make -C asm maps     regenerate the accumulator maps on the GPU (needs ../bench/hgemm and ../bench/sgemm built)
# Per binary: template (OpenCL C) --ocloc--> disassembly dir <name>.d --generator `loop` mode + map--> .bin
# Library binaries: gen_h, gen_hd, gen_s (embedded by lib/Makefile); older experiments: hgemm_asm, hgemm_kc_asm, sgemm_asm.
# gen_h and gen_hd share the template gen_h.cl (they patch different kernels: hgemm_v and hgemm_kc_v).
BIN_DIR ?= ../bench
BINS = gen_h gen_hd gen_s hgemm_asm hgemm_kc_asm sgemm_asm
MAPS = h hd s hgemm hgemm_kc sgemm
DIRS = gen_h.d gen_hd.d gen_s.d hgemm.d hgemm_kc.d sgemm.d

.PHONY: all check maps clean
all: $(addprefix $(BIN_DIR)/,$(addsuffix .bin,$(BINS)))

# disassembly dirs (the generators read asm/<dir>/.text.<kernel>.asm)
gen_h.d: gen_h.cl disassemble.sh
	./disassemble.sh gen_h.cl $@ >/dev/null
gen_hd.d: gen_h.cl disassemble.sh
	./disassemble.sh gen_h.cl $@ >/dev/null
gen_s.d: gen_s.cl disassemble.sh
	./disassemble.sh gen_s.cl $@ >/dev/null
hgemm.d: hgemm.cl disassemble.sh
	./disassemble.sh hgemm.cl $@ >/dev/null
hgemm_kc.d: hgemm_kc.cl disassemble.sh
	./disassemble.sh hgemm_kc.cl $@ >/dev/null
sgemm.d: sgemm.cl disassemble.sh
	./disassemble.sh sgemm.cl $@ >/dev/null

$(BIN_DIR)/gen_h.bin: gen_h.d gen_h_map.txt gemm_gen.py
	python3 -I gemm_gen.py h loop $@
$(BIN_DIR)/gen_hd.bin: gen_hd.d gen_hd_map.txt gemm_gen.py
	python3 -I gemm_gen.py hd loop $@
$(BIN_DIR)/gen_s.bin: gen_s.d gen_s_map.txt gemm_gen.py
	python3 -I gemm_gen.py s loop $@
$(BIN_DIR)/hgemm_asm.bin: hgemm.d hgemm_map.txt hgemm_asm.py
	python3 -I hgemm_asm.py loop $@
$(BIN_DIR)/hgemm_kc_asm.bin: hgemm_kc.d hgemm_kc_map.txt hgemm_kc_asm.py
	python3 -I hgemm_kc_asm.py loop $@
$(BIN_DIR)/sgemm_asm.bin: sgemm.d sgemm_map.txt sgemm_asm.py
	python3 -I sgemm_asm.py loop $@

check:
	@tmp=$$(mktemp -d); \
	$(MAKE) --no-print-directory -B BIN_DIR=$$tmp all >/dev/null || exit 1; \
	fail=; \
	for b in $(BINS); do \
	  if cmp -s $$tmp/$$b.bin ../bench/$$b.bin; then echo "OK        $$b.bin"; else echo "DIFFERENT $$b.bin"; fail=1; fi; \
	done; \
	rm -r $$tmp; \
	test -z "$$fail"

maps: $(DIRS)
	@for k in $(MAPS); do echo "== map $$k"; ./maps.sh $$k || exit 1; done

clean:
	rm -r -f $(DIRS)
