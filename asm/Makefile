# Reproducible build of the six hand-assembled GEMM binaries (no GPU needed):
#   make -C asm          rebuild all binaries into ../bench from the frozen templates (*.cl) and tracked maps (*_map.txt)
#   make -C asm check    gemm-check + attn-check (below)
#   make -C asm gemm-check  rebuild the GEMM binaries into a temporary directory (from scratch), cmp with ../bench
#   make -C asm maps     regenerate the accumulator maps on the GPU (needs ../bench/hgemm and ../bench/sgemm: make -C ../bench)
#   make -C asm attn     hand-scheduled attention kernels (one binary per shape; DEVICE=skl for HD 530), not in `all`
#   make -C asm attn-check  rebuild the attention binaries for skl and whl in a temporary directory and compare their
#                        SHA-256 with attn_sha256.txt (the binaries are not tracked; ocloc pinned, see attn_gen.py)
#   make -C asm attn-sha    rewrite attn_sha256.txt from a fresh build (only after a deliberate generator change)
# Per binary: template (OpenCL C) --ocloc--> disassembly dir <name>.d --generator `loop` mode + map--> .bin
# Library binaries: gen_h, gen_hd, gen_s (embedded by lib/Makefile); older experiments: hgemm_asm, hgemm_kc_asm, sgemm_asm.
# gen_h and gen_hd share the template gen_h.cl (they patch different kernels: hgemm_v and hgemm_kc_v).
# Guide to writing a new hand-assembled kernel (rules, workflow, checklist): ../docs/isa-notes.md
BIN_DIR ?= ../bench
BINS = gen_h gen_hd gen_s hgemm_asm hgemm_kc_asm sgemm_asm
MAPS = h hd s hgemm hgemm_kc sgemm
DIRS = gen_h.d gen_hd.d gen_s.d hgemm.d hgemm_kc.d sgemm.d

.PHONY: all check gemm-check attn-check attn-sha maps clean attn FORCE
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

# Attention, stage 2b (docs/attention-gpu.md section 9): template attn.cl -> attn_gen.py sched<NS> -> .bin for
# bench/attn.c (CL=<bin>). No map step: attn_gen.py reads the register maps from IGC's own loop.
# attn_gen.py refuses to run with another ocloc than the pinned one (OCLOC_VERSION there, 24.35.30872.36).
DEVICE ?= whl
ATTN_OPTS = -cl-fast-relaxed-math -cl-mad-enable
ATTN_BINS = attn_qwen attn_llama attn_qwen_tok attn_llama_tok attn_qwen_tokc attn_llama_tokc attn_whisper_tokc attn_qwen7b_tokc attn_qwen3_tokc attn_nemo_tokc
# *_tok: Q in llama.cpp's layout Q[t][h][d] (-DQTOK, docs/attention-gpu.md section 10), for bench/attn_kcl and
# bench/attn with QTOK=1; the loop is generated the same way (only IGC's prologue differs).
attn: $(addprefix $(BIN_DIR)/,$(addsuffix .bin,$(ATTN_BINS)))
# A binary for one device is CL_INVALID_BINARY on the other: the stamp holds the DEVICE of the last build in
# BIN_DIR and is rewritten (which forces a rebuild of both binaries) whenever DEVICE differs.
$(BIN_DIR)/.attn-device: FORCE
	@mkdir -p $(BIN_DIR); test "$$(cat $@ 2>/dev/null)" = "$(DEVICE)" || echo "$(DEVICE)" > $@
$(BIN_DIR)/attn_qwen.bin: attn.cl attn_gen.py isa.py $(BIN_DIR)/.attn-device
	DEVICE=$(DEVICE) python3 -I attn_gen.py sched4 attn.cl "$(ATTN_OPTS) -DD=128 -DG=6 -DHKV=2 -DRQ=2 -DNK=2" $@
$(BIN_DIR)/attn_llama.bin: attn.cl attn_gen.py isa.py $(BIN_DIR)/.attn-device
	DEVICE=$(DEVICE) python3 -I attn_gen.py sched2 attn.cl "$(ATTN_OPTS) -DD=64 -DG=4 -DHKV=8 -DRQ=4 -DNK=2" $@
$(BIN_DIR)/attn_qwen_tok.bin: attn.cl attn_gen.py isa.py $(BIN_DIR)/.attn-device
	DEVICE=$(DEVICE) python3 -I attn_gen.py sched4 attn.cl "$(ATTN_OPTS) -DD=128 -DG=6 -DHKV=2 -DRQ=2 -DNK=2 -DQTOK" $@
$(BIN_DIR)/attn_llama_tok.bin: attn.cl attn_gen.py isa.py $(BIN_DIR)/.attn-device
	DEVICE=$(DEVICE) python3 -I attn_gen.py sched2 attn.cl "$(ATTN_OPTS) -DD=64 -DG=4 -DHKV=8 -DRQ=4 -DNK=2 -DQTOK" $@
# *_tokc: *_tok plus index-based causal masking (-DCAUSAL=1: per-subgroup key count, the partial block masked in the
# generated loop; docs/attention-gpu.md section 11), the binaries the library (lib/gen9blas_attn.c) runs.
$(BIN_DIR)/attn_qwen_tokc.bin: attn.cl attn_gen.py isa.py $(BIN_DIR)/.attn-device
	DEVICE=$(DEVICE) python3 -I attn_gen.py sched4 attn.cl "$(ATTN_OPTS) -DD=128 -DG=6 -DHKV=2 -DRQ=2 -DNK=2 -DQTOK -DCAUSAL=1" $@
$(BIN_DIR)/attn_llama_tokc.bin: attn.cl attn_gen.py isa.py $(BIN_DIR)/.attn-device
	DEVICE=$(DEVICE) python3 -I attn_gen.py sched2 attn.cl "$(ATTN_OPTS) -DD=64 -DG=4 -DHKV=8 -DRQ=4 -DNK=2 -DQTOK -DCAUSAL=1" $@
# whisper large-v3(-turbo) encoder: 20 heads of 64, no GQA (G 1), no mask; CAUSAL=1 for the partial last key block
# (n_k = 1500, run with vis0 = 1500, dvis = 0 on 1504 rows).
$(BIN_DIR)/attn_whisper_tokc.bin: attn.cl attn_gen.py isa.py $(BIN_DIR)/.attn-device
	DEVICE=$(DEVICE) python3 -I attn_gen.py sched2 attn.cl "$(ATTN_OPTS) -DD=64 -DG=1 -DHKV=20 -DRQ=4 -DNK=2 -DQTOK -DCAUSAL=1" $@
# NeMo-Speech.cpp Nemotron 3 Diarization (Sortformer v3 RoPE encoder, d_model 512, 8 heads): whisper's kernel with 8 heads
$(BIN_DIR)/attn_nemo_tokc.bin: attn.cl attn_gen.py isa.py $(BIN_DIR)/.attn-device
	DEVICE=$(DEVICE) python3 -I attn_gen.py sched2 attn.cl "$(ATTN_OPTS) -DD=64 -DG=1 -DHKV=8 -DRQ=4 -DNK=2 -DQTOK -DCAUSAL=1" $@
# Qwen2.5-7B: 28 heads of 128, H_kv 4, G 7 (odd: the two rows of a subgroup may be two tokens): -DROWVIS=1, per-row
# visible key counts and clamped rows (docs/attention-gpu.md section 13).
$(BIN_DIR)/attn_qwen7b_tokc.bin: attn.cl attn_gen.py isa.py $(BIN_DIR)/.attn-device
	DEVICE=$(DEVICE) python3 -I attn_gen.py sched4 attn.cl "$(ATTN_OPTS) -DD=128 -DG=7 -DHKV=4 -DRQ=2 -DNK=2 -DQTOK -DCAUSAL=1 -DROWVIS=1" $@
# Qwen3 (Qwen3-Embedding-0.6B, Qwen3-0.6B/1.7B): 16 heads of 128, H_kv 8, G 2: both rows of a subgroup are one token
# (G % RQ == 0), so no ROWVIS; sched4 as for Qwen2.5-1.5B (docs/attention-gpu.md section 14).
$(BIN_DIR)/attn_qwen3_tokc.bin: attn.cl attn_gen.py isa.py $(BIN_DIR)/.attn-device
	DEVICE=$(DEVICE) python3 -I attn_gen.py sched4 attn.cl "$(ATTN_OPTS) -DD=128 -DG=2 -DHKV=8 -DRQ=2 -DNK=2 -DQTOK -DCAUSAL=1" $@
FORCE:

attn-check:
	@tmp=$$(mktemp -d); fail=; \
	for d in skl whl; do \
	  $(MAKE) --no-print-directory -B BIN_DIR=$$tmp/$$d DEVICE=$$d attn >$$tmp/log 2>&1 || { cat $$tmp/log; rm -r $$tmp; exit 1; }; \
	done; \
	for d in skl whl; do for b in $(ATTN_BINS); do \
	  want=$$(awk -v f="$$d/$$b.bin" '$$2 == f { print $$1 }' attn_sha256.txt); \
	  got=$$(sha256sum $$tmp/$$d/$$b.bin | cut -d' ' -f1); \
	  if [ -n "$$want" ] && [ "$$want" = "$$got" ]; then echo "OK        $$d/$$b.bin"; \
	  else echo "DIFFERENT $$d/$$b.bin (expected $${want:-none}, got $$got)"; fail=1; fi; \
	done; done; \
	rm -r $$tmp; \
	test -z "$$fail"

attn-sha:
	@tmp=$$(mktemp -d); \
	for d in skl whl; do \
	  $(MAKE) --no-print-directory -B BIN_DIR=$$tmp/$$d DEVICE=$$d attn >/dev/null || { rm -r $$tmp; exit 1; }; \
	done; \
	(cd $$tmp && sha256sum skl/*.bin whl/*.bin) > attn_sha256.txt; \
	rm -r $$tmp; \
	cat attn_sha256.txt

check: gemm-check attn-check

gemm-check:
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
