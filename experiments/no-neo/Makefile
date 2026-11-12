# Eksperymenty bez NEO (opis: ../BEZNEO.md). Nagłówek komend Gen9 generowany z Mesa genxml.
GENXML = ../../third_party/mesa-25.2.8/genxml
SYMBOLE = PIPELINE_SELECT,STATE_BASE_ADDRESS,MEDIA_VFE_STATE,MEDIA_INTERFACE_DESCRIPTOR_LOAD,INTERFACE_DESCRIPTOR_DATA,GPGPU_WALKER,MEDIA_STATE_FLUSH,PIPE_CONTROL,MI_BATCH_BUFFER_END,MI_LOAD_REGISTER_IMM,MI_STORE_REGISTER_MEM,MI_NOOP,MEMORY_OBJECT_CONTROL_STATE,L3CNTLREG,RENDER_SURFACE_STATE,BINDING_TABLE_STATE
all: pierwszy neo_por sonda zapis.isa lancuch neo_lancuch lancuch_bti gemm_h userptr

../../lib/third_party/mesa/gen9_pack.h: $(GENXML)/gen90.xml
	python3 $(GENXML)/gen_pack_header.py --include-symbols "$(SYMBOLE)" $< > $@

zapis.isa: store.cl extract.py
	ocloc compile -file store.cl -device whl -options "-cl-intel-greater-than-4GB-buffer-required" -output_no_suffix -output zapis >/dev/null
	python3 extract.py zapis.bin zapis $@

pierwszy: first.c ../../lib/third_party/mesa/gen9_pack.h
	gcc -O2 -Wall -I../../lib/third_party/mesa -o $@ first.c
neo_por: neo_compare.c
	gcc -O2 -Wall -o $@ $< -lOpenCL
sonda: probe.c
	gcc -O2 -Wall -o $@ $<

kernels.h: store.cl kernels.py
	ocloc compile -file store.cl -device whl -options "-cl-intel-greater-than-4GB-buffer-required" -output_no_suffix -output zapis >/dev/null
	rm -rf dis && ocloc disasm -file zapis.bin -device whl -dump dis >/dev/null 2>&1
	python3 kernels.py zapis.bin dis/.ze_info > $@

lancuch: chain.c kernels.h gen9i.h ../../lib/third_party/mesa/gen9_pack.h
	gcc -O2 -Wall -I../../lib/third_party/mesa -I. -o $@ chain.c
neo_lancuch: neo_chain.c
	gcc -O2 -Wall -o $@ $< -lOpenCL

# Te same jądra z binding table (domyślny tryb IGC: wskaźniki stateful, 32-bitowe offsety)
kernels_bti.h: store.cl kernels.py
	ocloc compile -file store.cl -device whl -output_no_suffix -output zapis_bti >/dev/null
	rm -rf dis_bti && ocloc disasm -file zapis_bti.bin -device whl -dump dis_bti >/dev/null 2>&1
	python3 kernels.py zapis_bti.bin dis_bti/.ze_info > $@

lancuch_bti: chain.c kernels_bti.h gen9i.h ../../lib/third_party/mesa/gen9_pack.h
	gcc -O2 -Wall -I../../lib/third_party/mesa -I. -DBTI -o $@ chain.c

# HGEMM: jądra pakowania z bench/hgemm.cl (IGC, binding table) i hgemm_v z bench/gen_h.bin (nasz asembler)
kernels_pack.h: ../../bench/hgemm.cl kernels.py
	ocloc compile -file ../../bench/hgemm.cl -device whl -output_no_suffix -output hgemm_cl >/dev/null
	rm -rf dis_h && ocloc disasm -file hgemm_cl.bin -device whl -dump dis_h >/dev/null 2>&1
	python3 kernels.py hgemm_cl.bin dis_h/.ze_info pk_a2_q,pk_bq_w2 > $@
kernels_gemm.h: ../../bench/gen_h.bin kernels.py
	rm -rf dis_gen && ocloc disasm -file ../../bench/gen_h.bin -device whl -dump dis_gen >/dev/null 2>&1
	python3 kernels.py ../../bench/gen_h.bin dis_gen/.ze_info hgemm_v > $@
gemm_h: gemm_h.c kernels_pack.h kernels_gemm.h gen9i.h ../../lib/third_party/mesa/gen9_pack.h
	gcc -O2 -Wall -I../../lib/third_party/mesa -I. -o $@ gemm_h.c -lm

# userptr: koszt przypięcia stron wobec kopii, odczyt GPU z pamięci procesu (BEZNEO.md 6g); FAZY=1: rozbicie na fazy
userptr: userptr.c kernels.h gen9i.h ../../lib/third_party/mesa/gen9_pack.h
	gcc -O2 -Wall -I../../lib/third_party/mesa -I. -o $@ userptr.c
