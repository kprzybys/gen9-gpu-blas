# Writing Gen9 ISA kernels in this project

Practical guide for the next hand-assembled kernel (GEMM loops in `asm/gemm_gen.py`, attention in `asm/attn_gen.py`).
Evidence tags as in [performance-model.md](performance-model.md): **[SPEC]** PRM KBL (volume, section, page of the
PDFs listed in [references.md](references.md); text copies in `third_party/txt/`), **[MEAS]** results file,
**[SW]** behaviour of NEO 24.35.30872.36 / IGC 1.0.17537.24 (`ocloc` 24.35), **[EST]** Mesa 25.2.8 heuristics,
**[HYP]** unconfirmed. Hardware constants live in [performance-model.md](performance-model.md) and
[hardware.md](hardware.md); the chronology in [history.md](history.md) (Appendix A: reading machine code, Appendix B:
GEMM workflow); symptoms and fixes in [pitfalls.md](pitfalls.md). This file links to them and adds what a kernel author
needs in one place.

## 1. When to drop to ISA (and when not)

| Kernel | Best OpenCL C | Hand loop | What the hand loop fixed | Source |
|---|---|---|---|---|
| HGEMM 16x32, FP16 | 654 GFLOPS | 777 (+19 %) | bank conflicts (256 of 512 `mad`), no `mov` in the loop, double-buffered half k steps | [MEAS] `results/hgemm-compare-2026-11-11.txt`, [history.md](history.md) §8 |
| SGEMM 16x16 | 342-351 | 359-376 (+5-7 %) | 128 of 256 `mad` conflicting | [MEAS] [history.md](history.md) §9 |
| HGEMM, FP32 chunk sums (KC 32) | 441 | 674-689 (+53 %) | 0 conflicts, chunk sums in registers (IGC's 16x32 chunked variants spilled 1280-2880 B) | [MEAS] [history.md](history.md) §9 |
| Attention (Qwen shape, SIMD8) | 231 (template) | 296 (+28 %) | load scheduling: K/V reads 1-3 steps ahead | [MEAS] `results/attn-asm-hd530-2026-11-13.txt` |
| Attention, operands remapped only | 218 | 218 (+0 %), Llama +5 % | conflicts 512 -> 0 per 1024 `mad`, nothing else | same file, variant R |

Drop to ISA when the hot loop is **compute-bound** and IGC demonstrably loses on something it does not let you
control: GRF bank parity of `mad` operands ([SW] IGC does not avoid it: [pitfalls.md](pitfalls.md) §1), loads issued
right before use (attention: IGC issues each block read just before its `mad`s, [MEAS] variant S vs template), or a
register allocation that spills. Measure the ceiling first: the attention `ADV=0` diagnostic (pointers frozen, every
read hits) showed IGC's stream at 261-269 GFLOPS, i.e. the gap was memory-side, which predicted that remapping alone
would not help (`results/attn-kernel-hd530-2026-11-13.txt`).

Do **not** drop to ISA for memory-bound work: packing removed a 20x TLB collapse where assembly added 19 %
([pitfalls.md](pitfalls.md) §5); for level 1-2, packing and skinny GEMM the levers are the message type, 16 B per
lane, page locality and registers ([skinny-gemm.md](skinny-gemm.md)). Costs of ISA: one binary per shape and per
device, a pinned `ocloc`, a generator that parses IGC's code and must be re-checked after every driver update.

## 2. Workflow

```
template.cl --ocloc compile -device whl|skl--> zebin --ocloc disasm--> dir/.text.<kernel>.asm + .ze_info
   --rm .spv + its line in sections.txt--> generator: keep IGC's prologue/epilogue, replace label..back-jump
   --ocloc asm -file X.bin -dump dir--> dir/X.bin --> clCreateProgramWithBinary (clbench, bench/attn, kcl)
```

1. **Template in OpenCL C** with the real signature: IGC writes the prologue (argument loads, ids, pointers,
   accumulator zeroing), `.ze_info` and the epilogue (stores). Make the template do the memory layout you want the
   epilogue to use (attention: `asm/attn.cl` stores O with block writes; +7 % already in C and fewer live registers).
   The template itself must not spill (section 4.5).
2. **Compile and disassemble** (`asm/disassemble.sh`, or `disassemble()` in `asm/attn_gen.py`). Device per binary
   [SW]: `whl` = UHD 620 (laptop), `skl` = HD 530 (desktop); a binary for one is `CL_INVALID_BINARY` on the other,
   and `cfl` is rejected on the UHD 620 (product family in `.note.intelgt.compat`; [pitfalls.md](pitfalls.md) §1).
   `asm/attn_gen.py` reads `DEVICE`; `asm/disassemble.sh`, `gemm_gen.py`, `micro.py` are hard-wired to `whl`.
3. **Delete `.spv`** and its `sections.txt` line, or the driver silently recompiles from SPIR-V and ignores the
   edited ISA [SW].
4. **Generator** (Python, `-I`): split prologue / loop / epilogue around the hot `while` (the one with many `mad`),
   recover the registers IGC chose (pointers, counter, limit, accumulators; GEMM: accumulator map from a GPU run,
   `asm/maps.sh`; attention: symbolic pass `igc_maps()`), allocate free registers, emit the loop. Replace
   **everything** between the loop label and the back-jump: IGC places `mad`s after the counter update ([MEAS] a
   false 360 GFLOPS ceiling, [pitfalls.md](pitfalls.md) §1). Useful modes: `analyze`, `plain` (IGC's code without
   `.spv`), `poison` (one `mad` negated), `map`.
5. **Assemble**: `ocloc asm -file <name> -dump <dir> -device <dev>` (`-file`, not `-out`; output lands in the dump
   dir) [SW]. Single encodings: `iga64 -p=9` from `libigc-tools`.
6. **Pin and hash**: `asm/attn_gen.py` refuses any `ocloc` other than 24.35.30872.36 (`ocloc query
   OCL_DRIVER_VERSION`; `ATTN_OCLOC_ANY=1` overrides). `make -C asm check` = `gemm-check` (rebuild, `cmp` with
   `bench/*.bin`) + `attn-check` (rebuild for `skl` and `whl`, SHA-256 vs `asm/attn_sha256.txt`). The build is
   deterministic: a CPU-only rebuild of `attn_qwen.bin` for `whl` reproduced the tracked hash (2026-11-13).
7. **Load**: `bench/clbench <file>.bin <kernel> ...` and `CL=<file>.bin bench/attn ...` use
   `clCreateProgramWithBinary` + `clBuildProgram`; build options are baked in, so host arguments must match.
   Library binaries (`gen_h`, `gen_hd`, `gen_s`) are embedded by `lib/gen_kernels.py` into `lib/kernels.c` for kcl.

## 3. Reading Gen9 ISA

Field-by-field legend: [history.md](history.md) Appendix A. Real lines from this repository (IGC's `hgemm_v`, the
generated attention loop):

```
(W)     send (16|M0)  r8:w  r62  0xC  0x024D0CFF   // wr:1h+0, rd:4; hdc.dc1; a64 aligned oword block read x8
        mad (8|M0)   r1.0<1>:f   r1.0<2;1>:f   r8.0<2;1>:hf   r97.0<0>:f       // S += K(hf) * q (scalar)
        mad (16|M0)  r124.0<1>:hf r68.0<2;1>:hf r16.0<2;1>:hf r118.0<0>:hf      // HGEMM: acc += B * A[k]
(W)     add (1|M0)   r62.0<1>:q  r62.0<0;1,0>:q  128:w                          // header address += 128 B
        add (8|M0)   r108.0<1>:q r112.0<4;4,1>:q r5.0<0;1,0>:q                  // IGC: per-lane 64-bit address
(W)     cmp (8|M0)   (gt)f1.0  null<1>:f  r60.0<0;1,0>:f  r60.4<0;1,0>:f
(W&~f1.0) jmpi  LATT_SKIP0                                                      // uniform branch
(~f0.0) break (16|M0)  L12024  L12024                                           // IGC: per-channel loop exit
        while (16|M0)  L2952
(W)     sends (16|M0) null:uw r72 r64 0x10A 0x020A0401 // wr:1h+4, rd:0; hdc.dc0; oword block write x8
(W)     send (1|M0)  null  r112  0x27  0x02000010  {EOT}                        // end of thread
```

| Item | Meaning | Evidence |
|---|---|---|
| GRF | `r0`-`r127`, 32 B each (4 KB per thread); subregister `.n` counts elements of the operand type (`r123.15:hf` = byte 30, `r72.2:ud` = byte 8) | [SPEC] whitepaper; Vol. 7 "GRF Registers" p. 754 |
| `r0` | thread header (group ids, scratch/SLM/barrier fields): never allocate it; `r1`-`r3` local ids (SIMD16), `r4`+ cross-thread data per `.ze_info` | [SPEC] Vol. 7 pp. 662-663; [MEAS] [no-neo.md](no-neo.md) §4 |
| `acc0/acc1` | accumulator; IGC uses it for 32x16-bit multiply idioms (`mul (8) acc0.0:ud ... :uw`); 3-source instructions cannot address it explicitly; `asm/isa.py` rejects acc operands | [SPEC] Vol. 2a `mad` p. 1091 |
| Types | `b/ub` 1 B, `w/uw/hf` 2, `d/ud/f` 4, `q/uq/df` 8; immediates `128:w`, `0x0:ud`, `-1e+30:f` | `asm/isa.py` `TS` |
| Exec size, mask | `(8|M0)`, `(16|M0)`; `(16|M16)` = lanes 16-31 of SIMD32, `(8|M8)` = second half of a split SIMD16 | [history.md](history.md) App. A |
| Size limit | ExecSize x largest element size <= 64 B; an operand spans at most 2 adjacent GRFs; dst stride never 0 | [SPEC] Vol. 7 "Register Region Restrictions" p. 799 |
| 2-source region `<vs;w,hs>` | `<8;8,1>` 8 contiguous, `<4;4,1>:q` 4 qwords per row, `<0;1,0>` scalar broadcast; width elements may not cross a GRF | [SPEC] Vol. 7 p. 799 |
| 3-source (`mad`) | Align16 only, GRF only, no immediates, dword-granular subregisters, replicate instead of regions ([SW] IGA still accepts half scalars such as `r118.15<0>:hf` in IGC's code). IGA prints src0/src1 vector `<2;1>`, scalar `<0;0>`, src2 vector `<1>`, scalar `<0>` (notation, not a stride; `isa.py` treats them as contiguous). `dst = src1 * src2 + src0` | [SPEC] Vol. 7 p. 870, p. 885; Vol. 2a p. 1091; [SW] `asm/micro.py` checked with `iga64 -p=9` |
| Immediates | only src1 of a 2-source instruction; 64-bit immediates only in 1-source instructions | [SPEC] Vol. 7 p. 885 |
| `(W)` | NoMask: runs regardless of the execution mask. Use it for scalar work (headers, counters) and on every `jmpi` (required) | [SPEC] Vol. 7 p. 814; Vol. 2a `jmpi` p. 618 |
| Flags | `f0.0 f0.1 f1.0 f1.1` (16 bits each); `(lt)f0.0` = conditional modifier; only enabled channels' bits are written; `sel` does not update flags | [SPEC] Vol. 7 "Flag Register" p. 766 |
| `jmpi` vs `break/while` | IGC's loops use structured per-channel `break`/`while`; generated loops use `(W&f0.0) jmpi` (exec size 1, channel 0 of the predicate decides), valid only for a uniform condition | [SPEC] Vol. 2a p. 618 |
| `send`/`sends` | the only memory access; `wr:a+b` registers sent (header + data), `rd:c` returned; `0xC` = DC1 (A64), `0x10A`/`0x4A` = DC0 (stateful BTS); comment decoded by IGA | [history.md](history.md) App. A |
| `{Compacted}` | 8 B encoding; no speed difference measured | [MEAS] `results/micro-2026-11-11.txt` (`compiler_full` vs `compiler_compact`) |

Descriptor `0x024D0CFF` (all block reads in `asm/`): mlen 1 (bits 28:25), rlen 4 (24:20), header present (19),
message type 14h A64 OWord block read (18:14), subtype 1 (12:11), 8 OWords (10:8 = 4), BTI 0xFF stateless [SPEC]
Vol. 2a "A64 Oword Unaligned Block Read MSD" p. 329, Vol. 2d `MDC_A64_DB_OW` p. 14. Naming trap: the PRM calls
subtype 1 "unaligned" (dword-aligned address, execution mask ignored, Vol. 7 p. 191-192), IGA prints it as "a64
aligned" (`IGALibrary/Backend/Messages/MessageDecoderHDC.cpp`); only the read path has it (Vol. 7 p. 192). Copy
descriptors from IGC's own disassembly rather than composing them.

## 4. Rules learned

### 4.1 GRF bank parity for `mad` src1/src2
Two banks by register-number parity; measured only for `mad`, where the conflict is between src1 and src2; src0 and
the destination do not matter; `src1 == src2` is free. Cost: vector/vector 410 -> 272 Gop/s (x0.66), scalar src2 of
the same parity 326-328 (x0.8) [MEAS] `results/micro-2026-11-11.txt`. Mesa models 4 banks
(`brw_opt_bank_conflicts.cpp`) [EST]; `bank_different_halves` (r70/r100) conflicts, so the 4-bank model is wrong
here. The PRM does not document banks. How we satisfy it: data layouts that alternate parity (`pack_a2`, A3, `pack_bf2`),
slot bases at even registers, Q elements permuted to the opposite parity, p kept in an even and an odd copy
(`asm/attn_gen.py`). Count with `asm/loop_analysis.py <file.asm> <label>`. Conflicts matter only when `mad` issue is
the limit (attention Qwen: 512 -> 0 conflicts, +0 %).

### 4.2 Mixed-mode hf -> f
An FP32 `mad` with a packed `:hf` src1 folds the conversion and runs at the full FP32 rate (`k_mix`: 323 vs 323
GFLOPS) [MEAS] `results/attn-kernel-hd530-2026-11-13.txt`. PRM: execution size counts float elements; no SIMD16
when the destination is packed f16; packed f16 output must be OWord-aligned [SPEC] Vol. 7 "Special Requirements for
Handling Mixed Mode Float Operations" pp. 801-802. IGC splits SIMD16 mixed `mov`/`mul`/`mad` into 2 x SIMD8 [SW]
`results/skinny-gemm-rework-2026-11-12.txt`. Chunked FP16 -> FP32 sums: `add (8|M0) rF:f rF:f rH.0:hf` and
`(8|M8)` with `rH.8` (`gemm_gen.py` `hd`).

### 4.3 FP16 double rate
Needs `mad (16) :hf` (16 halves per instruction): 803 GFLOPS [MEAS] `results/arith-throughput-latency-2026-11-11.txt`.
A broadcast scalar src2 in `mad` keeps it (HGEMM loop `r118.15<0>:hf`, 777 GFLOPS = 97 %). Two-source `mul`/`add :hf`
with a `<0;1,0>` scalar lose it (205 vs 400 Gop/s) [MEAS]. Note: [attention-gpu.md](attention-gpu.md) §8 attributes
variant B's slowness to "broadcast-scalar FP16 `mad`s"; that `mad` case was not measured in isolation and the HGEMM
result contradicts it; SIMD8 `:hf` `mad`s (8 halves per instruction) and dependent 8-`mad` chains are the likelier
cause [HYP].

### 4.4 Register budget: SIMD8 vs SIMD16
One FP32 value per lane = 1 GRF in SIMD8, 2 in SIMD16 (16 halves fit one GRF). Attention at d_head 128: FP32 Q and O
take 64 of 128 GRF for 2 rows; after IGC's prologue/epilogue needs, 50 registers were free for the loop and NS = 4
used them all [MEAS] `asm/attn_gen.py` output. Generators allocate explicitly (`Alloc`/`Registers` with parity) and
exit when a block does not fit; keep it that way rather than reusing registers by hand.

### 4.5 No spill, no private arrays
Private arrays above ~128 B per lane (`float s[8][8]` in SIMD8) go to scratch even when unrolled, and spills go to
scratch; kcl does not configure scratch, so results are **silently wrong** (correct through NEO) [SW]
`results/skinny-gemm-rework-2026-11-12.txt`, `results/skinny-gemm-2026-11-12.txt`. Check `.ze_info`
(`spill_size`, `private_size`, `per_thread_scratch_size`) or `CL_KERNEL_SPILL_MEM_SIZE_INTEL` /
`CL_KERNEL_PRIVATE_MEM_SIZE` (printed by `bench/attn`). `lib/gen_kernels.py` rejects `spill_size` and scratch but
not `private_size` alone. A spilling template is useless: its prologue/epilogue would carry scratch traffic.
**Scratch can hide outside `execution_env`** [MEAS] (commit e228aef): the old Q4_K dequant kernel showed only
`private_size: 128` and no spill, but declared `per_thread_memory_buffers: type scratch, size 1024`, addressed it
from `r0.5 & -1024` and wrote through BTI 0xFF; under kcl 198-335 thousand of 6.3 M values were wrong at N = 4096,
through NEO none. Always grep the whole `.ze_info` for `per_thread_memory_buffers` (the attention binaries have
none); `lib/gen_kernels.py` now rejects it. Open: `gemv_n_*` / `gemv_nw_*` (`level12.cl`) also declare scratch
and read `r0.5`, yet pass all tests (allowed by name; cause not explained).

### 4.6 Addresses
64-bit multiply is emulated (73 instructions; INT64 mul 12-15 Gop/s, add 50) [MEAS/SW]
`results/arith-throughput-latency-2026-11-11.txt`, [pitfalls.md](pitfalls.md) §1; in a loop it cost 1.2-1.9x
(skinny GEMM). Compute row pointers before the loop. For block reads keep one 64-bit address per stream in the send
header and step it with one scalar `(W) add (1|M0) rH.0:q rH.0:q imm:w`: the attention loop has 67 such adds per
iteration where IGC had 72 SIMD8 `add (8) :q` [MEAS] `results/attn-asm-hd530-2026-11-13.txt`.

### 4.7 Latency: keep loads in flight
Dependent A64 OWord block read: L3 133 ns (147 clocks), LLC 200-270 ns, RAM 400 ns (440 clocks) [MEAS]
`results/memory-latency-2026-11-11.txt`; Mesa assumes 200-300 clocks for block reads [EST]
`brw_schedule_instructions.cpp`. L3 bandwidth needs >= 2 block reads in flight per thread (117 vs 156-159 GB/s)
[MEAS] `results/memory-bandwidth-2026-11-11.txt`. Pattern: a ring of NS slots of 8 GRF; after the `mad`s of step s
issue the load for step s+NS into the slot just consumed (attention: +25 % over the same template, NS 4 +3 % over
NS 2); GEMM: double-buffered halves/quarters of k, 2 quarters ahead. Writebacks set in-flight bits on the destination
registers; any instruction reading them waits, and registers of one message may return out of order [SPEC] Vol. 7
"Writebacks", "Message Delivery Ordering Rules" pp. 749-750. Gen9 has no SBID/SWSB (IGA `getSWSBEncodeMode()`
returns invalid before Xe): no software dependency annotations, no barriers needed between `send` and `mad`.

### 4.8 Message type and width
Prefer A64 OWord block reads (`send (16) ... rd:4` = 128 B into 4 GRF) over what IGC emits for vector loads: `vload8`
of `half` becomes byte-scattered 4 B reads [SW]. Per SIMD16 lane, 16 B covers whole 64 B lines; 32 B per lane splits
lines between two messages (FP64 COPY 17.4 -> 24.2 GB/s after the change) [MEAS] `results/message-16B-2026-11-12.txt`.
Media block reads equal OWord from L3 (153 GB/s) but fall to 83 at 4 MB [MEAS] `results/memory-bandwidth-2026-11-11.txt`.
SLM: 177 GB/s consecutive or broadcast, 96 GB/s at stride 2 (HD 530, 1.0 GHz) [MEAS]
`results/attn-micro-hd530-2026-11-13.txt`; SLM shares the data port with L3 [EST] [attention-gpu.md](attention-gpu.md) §9.

### 4.9 Send headers
A64 OWord block header: DW0-1 = 64-bit address, DW2-7 ignored [SPEC] Vol. 2d `MH_A64_OWB` p. 13; the BTS block header
uses only DW2 (global offset) [SPEC] Vol. 2d `MH_BTS_GO` p. 56. The header may be updated right after the `send`
([MEAS] `asm/sgemm_asm.py`); the PRM backs it: with `NoSrcDepSet` = 0 (default) the hardware tracks write-after-read
on send sources [SPEC] Vol. 7 `NoSrcDepSet` p. 877. Zero each header once with `(W) mov (8|M0) rH.0:ud 0x0:ud` and
then write only the address field. Hardening case [MEAS] `results/attn-asm-hd530-2026-11-13.txt`: IGC's epilogue
builds the O-store header in r72 (a dead Q register, also used by our loop) with partial writes `mov (1) r72.2:ud`,
leaving the other dwords as found; the generator now zeroes such epilogue headers after the loop. For this message
the stale dwords are ignored by the spec, so it was a latent rather than an observed bug; for messages that read more
header fields it would not be.

### 4.10 Prefetch over-read needs padding
A ring that prefetches NS steps also prefetches past the last block: the attention K buffer is read up to
(NS-1)*128 + (NK-1)*D*16 + 128 B past its end (Qwen ring 4: 2560 B), so the host pads NK*D*16 B (`KT_PAD`) and the
generator asserts NS <= D/8 [SW] `asm/attn_gen.py` docstring. The PRM defines out-of-bounds only for addresses outside
the 48-bit canonical range [SPEC] Vol. 2d p. 13; what an over-read past the allocation returns is not specified: pad.

### 4.11 Liveness with partial and predicated writes
Only full, unpredicated writes end a live range: `(1|M0) :q` header updates, `.2:ud` fields and predicated writes do
not (`isa.full_writes()`). Non-(W) instructions are treated as full writes only because the kernels run with full
SIMD8 subgroups in straight-line regions; the host must enforce that (`bench/attn.c` rejects partial subgroups).
Anything the prologue modifies in place (permuted Q) must not be read after the loop: `assert_not_read_after()`.
When reusing IGC's epilogue, feed it consistent inputs (attention sets IGC's `l` registers to 1/8 per lane so the
epilogue's division by sum(l) divides by 1).

### 4.12 Loop structure assumptions
The generators assume IGC's loop has exactly one `cmp (lt)` on two scalar `:d` registers, a counter written only by
`cnt += step`, a limit not written, and an exit `(~flag) break` on that flag; anything else stops the generator
(`igc_maps()` asserts), and every analysed line must be a form `asm/isa.py` understands (`check_all`: no indirect,
no acc, known regions, sends with `rd:`/`wr:`). Generated loops end with `(W) add` counter, `(W) cmp`, `(W&f0.0) jmpi`
(the GEMM generators omit `(W)` on `cmp`; harmless with all channels enabled, but per Vol. 7 p. 766 a disabled
channel 0 would leave its flag bit stale). Host preconditions the loop does not check: no tail (n_k multiple of 8*NK).

Pointer registers kept across the loop may sit in any :q subregister (`r6.1:q` = bytes 8-15), not only `.0`:
a generator that tracks registers by number alone emits the wrong half of the register and the binary runs at full
speed with error 1.0 (seen in `asm/attn_gen.py` with `-DHKV_INNER`, fixed) [MEAS]
`results/attn-kcl-hd530-2026-11-13.txt`. Track (register, subregister) for every value taken from IGC's code.

### 4.13 Launch length and memory kind
A launch that keeps the GPU longer than ~640 ms while the desktop wants it is reset and kcl's wait still returns
success: silent zeros [MEAS] [pitfalls.md](pitfalls.md) §4, [no-neo.md](no-neo.md) §6a; keep launches far below
(gen9blas: 100 ms; attention: at most 2e10 FLOP per launch). userptr memory loses 2x when a kernel touches hundreds of
4 KB pages at once [MEAS] `results/gemv-n-2026-11-12.txt`: keep each work-group's walk sequential.

## 5. Verification checklist for every new binary

1. **FP64 reference**, error bound fixed before the run (attention: max |O - O_ref| / max |O_ref| <= 2e-3).
2. **Whole output once** (`STEP=1` in `bench/attn`): the sampled default (`STEP` 7) missed rows before
   (`STEP` 16 visited only rows 0-5 of each 32-row group) [MEAS] `results/attn-asm-hd530-2026-11-13.txt`.
3. **Poison binary**: one `mad` negated must give a wrong result at the same speed (attention: error 0.775), and
   the `plain` binary (IGC's code without `.spv`) must match the source's speed: proves the driver runs our ISA.
4. **spill = private = 0** in `.ze_info` / the host printout; template too.
5. **`isa.check_all`** passes on every analysed and generated region; conflicts reported by the generator = 0.
6. **Deterministic rebuild + hash**: `make -C asm check` (`gemm-check`, `attn-check`); `make -C asm attn-sha`
   only after a deliberate generator change.
7. **Both devices**: `skl` (HD 530) and `whl` (UHD 620) binaries, run on each machine.
8. **Clock recorded** (`GPU~MHz` from `gt_act_freq_mhz` in `clbench`/`bench/attn`), nice 19, nothing else on the GPU.
9. **Raw output kept** in the results file next to the prediction written before the run.
10. Host preconditions checked by the host (sizes, full subgroups, padding), not assumed.

## 6. Measurement tools

- **Kernel time**: `bench/clbench` (profiling events `COMMAND_START/END`, run 0 = warm-up, best of `REPS`=5, GPU
  clock sampled during the run); `bench/attn` prints the same plus error. Profiling costs ~20 us CPU per event, so
  never use a profiling queue for call-time (wall) measurements ([performance-model.md](performance-model.md) §3).
- **Instruction micro-tests**: `asm/micro.py` inserts a generated body into the `asm/fma32.cl` loop (registers
  r51-r126 free), `asm/micro.sh` runs all; memory: `bench/memory.cl` (`k_owb`, `k_owb2`, `k_media`, `k_lat`);
  `bench/attn_micro.cl` (`exp`, SLM, `k_mix`).
- **Clock and power**: `intel_gpu_top -J` via `bench/power.sh` / `bench/long_run.sh`; package energy via RAPL
  (`bench/compare/rapl_run.sh`, needs read access to `energy_uj`). Method: [benchmarks.md](benchmarks.md).

### OA counters

The i915 observation architecture (OA) counters say **where the time goes**, which kernel timing alone cannot:
EU active vs stalled vs without a thread, dual-FPU issue, send pipe, L3/GTI/SLM traffic. Every kernel result
should come with this breakdown next to its GFLOPS.

- **Enable** (root, once per boot or via `/etc/sysctl.d/`): `sysctl dev.i915.perf_stream_paranoid=0`. The counters
  are GPU-wide, so other processes' GPU activity is visible; acceptable on a single-user test machine.
- **Tools**: igt-gpu-tools `i915-perf-recorder` / `i915-perf-reader` (metric sets are loaded by the tool; Gen9 has
  21, `i915-perf-recorder -m list`). The project wrapper: `bench/oa.sh [-n runs] <command>` records metric set
  `ComputeBasic` around n runs and prints `bench/oa_sum.py`'s summary; `OA_KEEP=<file>` keeps the raw reports.
- **Reading** (summary lines): EU active / stall / idle (= 100 - active - stall, derived: EU time with no thread),
  thread occupancy, both FPU pipes active (and as a share of active time: dual issue of the FPU pipes), send pipe,
  IPC, L3 shader / GTI / SLM bytes per second of busy GPU time. Counter symbols for `i915-perf-reader -c`:
  `i915-perf-reader -c list <record>` (e.g. `EuActive`, `EuStall`, `EuFpuBothActive`, `L3ShaderThroughput`).
- **Limits** [MEAS]: on the desktop HD 530 the reports arrive only at context switches, so each number averages
  whole launches (no start/tail split); repeated recordings agree within ~3 points. With launches of ~90 ms the
  summary became inconsistent with the rate (EU active 55 % at the same GFLOPS as a 75 % run) [HYP: report
  intervals above ~180 ms wrap a 32-bit EU-cycle counter, 24 EU x 1 GHz]: keep launches under OA <= ~50 ms. `GtiReadThroughput` varied
  0.8-12.9 GB/s between similar kernels and is not interpreted until its semantics are checked.
  `L3ShaderThroughput` is **not calibrated** either: the Qwen attention loop must read 148 GB/s at 296 GFLOPS
  (64 sends x 128 B per 1024 SIMD8 mads = 8 B per mad), yet the counter shows 112-118 GB/s (76-80 %); calibrate it
  on a microbenchmark of known demand before using it as a bandwidth figure. Counters say nothing about correctness.
- **First use** (`results/oa-attention-hd530-2026-11-13.txt`): the asm attention kernel is not memory-bound
  (stall 2-3 %, both FPU pipes busy 86-88 % of active time); its loss is ~20 % (Qwen) / 13 % (Llama) of EU time
  without a thread, which overturned the L3-bandwidth explanation and points at work distribution [HYP:
  work-group granularity, 3 groups of 16 threads in 56 slots per subslice]. Follow-up
  (`results/attn-kcl-hd530-2026-11-13.txt`): not the slot cap; work-group 128 alone has 8x the GTI reads and
  29 % idle, every size from 8 to 96 gives 7-14 % idle and +4 % (Qwen); 64 is now the default.

## 7. Open gaps and how to measure them

| Unknown | Why it matters | How |
|---|---|---|
| Latency per message type (scattered, untyped, media, SLM, A64 vs BTS, writes) | ring depth for other kernels; only the dependent A64 OWord read is measured | `k_lat`-style chase in `bench/memory.cl` per message, or a `micro.py` body with one `send` + dependent `mov` |
| Write bandwidth per message (OWord/media block writes) | epilogues, packing | `bench/memory.cl` write variants ([performance-model.md](performance-model.md) §5) |
| Co-issue of `mad` with `send`/`add`/`math` (FPU vs EM pipe) | Llama attention at 81 % of peak is held by ~15 % non-`mad` instructions [EST] | `micro.py`: fixed `mad` stream plus k interleaved `add (1) :q` / `send`; watch whether time grows |
| Bank rules beyond `mad` (2-source ops, `math`, sends reading GRF) | softmax and header code | `micro.py` variants of the `bank_*` tests |
| ~~Thread arbitration (round-robin `0xE404` vs age) on ISA loops~~ | measured: the attention kernel loses 10 % with `KCL_ARB=age` (275 vs 305 GFLOPS) [MEAS] `results/attn-kcl-hd530-2026-11-13.txt` | done |
| Why exp in place costs 5 % | attention tuning | instruction-order variants; repeated alternating runs (the `LOCAL=24` question is answered: every work-group size except 128 is equally fast, [attention-gpu.md](attention-gpu.md) section 10) |
| Why work-group 128 has 8x the GTI reads and ~2x the idle time of 8-96 in the attention kernel | the cause is open; 64 is used (+4 % Qwen) | thread placement per subslice (e.g. a counter of dispatched groups per subslice, or 2D grids) |
| Preemption granularity under kcl (no mid-thread preemption) | safe launch length | [HYP] work-group boundary; test with a long single group while the desktop draws |
| TLB reach | page-heavy access patterns | [performance-model.md](performance-model.md) §5 |
