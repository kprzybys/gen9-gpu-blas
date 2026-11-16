# Gen9 schedule rules for TVM DLight and an MLC-LLM preset

Status (2026-11-16): **built and measured on gp-laptop (UHD 620).** Qwen2.5-1.5B-Instruct `q4f16_1`, prompt 512,
128 generated tokens, identical output: prefill **10.9 -> 34.8 t/s (3.2x)**, decode **8.15 -> 8.40 t/s (+3 %)**
against MLC's generic OpenCL constants. Full protocol and all numbers: `results/mlc-dlight-gen9-2026-11-16.txt`.
Both stay far below this project's hand-written path (GEN9 ggml backend ~220 t/s pp512 on the same model class).

Background: MLC-LLM compiles models with TVM; kernels come from DLight rules that pick constants by target kind
(`docs/kernel-guide.md` section 6). There is no Intel branch, so an Intel GPU gets the generic constants (scalar
GEMV loads, 8x8 matmul). These patches add a Gen9 branch, selected by `"device": "intel_gen9"` in the target.

## Files

| File | Applies to | Content |
|---|---|---|
| `relax-dlight-gen9.patch` | `github.com/mlc-ai/relax` at `45221f267` (MLC-LLM's TVM submodule) | Gen9 branches in `python/tvm/s_tir/dlight/gpu/{gemv,low_batch_gemv,matmul}.py` |
| `mlc-llm-preset-gen9.patch` | `github.com/mlc-ai/mlc-llm` at `3b7663a` | presets `intel:gen9` (OpenCL) and `intel:gen9-vulkan` in `python/mlc_llm/support/auto_target.py` |
| `opencl_local_hoist.py`, `sitecustomize.py` | TVM's OpenCL codegen, at compile time | moves `__local` declarations to kernel scope; **required on Intel NEO for any target**, generic included |
| `bench_mlc.py` | an MLC model library | prefill/decode speed, batch 1, distinct prompt per run |

Apply with `git apply <patch>` inside the checkout (`3rdparty/tvm` for the relax patch). Compile with this
directory on `PYTHONPATH` (for the hook) and `mlc_llm compile ... --device intel:gen9`. The Vulkan preset is
untested (the build had OpenCL only).

## Constants (final) and evidence

| Rule | Constant | Value | Evidence |
|---|---|---|---|
| GEMV, inner reduction (decode path at batch 1) | `TS, TR` | 1, 32 | sweep: TR 256: 5.7 t/s, 128: 8.2, 64: 9.8, **32: 10.4**, 16: 7.5, 8: 4.9 (prompt 64); TS 2-16 < 3 % |
| | `VEC_C`, `VEC_LOAD`, `UNROLL` | 1, 1, 64 | widths 1-4 / 1-8 and unroll 16-64 changed < 3 % |
| GEMV, outer reduction (KN layout) | lanes along N, `TS, TR` | 64, 4 | like `skinny_n4`; not on the path of this model (NK weights); no measurable effect |
| low-batch GEMV | `TS x TR`, `VEC_C`, `UNROLL` | 2x128 / 4x64, 4, 16 | not on the decode path at batch 1 (8 variants: 7.73-7.76 t/s); untuned |
| Matmul (prefill) | output tile | 64 x 64 (`block 16 x 8`, `micro 4 x 8`) | CLBlast's tuned MWG = NWG = 64 on the UHD 620 [EXT]; gives the whole 3.2x |
| | `micro 4 x 8` | 32 FP32 accumulators = 128 B per lane | largest private array IGC 24.35 keeps in registers (`results/skinny-gemm-rework-2026-11-12.txt`) |
| | `vector_size`, `storage_align`, `use_shared` | 4, on, on | 16 B loads; SLM stride-2 reads at half speed; `use_shared` off not yet measured |
| preset | threads, local memory | 256, 64 KB | `clinfo`, `vulkaninfo` on the UHD 620 |

**Lesson:** the first GEMV constants (TR 128, from our skinny kernels' "1 KB contiguous per row" rule) made decode
19 % slower. Our kernels reduce mostly inside a thread and finish with sub-group shuffles; TVM's schedules (with
`thread_warp_size` 1) reduce across TR threads in local memory with log2(TR) barriers, so a small TR wins.
A rule transfers only together with the code shape it was measured on.

## Build notes (2026-11-16)

- Built on evox2 (`~/gen9-build/tvm-mlc`, glibc 2.44 = gp), run on gp (`~/src/mlc-llm`, `env.sh`, venv
  `~/opt/venv-mlc`, Python 3.14). LLVM 21 linked statically through a wrapper that drops `libPolly*.a`
  (listed by `llvm-config` but not installed); `libz3.so.4` copied next to the TVM libraries (gp lacks it).
- `apache-tvm-ffi` must match the submodule: installed from `3rdparty/tvm/3rdparty/tvm-tirx/3rdparty/tvm-ffi`
  (0.1.15.post6), not the PyPI 0.1.16.
- Without the hoist hook NEO fails with `CL_BUILD_PROGRAM_FAILURE` ("variables in the local address space can only
  be declared in the outermost scope") and MLC's engine thread dies while the caller waits forever.
- MLC's prefix cache: repeating the same prompt skips the prefill (the first attempt "measured" 2400-3500 t/s).
- Compile time ~7 min the first time, ~40 s afterwards.

## Open

- Matmul sweep (`use_shared` off, micro 4x4 / 8x4, block 8x8 / 16x16); this is where the gap to hand-written
  kernels is largest (34.8 vs ~220 t/s).
- Sub-group shuffles in TVM's OpenCL codegen for Intel (`cl_intel_subgroups`) would let GEMV use larger TR.
- Vulkan path (needs a TVM build with Vulkan/SPIR-V).
