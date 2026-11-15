# tools/ — refactor and maintenance tools

| Tool | What it does |
|---|---|
| `repo-map.toml` | target repository of every tracked path (lib / backend / apps / badania) and the allowed dependency order |
| `depgraph.py` | dependency edges between paths (Makefile paths, `#include`, file names in strings, Python imports, kernel names); `--check` fails on edges that break the repo order, `--dirs` shows directory coupling, `--docs` cross-repo links |
| `kernels.py` | inventory of OpenCL kernels: users, unnamed (unused) kernels, same name / identical body in several files |
| `dup.sh` | copy-paste detection (jscpd 4.0.5) |
| `metrics.sh` | all of the above plus portability counters; baseline in `results/refactor-metrics-2026-11-15.txt` |
| `golden/golden.sh` | builds every component and compares GPU outputs with recorded values (`expected/<gpu>/`): `isa` (kernel binaries and tables), `lib` (BLAS result hashes + test suites), `backend` (test-backend-ops, perplexity), `apps` (mel, silero, ina, emb, dec). `--record` rewrites the expected files. ~6.5 min on UHD 620; run under the GPU lock. |
| `check_rust_ffi.py` | the Rust FFI declarations (`bindings/rust/gen9blas-sys`) match the C headers: every API function with the same parameter and return types, and the same constants. Exit 1 on a mismatch; also run by golden `lib`. |

Known failure recorded in the baseline (not caused by the refactor): test-backend-ops `MUL_MAT_VEC_FUSION` type nvfp4 with
bias + lane scale, the fused `ADD` on GEN9 (3 of 1807 tests).
