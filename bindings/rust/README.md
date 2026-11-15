# Rust bindings for libgen9blas 0.2

- `gen9blas-sys`: raw FFI to every function of `lib/gen9blas.h` and `lib/gen9blas_ml.h` (hand-written; checked against
  the headers by `tools/check_rust_ffi.py`).
- `gen9blas`: safe API for the BLAS part. Row-major slices with leading dimensions, every length checked against the
  shape. Element types `f32`, `f64`, `F16`. `Result<_, Error>` with the library's status class and message. One
  mutex around all calls, because the C library has process-wide state. `UsmBuffer` gives zero-copy memory and
  `PackedB` keeps weights packed on the GPU; both free themselves on drop. The neural-network part (unstable) is only
  available through `gen9blas::ffi`.

```sh
make -C lib libgen9blas.so            # the library (see the top-level README)
cd bindings/rust
cargo test                            # no GPU needed: F16 conversion, shape checks
cargo test -- --ignored               # GPU tests against a CPU reference (Gen9 iGPU; under the GPU lock on shared hosts)
cargo run --release --example sgemm 2048
```

Linking: `GEN9BLAS_LIB_DIR` (directory with `libgen9blas.so`) or `GEN9BLAS_PREFIX` (an install prefix). Without
either, the build uses `lib/` of this repository. Tests and examples get an rpath to that directory. Other crates
find the library through the system paths, `LD_LIBRARY_PATH` or their own rpath.

```rust
use gen9blas::{gemm, PackedB, Trans};
let (m, n, k) = (64, 4096, 1024);
let (x, w) = (vec![0.5f32; m * k], vec![0.25f32; k * n]);
let mut y = vec![0f32; m * n];
gemm(Trans::No, Trans::No, m, n, k, 1.0, &x, k, &w, n, 0.0, &mut y, n)?;
let wp = PackedB::new(Trans::No, k, n, &w, n)?;                  // pack once
wp.gemm(Trans::No, m, 1.0, &x, k, 0.0, &mut y, n)?;              // multiply many times
```
