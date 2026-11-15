// Links libgen9blas. Search order: GEN9BLAS_LIB_DIR, pkg-config style GEN9BLAS_PREFIX/lib, then the library built in
// this repository (lib/, three levels up from this crate). An rpath to the chosen directory is added so tests and
// examples run without LD_LIBRARY_PATH.
use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=GEN9BLAS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=GEN9BLAS_PREFIX");
    let dir = env::var_os("GEN9BLAS_LIB_DIR")
        .map(PathBuf::from)
        .or_else(|| env::var_os("GEN9BLAS_PREFIX").map(|p| PathBuf::from(p).join("lib")))
        .unwrap_or_else(|| {
            PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../../lib")
        });
    let dir = dir.canonicalize().unwrap_or(dir);
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=dylib=gen9blas");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    println!("cargo:lib_dir={}", dir.display());
}
