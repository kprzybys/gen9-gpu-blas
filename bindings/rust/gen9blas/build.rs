// rpath to the libgen9blas directory found by gen9blas-sys (links metadata), for this crate's tests and examples.
// Crates using gen9blas find the library through the system paths, LD_LIBRARY_PATH or their own rpath.
fn main() {
    if let Ok(dir) = std::env::var("DEP_GEN9BLAS_LIB_DIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
}
