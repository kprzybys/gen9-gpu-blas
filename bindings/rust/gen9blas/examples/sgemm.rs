//! cargo run --release --example sgemm [n]   -- FP32 GEMM n×n×n on the GPU, time and GFLOP/s
use gen9blas::{gemm, init, last_time_ms, Trans, UsmBuffer};

fn main() -> gen9blas::Result<()> {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2048);
    init()?;
    let a = UsmBuffer::from_slice(&vec![1.0f32; n * n])?;
    let b = UsmBuffer::from_slice(&vec![0.5f32; n * n])?;
    let mut c = UsmBuffer::<f32>::zeroed(n * n)?;
    gemm(Trans::No, Trans::No, n, n, n, 1.0, &a, n, &b, n, 0.0, &mut c, n)?;   // warm-up
    let t = std::time::Instant::now();
    gemm(Trans::No, Trans::No, n, n, n, 1.0, &a, n, &b, n, 0.0, &mut c, n)?;
    let s = t.elapsed().as_secs_f64();
    println!("sgemm {n}: {:.1} ms ({:.1} GFLOP/s), library {:.1} ms, c[0] = {}", s * 1e3, 2.0 * (n as f64).powi(3) / s * 1e-9,
             last_time_ms(), c[0]);
    Ok(())
}
