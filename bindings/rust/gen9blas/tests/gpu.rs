//! GPU tests: results against a plain CPU reference. They need a Gen9 GPU and the library built in lib/:
//!   cargo test -- --ignored      (run under the GPU lock on a shared machine)
use gen9blas::*;

fn data(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5 }).collect()
}

fn gemm_ref(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f64> {
    let mut c = vec![0f64; m * n];
    for i in 0..m { for p in 0..k { let x = a[i * k + p] as f64; for j in 0..n { c[i * n + j] += x * b[p * n + j] as f64; } } }
    c
}

fn max_rel(got: impl Iterator<Item = f64>, want: &[f64]) -> f64 {
    let scale = want.iter().fold(0f64, |s, x| s.max(x.abs())).max(1e-30);
    got.zip(want).fold(0f64, |e, (g, w)| e.max((g - w).abs() / scale))
}

#[test]
#[ignore = "needs a Gen9 GPU"]
fn sgemm_dgemm_hgemm() {
    init().unwrap();
    let (m, n, k) = (129, 77, 300);
    let (a, b) = (data(1, m * k), data(2, k * n));
    let want = gemm_ref(m, n, k, &a, &b);
    let mut c = vec![0f32; m * n];
    gemm(Trans::No, Trans::No, m, n, k, 1.0, &a, k, &b, n, 0.0, &mut c, n).unwrap();
    assert!(max_rel(c.iter().map(|&x| x as f64), &want) < 1e-5);
    let (ad, bd): (Vec<f64>, Vec<f64>) = (a.iter().map(|&x| x as f64).collect(), b.iter().map(|&x| x as f64).collect());
    let mut cd = vec![0f64; m * n];
    gemm(Trans::No, Trans::No, m, n, k, 1.0, &ad, k, &bd, n, 0.0, &mut cd, n).unwrap();
    assert!(max_rel(cd.iter().copied(), &want) < 1e-12);
    if has(Feature::Fp16) {
        let (ah, bh): (Vec<F16>, Vec<F16>) = (a.iter().map(|&x| F16::from(x)).collect(), b.iter().map(|&x| F16::from(x)).collect());
        let mut ch = vec![F16::default(); m * n];
        hgemm_accurate(Trans::No, Trans::No, m, n, k, 1.0, &ah, k, &bh, n, 0.0, &mut ch, n).unwrap();
        assert!(max_rel(ch.iter().map(|&x| x.to_f32() as f64), &want) < 3e-3);
    }
}

#[test]
#[ignore = "needs a Gen9 GPU"]
fn packed_b_and_usm() {
    init().unwrap();
    let (m, n, k) = (64, 96, 128);
    let (a, b) = (data(3, m * k), data(4, k * n));
    let want = gemm_ref(m, n, k, &a, &b);
    let p = PackedB::new(Trans::No, k, n, &b, n).unwrap();
    let au = UsmBuffer::from_slice(&a).unwrap();
    let mut cu = UsmBuffer::<f32>::zeroed(m * n).unwrap();
    for _ in 0..3 { p.gemm(Trans::No, m, 1.0, &au, k, 0.0, &mut cu, n).unwrap(); }
    assert!(max_rel(cu.iter().map(|&x| x as f64), &want) < 1e-5);
}

#[test]
#[ignore = "needs a Gen9 GPU"]
fn level12_and_errors() {
    init().unwrap();
    let x = data(5, 4097);
    let mut y = data(6, 4097);
    let d: f64 = x.iter().zip(&y).map(|(a, b)| *a as f64 * *b as f64).sum();
    assert!((dot(&x, &y).unwrap() - d).abs() < 1e-4);
    let y0 = y.clone();
    axpy(0.5, &x, &mut y).unwrap();
    assert!(y.iter().zip(&y0).zip(&x).all(|((r, b), a)| (r - (b + 0.5 * a)).abs() < 1e-6));
    let i = iamax(&x).unwrap().unwrap();
    assert_eq!(x[i].abs(), x.iter().fold(0f32, |m, v| m.max(v.abs())));
    let (m, n) = (300, 200);
    let a = data(7, m * n);
    let mut yv = vec![0f32; m];
    gemv(Trans::No, m, n, 1.0, &a, n, &x[..n], 0.0, &mut yv).unwrap();
    let w0: f64 = (0..n).map(|j| a[j] as f64 * x[j] as f64).sum();
    assert!((yv[0] as f64 - w0).abs() < 1e-4);
    // an A slice too short for the packed B shape is caught before the library call
    let p = PackedB::new(Trans::No, 8, 8, &a[..64], 8).unwrap();
    let mut c = vec![0f32; 64];
    assert_eq!(p.gemm(Trans::No, 8, 1.0, &a[..32], 8, 0.0, &mut c, 8).unwrap_err().status, Status::InvalidArg);
}
