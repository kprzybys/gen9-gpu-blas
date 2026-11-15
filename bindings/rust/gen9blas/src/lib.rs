//! Safe Rust API for libgen9blas 0.2: BLAS levels 1-3 on Intel Gen9 iGPUs (UHD 620 / HD 530).
//!
//! - All matrices are row-major slices with a leading dimension (`ld` = elements between row starts).
//!   Every slice length is checked against the shape before the GPU sees it.
//! - Element types: `f32`, `f64` and [`F16`] (IEEE half as bits). FP16 GEMM accumulates in FP16; [`hgemm_accurate`] and
//!   [`PackedB::new_accurate`] use FP32 sums of 32-product FP16 chunks.
//! - The C library has process-wide state; this crate serializes every call through one mutex, so it is safe to use
//!   from several threads (calls do not run concurrently).
//! - Memory from [`UsmBuffer`] is shared with the GPU (zero-copy); ordinary slices are pinned (i915 userptr) or copied.
//! - Neural-network operations (`gen9blas_ml.h`, unstable) are only in [`ffi`].
//!
//! ```no_run
//! let (m, n, k) = (256, 256, 256);
//! let a = vec![1.0f32; m * k];
//! let b = vec![1.0f32; k * n];
//! let mut c = vec![0.0f32; m * n];
//! gen9blas::gemm(gen9blas::Trans::No, gen9blas::Trans::No, m, n, k, 1.0, &a, k, &b, n, 0.0, &mut c, n).unwrap();
//! assert_eq!(c[0], k as f32);
//! ```
use std::ffi::{c_int, c_void, CStr};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, MutexGuard};

pub use gen9blas_sys as ffi;

// ---------------------------------------------------------------------------------------------------------------------
// Errors

/// Class of a failure (`gen9blas_status` of the C library).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The fast path did not take the call; nothing was written (only some functions decline).
    Declined,
    /// GPU execution or driver error.
    Gpu,
    /// Type, kernel or layout not available in this build or on this device.
    Unsupported,
    /// Invalid arguments (detected by this crate or by the library).
    InvalidArg,
    /// GPU or host allocation failed.
    OutOfMemory,
    /// No usable GPU (initialization failed; permanent for the process).
    NoDevice,
    /// A status this crate does not know (newer library).
    Other(i32),
}

impl Status {
    fn from_code(rc: c_int) -> Status {
        match rc {
            ffi::GEN9BLAS_DECLINED => Status::Declined,
            ffi::GEN9BLAS_ERR => Status::Gpu,
            ffi::GEN9BLAS_ERR_UNSUPPORTED => Status::Unsupported,
            ffi::GEN9BLAS_ERR_INVALID_ARG => Status::InvalidArg,
            ffi::GEN9BLAS_ERR_OUT_OF_MEMORY => Status::OutOfMemory,
            ffi::GEN9BLAS_ERR_NO_DEVICE => Status::NoDevice,
            x => Status::Other(x),
        }
    }
}

/// Error of a call: its class and the library's description.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub status: Status,
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gen9blas: {:?}: {}", self.status, self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn invalid(msg: impl Into<String>) -> Error {
    Error { status: Status::InvalidArg, message: msg.into() }
}

// ---------------------------------------------------------------------------------------------------------------------
// Serialization: one lock for every call into the library (its state is process-wide)

static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Error from the library's last call; must be read while the lock is held.
fn last_error(rc: c_int) -> Error {
    let msg = unsafe { CStr::from_ptr(ffi::gen9blas_error()) }.to_string_lossy().into_owned();
    Error { status: Status::from_code(rc), message: msg }
}

fn status(rc: c_int) -> Result<()> {
    if rc == ffi::GEN9BLAS_OK { Ok(()) } else { Err(last_error(rc)) }
}

/// Runs one library call under the lock and converts its status.
fn call(f: impl FnOnce() -> c_int) -> Result<()> {
    let _g = lock();
    status(f())
}

fn int(x: usize, what: &str) -> Result<c_int> {
    c_int::try_from(x).map_err(|_| invalid(format!("{what} = {x} does not fit a C int")))
}

/// Elements a row-major `rows` x `cols` matrix with leading dimension `ld` spans.
fn span(rows: usize, cols: usize, ld: usize, what: &str) -> Result<usize> {
    if ld < cols.max(1) {
        return Err(invalid(format!("{what}: ld {ld} < {cols} columns")));
    }
    Ok(if rows == 0 || cols == 0 { 0 } else { (rows - 1) * ld + cols })
}

fn check_len(len: usize, need: usize, what: &str) -> Result<()> {
    if len < need { Err(invalid(format!("{what}: slice of {len} elements, the shape needs {need}"))) } else { Ok(()) }
}

/// Checks that `s` holds a `rows` x `cols` (row-major, leading dimension `ld`) matrix.
fn mat<T>(s: &[T], rows: usize, cols: usize, ld: usize, what: &str) -> Result<()> {
    check_len(s.len(), span(rows, cols, ld, what)?, what)
}

// ---------------------------------------------------------------------------------------------------------------------
// Element types

/// IEEE 754 binary16 stored as its bits (convert with [`F16::from_f32`] / [`F16::to_f32`]).
#[repr(transparent)]
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct F16(pub u16);

impl F16 {
    /// Round-to-nearest-even conversion (as the GPU kernels do).
    pub fn from_f32(x: f32) -> F16 {
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32;
        let man = b & 0x7f_ffff;
        if exp == 0xff {
            return F16(sign | 0x7c00 | if man != 0 { 0x200 } else { 0 });   // inf / NaN
        }
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return F16(sign | 0x7c00);   // overflow -> inf
        }
        if e <= 0 {   // subnormal or zero
            if e < -10 {
                return F16(sign);
            }
            let m = man | 0x80_0000;
            let shift = (14 - e) as u32;
            let half = 1u32 << (shift - 1);
            let rest = m & ((1u32 << shift) - 1);
            let mut r = m >> shift;
            if rest > half || (rest == half && (r & 1) == 1) {
                r += 1;
            }
            return F16(sign | r as u16);
        }
        let mut r = ((e as u32) << 10) | (man >> 13);
        let rest = man & 0x1fff;
        if rest > 0x1000 || (rest == 0x1000 && (r & 1) == 1) {
            r += 1;   // may carry into the exponent (correct: rounds up to the next binade or inf)
        }
        F16(sign | r as u16)
    }

    pub fn to_f32(self) -> f32 {
        let h = self.0 as u32;
        let sign = (h & 0x8000) << 16;
        let exp = (h >> 10) & 0x1f;
        let man = h & 0x3ff;
        let bits = if exp == 0 {
            if man == 0 {
                sign
            } else {   // subnormal: normalize
                let mut e = 0i32;
                let mut m = man;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (((127 - 15 + 1 + e) as u32) << 23) | ((m & 0x3ff) << 13)
            }
        } else if exp == 0x1f {
            sign | 0x7f80_0000 | (man << 13)
        } else {
            sign | ((exp + 127 - 15) << 23) | (man << 13)
        };
        f32::from_bits(bits)
    }
}

impl fmt::Debug for F16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}f16", self.to_f32())
    }
}

impl From<f32> for F16 {
    fn from(x: f32) -> F16 {
        F16::from_f32(x)
    }
}

impl From<F16> for f32 {
    fn from(x: F16) -> f32 {
        x.to_f32()
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for super::F16 {}
}

/// Element types the library computes with.
pub trait Element: sealed::Sealed + Copy + Default + Send + Sync + 'static {
    /// `GEN9BLAS_S`, `GEN9BLAS_D` or `GEN9BLAS_H`.
    const TYPE: c_int;
    /// Feature that must be present ([`has`]).
    const FEATURE: Feature;
}
impl Element for f32 {
    const TYPE: c_int = ffi::GEN9BLAS_S;
    const FEATURE: Feature = Feature::Fp32;
}
impl Element for f64 {
    const TYPE: c_int = ffi::GEN9BLAS_D;
    const FEATURE: Feature = Feature::Fp64;
}
impl Element for F16 {
    const TYPE: c_int = ffi::GEN9BLAS_H;
    const FEATURE: Feature = Feature::Fp16;
}

// ---------------------------------------------------------------------------------------------------------------------
// Options

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trans {
    No,
    Yes,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Uplo {
    Lower,
    Upper,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Diag {
    NonUnit,
    Unit,
}

impl Trans {
    fn c(self) -> c_int {
        match self { Trans::No => ffi::GEN9BLAS_NO_TRANS, Trans::Yes => ffi::GEN9BLAS_TRANS }
    }
    /// (rows, cols) of the stored matrix whose op() is `r` x `c`.
    fn stored(self, r: usize, c: usize) -> (usize, usize) {
        match self { Trans::No => (r, c), Trans::Yes => (c, r) }
    }
}
impl Side {
    fn c(self) -> c_int {
        match self { Side::Left => ffi::GEN9BLAS_LEFT, Side::Right => ffi::GEN9BLAS_RIGHT }
    }
}
impl Uplo {
    fn c(self) -> c_int {
        match self { Uplo::Lower => ffi::GEN9BLAS_LOWER, Uplo::Upper => ffi::GEN9BLAS_UPPER }
    }
}
impl Diag {
    fn c(self) -> c_int {
        match self { Diag::NonUnit => ffi::GEN9BLAS_NON_UNIT, Diag::Unit => ffi::GEN9BLAS_UNIT }
    }
}

/// Optional capabilities of the device / build (`gen9blas_has`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feature {
    Fp16,
    Fp32,
    Fp64,
    SkinnyPb,
    EpilogExt,
    NormPack,
    RopeMulti,
    Cpy,
    SetRows,
}

impl Feature {
    fn c(self) -> c_int {
        match self {
            Feature::Fp16 => ffi::GEN9BLAS_FEAT_FP16,
            Feature::Fp32 => ffi::GEN9BLAS_FEAT_FP32,
            Feature::Fp64 => ffi::GEN9BLAS_FEAT_FP64,
            Feature::SkinnyPb => ffi::GEN9BLAS_FEAT_SKINNY_PB,
            Feature::EpilogExt => ffi::GEN9BLAS_FEAT_EPILOG_EXT,
            Feature::NormPack => ffi::GEN9BLAS_FEAT_NORM_PACK,
            Feature::RopeMulti => ffi::GEN9BLAS_FEAT_ROPE_MULTI,
            Feature::Cpy => ffi::GEN9BLAS_FEAT_CPY,
            Feature::SetRows => ffi::GEN9BLAS_FEAT_SET_ROWS,
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Library

/// Version of the loaded library as major * 10000 + minor * 100 + patch.
pub fn version() -> i32 {
    unsafe { ffi::gen9blas_version() }
}

/// Initializes the GPU (once per process; also done implicitly by the first call that needs it).
/// Fails with [`Status::Unsupported`] if the loaded library is not 0.2.x.
pub fn init() -> Result<()> {
    let v = version();
    if v / 100 != ffi::GEN9BLAS_VERSION / 100 {
        return Err(Error {
            status: Status::Unsupported,
            message: format!("libgen9blas {v} loaded, these bindings are for {}", ffi::GEN9BLAS_VERSION),
        });
    }
    call(|| unsafe { ffi::gen9blas_init() })
}

/// GPU generation: 9 (Gen9 / Gen9.5), 8 (Gen8 Cherryview) or 0 (no device).
pub fn device_gen() -> i32 {
    let _g = lock();
    unsafe { ffi::gen9blas_device_gen() }
}

/// Whether a feature is available (initializes the library).
pub fn has(f: Feature) -> bool {
    let _g = lock();
    unsafe { ffi::gen9blas_has(f.c()) != 0 }
}

/// GEMM kernel time of the last call in ms with `GEN9BLAS_PROFILE=1`, else the time of the whole call.
pub fn last_time_ms() -> f64 {
    let _g = lock();
    unsafe { ffi::gen9blas_last_time_ms() }
}

// ---------------------------------------------------------------------------------------------------------------------
// Zero-copy memory

/// Host memory shared with the GPU (USM, 4096-byte aligned): kernels read and write it in place.
pub struct UsmBuffer<T: Element> {
    ptr: *mut T,
    len: usize,
    _t: PhantomData<T>,
}

unsafe impl<T: Element> Send for UsmBuffer<T> {}
unsafe impl<T: Element> Sync for UsmBuffer<T> {}

impl<T: Element> UsmBuffer<T> {
    /// `len` elements set to zero.
    pub fn zeroed(len: usize) -> Result<UsmBuffer<T>> {
        let bytes = len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| invalid("UsmBuffer: size overflow"))?;
        let _g = lock();
        let p = unsafe { ffi::gen9blas_alloc(bytes) } as *mut T;
        if p.is_null() {
            return Err(last_error(unsafe { ffi::gen9blas_last_status() }));
        }
        unsafe { std::ptr::write_bytes(p, 0, len) };   // all element types are valid as zero bits
        Ok(UsmBuffer { ptr: p, len, _t: PhantomData })
    }

    /// A copy of `data` in USM memory.
    pub fn from_slice(data: &[T]) -> Result<UsmBuffer<T>> {
        let mut b = UsmBuffer::zeroed(data.len())?;
        b.copy_from_slice(data);
        Ok(b)
    }
}

impl<T: Element> Deref for UsmBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: Element> DerefMut for UsmBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Element> Drop for UsmBuffer<T> {
    fn drop(&mut self) {
        let _g = lock();
        unsafe { ffi::gen9blas_free(self.ptr as *mut c_void) };
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Level 3

/// C = alpha·op(A)·op(B) + beta·C; op(A): m×k, op(B): k×n, C: m×n.
#[allow(clippy::too_many_arguments)]
pub fn gemm<T: Element>(ta: Trans, tb: Trans, m: usize, n: usize, k: usize, alpha: f64, a: &[T], lda: usize, b: &[T],
                        ldb: usize, beta: f64, c: &mut [T], ldc: usize) -> Result<()> {
    gemm_typed(T::TYPE, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

/// FP16 GEMM with FP32 accumulation of 32-product chunks (`GEN9BLAS_HS`): ~10x smaller error than [`gemm`] on F16.
#[allow(clippy::too_many_arguments)]
pub fn hgemm_accurate(ta: Trans, tb: Trans, m: usize, n: usize, k: usize, alpha: f64, a: &[F16], lda: usize,
                      b: &[F16], ldb: usize, beta: f64, c: &mut [F16], ldc: usize) -> Result<()> {
    gemm_typed(ffi::GEN9BLAS_HS, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

#[allow(clippy::too_many_arguments)]
fn gemm_typed<T: Element>(ty: c_int, ta: Trans, tb: Trans, m: usize, n: usize, k: usize, alpha: f64, a: &[T],
                          lda: usize, b: &[T], ldb: usize, beta: f64, c: &mut [T], ldc: usize) -> Result<()> {
    let (ar, ac) = ta.stored(m, k);
    let (br, bc) = tb.stored(k, n);
    mat(a, ar, ac, lda, "gemm A")?;
    mat(b, br, bc, ldb, "gemm B")?;
    mat(c, m, n, ldc, "gemm C")?;
    let (mi, ni, ki) = (int(m, "m")?, int(n, "n")?, int(k, "k")?);
    let (la, lb, lc) = (int(lda, "lda")?, int(ldb, "ldb")?, int(ldc, "ldc")?);
    call(|| unsafe {
        ffi::gen9blas_gemm(ty, ta.c(), tb.c(), mi, ni, ki, alpha, a.as_ptr().cast(), la, b.as_ptr().cast(), lb, beta,
                           c.as_mut_ptr().cast(), lc)
    })
}

/// Strided batched GEMM: for i in 0..count, matrices at A + i·sa, B + i·sb, C + i·sc (strides in elements).
#[allow(clippy::too_many_arguments)]
pub fn gemm_batched<T: Element>(ta: Trans, tb: Trans, m: usize, n: usize, k: usize, alpha: f64, a: &[T], lda: usize,
                                sa: usize, b: &[T], ldb: usize, sb: usize, beta: f64, c: &mut [T], ldc: usize,
                                sc: usize, count: usize) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    let (ar, ac) = ta.stored(m, k);
    let (br, bc) = tb.stored(k, n);
    let last = count - 1;
    let need = |s: usize, one: usize| last.checked_mul(s).and_then(|x| x.checked_add(one));
    let na = need(sa, span(ar, ac, lda, "gemm_batched A")?).ok_or_else(|| invalid("gemm_batched: A size overflow"))?;
    let nb = need(sb, span(br, bc, ldb, "gemm_batched B")?).ok_or_else(|| invalid("gemm_batched: B size overflow"))?;
    let nc = need(sc, span(m, n, ldc, "gemm_batched C")?).ok_or_else(|| invalid("gemm_batched: C size overflow"))?;
    check_len(a.len(), na, "gemm_batched A")?;
    check_len(b.len(), nb, "gemm_batched B")?;
    check_len(c.len(), nc, "gemm_batched C")?;
    let s = |x: usize| i64::try_from(x).map_err(|_| invalid("gemm_batched: stride too large"));
    let (sai, sbi, sci) = (s(sa)?, s(sb)?, s(sc)?);
    let (mi, ni, ki, cnt) = (int(m, "m")?, int(n, "n")?, int(k, "k")?, int(count, "count")?);
    let (la, lb, lc) = (int(lda, "lda")?, int(ldb, "ldb")?, int(ldc, "ldc")?);
    call(|| unsafe {
        ffi::gen9blas_gemm_batched(T::TYPE, ta.c(), tb.c(), mi, ni, ki, alpha, a.as_ptr().cast(), la, sai,
                                   b.as_ptr().cast(), lb, sbi, beta, c.as_mut_ptr().cast(), lc, sci, cnt)
    })
}

/// B packed once on the GPU (e.g. constant weights) and multiplied many times with [`PackedB::gemm`].
pub struct PackedB<T: Element> {
    p: *mut ffi::gen9blas_packed_b,
    k: usize,
    n: usize,
    _t: PhantomData<T>,
}

unsafe impl<T: Element> Send for PackedB<T> {}
unsafe impl<T: Element> Sync for PackedB<T> {}

impl<T: Element> PackedB<T> {
    /// Packs op(B) (k×n; `Trans::Yes`: B stored as n×k).
    pub fn new(tb: Trans, k: usize, n: usize, b: &[T], ldb: usize) -> Result<PackedB<T>> {
        Self::pack(T::TYPE, tb, k, n, b, ldb)
    }

    fn pack(ty: c_int, tb: Trans, k: usize, n: usize, b: &[T], ldb: usize) -> Result<PackedB<T>> {
        let (br, bc) = tb.stored(k, n);
        mat(b, br, bc, ldb, "pack_b B")?;
        let (ki, ni, lb) = (int(k, "k")?, int(n, "n")?, int(ldb, "ldb")?);
        let _g = lock();
        let p = unsafe { ffi::gen9blas_pack_b(ty, tb.c(), ki, ni, b.as_ptr().cast(), lb) };
        if p.is_null() {
            return Err(last_error(unsafe { ffi::gen9blas_last_status() }));
        }
        Ok(PackedB { p, k, n, _t: PhantomData })
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// C = alpha·op(A)·B + beta·C with op(A): m×k, C: m×n.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(&self, ta: Trans, m: usize, alpha: f64, a: &[T], lda: usize, beta: f64, c: &mut [T], ldc: usize)
                -> Result<()> {
        let (ar, ac) = ta.stored(m, self.k);
        mat(a, ar, ac, lda, "gemm_pb A")?;
        mat(c, m, self.n, ldc, "gemm_pb C")?;
        let (mi, la, lc) = (int(m, "m")?, int(lda, "lda")?, int(ldc, "ldc")?);
        call(|| unsafe {
            ffi::gen9blas_gemm_pb(ta.c(), mi, alpha, a.as_ptr().cast(), la, self.p, beta, c.as_mut_ptr().cast(), lc)
        })
    }
}

impl PackedB<F16> {
    /// FP16 B packed for FP32-chunk accumulation (`GEN9BLAS_HS`, as [`hgemm_accurate`]).
    pub fn new_accurate(tb: Trans, k: usize, n: usize, b: &[F16], ldb: usize) -> Result<PackedB<F16>> {
        Self::pack(ffi::GEN9BLAS_HS, tb, k, n, b, ldb)
    }
}

impl<T: Element> Drop for PackedB<T> {
    fn drop(&mut self) {
        let _g = lock();
        unsafe { ffi::gen9blas_packed_b_free(self.p) };   // waits for queued work that may still read it
    }
}

/// C = alpha·A·B + beta·C (`Side::Left`, A: m×m symmetric) or alpha·B·A + beta·C (`Side::Right`, A: n×n).
#[allow(clippy::too_many_arguments)]
pub fn symm<T: Element>(side: Side, uplo: Uplo, m: usize, n: usize, alpha: f64, a: &[T], lda: usize, b: &[T],
                        ldb: usize, beta: f64, c: &mut [T], ldc: usize) -> Result<()> {
    let na = if side == Side::Left { m } else { n };
    mat(a, na, na, lda, "symm A")?;
    mat(b, m, n, ldb, "symm B")?;
    mat(c, m, n, ldc, "symm C")?;
    let (mi, ni, la, lb, lc) = (int(m, "m")?, int(n, "n")?, int(lda, "lda")?, int(ldb, "ldb")?, int(ldc, "ldc")?);
    call(|| unsafe {
        ffi::gen9blas_symm(T::TYPE, side.c(), uplo.c(), mi, ni, alpha, a.as_ptr().cast(), la, b.as_ptr().cast(), lb,
                           beta, c.as_mut_ptr().cast(), lc)
    })
}

/// C = alpha·op(A)·op(A)ᵀ + beta·C (C: n×n; op(A) n×k). Only the `uplo` triangle of C is written.
#[allow(clippy::too_many_arguments)]
pub fn syrk<T: Element>(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: f64, a: &[T], lda: usize, beta: f64,
                        c: &mut [T], ldc: usize) -> Result<()> {
    let (ar, ac) = trans.stored(n, k);
    mat(a, ar, ac, lda, "syrk A")?;
    mat(c, n, n, ldc, "syrk C")?;
    let (ni, ki, la, lc) = (int(n, "n")?, int(k, "k")?, int(lda, "lda")?, int(ldc, "ldc")?);
    call(|| unsafe {
        ffi::gen9blas_syrk(T::TYPE, uplo.c(), trans.c(), ni, ki, alpha, a.as_ptr().cast(), la, beta,
                           c.as_mut_ptr().cast(), lc)
    })
}

/// C = alpha·(op(A)·op(B)ᵀ + op(B)·op(A)ᵀ) + beta·C. Only the `uplo` triangle of C is written.
#[allow(clippy::too_many_arguments)]
pub fn syr2k<T: Element>(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: f64, a: &[T], lda: usize, b: &[T],
                         ldb: usize, beta: f64, c: &mut [T], ldc: usize) -> Result<()> {
    let (ar, ac) = trans.stored(n, k);
    mat(a, ar, ac, lda, "syr2k A")?;
    mat(b, ar, ac, ldb, "syr2k B")?;
    mat(c, n, n, ldc, "syr2k C")?;
    let (ni, ki) = (int(n, "n")?, int(k, "k")?);
    let (la, lb, lc) = (int(lda, "lda")?, int(ldb, "ldb")?, int(ldc, "ldc")?);
    call(|| unsafe {
        ffi::gen9blas_syr2k(T::TYPE, uplo.c(), trans.c(), ni, ki, alpha, a.as_ptr().cast(), la, b.as_ptr().cast(), lb,
                            beta, c.as_mut_ptr().cast(), lc)
    })
}

/// B = alpha·op(A)·B (`Side::Left`, A: m×m) or alpha·B·op(A) (`Side::Right`, A: n×n); A triangular.
#[allow(clippy::too_many_arguments)]
pub fn trmm<T: Element>(side: Side, uplo: Uplo, ta: Trans, diag: Diag, m: usize, n: usize, alpha: f64, a: &[T],
                        lda: usize, b: &mut [T], ldb: usize) -> Result<()> {
    tr3(false, side, uplo, ta, diag, m, n, alpha, a, lda, b, ldb)
}

/// Solves op(A)·X = alpha·B (`Side::Left`) or X·op(A) = alpha·B (`Side::Right`); X overwrites B.
#[allow(clippy::too_many_arguments)]
pub fn trsm<T: Element>(side: Side, uplo: Uplo, ta: Trans, diag: Diag, m: usize, n: usize, alpha: f64, a: &[T],
                        lda: usize, b: &mut [T], ldb: usize) -> Result<()> {
    tr3(true, side, uplo, ta, diag, m, n, alpha, a, lda, b, ldb)
}

#[allow(clippy::too_many_arguments)]
fn tr3<T: Element>(solve: bool, side: Side, uplo: Uplo, ta: Trans, diag: Diag, m: usize, n: usize, alpha: f64,
                   a: &[T], lda: usize, b: &mut [T], ldb: usize) -> Result<()> {
    let na = if side == Side::Left { m } else { n };
    mat(a, na, na, lda, "triangular A")?;
    mat(b, m, n, ldb, "triangular B")?;
    let (mi, ni, la, lb) = (int(m, "m")?, int(n, "n")?, int(lda, "lda")?, int(ldb, "ldb")?);
    let f = if solve { ffi::gen9blas_trsm } else { ffi::gen9blas_trmm };
    call(|| unsafe {
        f(T::TYPE, side.c(), uplo.c(), ta.c(), diag.c(), mi, ni, alpha, a.as_ptr().cast(), la, b.as_mut_ptr().cast(),
          lb)
    })
}

// ---------------------------------------------------------------------------------------------------------------------
// Levels 1-2 (contiguous vectors; FP16 is computed in FP32, reductions in FP64)

fn same_len(x: usize, y: usize, what: &str) -> Result<c_int> {
    if x != y {
        return Err(invalid(format!("{what}: vectors of {x} and {y} elements")));
    }
    int(x, "n")
}

/// y = a·x + y
pub fn axpy<T: Element>(a: f64, x: &[T], y: &mut [T]) -> Result<()> {
    let n = same_len(x.len(), y.len(), "axpy")?;
    call(|| unsafe { ffi::gen9blas_axpy(T::TYPE, n, a, x.as_ptr().cast(), 1, y.as_mut_ptr().cast(), 1) })
}

/// x = a·x
pub fn scal<T: Element>(a: f64, x: &mut [T]) -> Result<()> {
    let n = int(x.len(), "n")?;
    call(|| unsafe { ffi::gen9blas_scal(T::TYPE, n, a, x.as_mut_ptr().cast(), 1) })
}

/// y = x
pub fn copy<T: Element>(x: &[T], y: &mut [T]) -> Result<()> {
    let n = same_len(x.len(), y.len(), "copy")?;
    call(|| unsafe { ffi::gen9blas_copy(T::TYPE, n, x.as_ptr().cast(), 1, y.as_mut_ptr().cast(), 1) })
}

/// x <-> y
pub fn swap<T: Element>(x: &mut [T], y: &mut [T]) -> Result<()> {
    let n = same_len(x.len(), y.len(), "swap")?;
    call(|| unsafe { ffi::gen9blas_swap(T::TYPE, n, x.as_mut_ptr().cast(), 1, y.as_mut_ptr().cast(), 1) })
}

/// x·y
pub fn dot<T: Element>(x: &[T], y: &[T]) -> Result<f64> {
    let n = same_len(x.len(), y.len(), "dot")?;
    let mut r = 0.0f64;
    call(|| unsafe { ffi::gen9blas_dot(T::TYPE, n, x.as_ptr().cast(), 1, y.as_ptr().cast(), 1, &mut r) })?;
    Ok(r)
}

/// Σ|x|
pub fn asum<T: Element>(x: &[T]) -> Result<f64> {
    let n = int(x.len(), "n")?;
    let mut r = 0.0f64;
    call(|| unsafe { ffi::gen9blas_asum(T::TYPE, n, x.as_ptr().cast(), 1, &mut r) })?;
    Ok(r)
}

/// ‖x‖₂
pub fn nrm2<T: Element>(x: &[T]) -> Result<f64> {
    let n = int(x.len(), "n")?;
    let mut r = 0.0f64;
    call(|| unsafe { ffi::gen9blas_nrm2(T::TYPE, n, x.as_ptr().cast(), 1, &mut r) })?;
    Ok(r)
}

/// Index of the first element with the largest |x| (None for an empty vector).
pub fn iamax<T: Element>(x: &[T]) -> Result<Option<usize>> {
    if x.is_empty() {
        return Ok(None);
    }
    let n = int(x.len(), "n")?;
    let mut r = 0i64;
    call(|| unsafe { ffi::gen9blas_iamax(T::TYPE, n, x.as_ptr().cast(), 1, &mut r) })?;
    Ok(Some(r as usize))
}

/// y = alpha·op(A)·x + beta·y, A: m×n (`Trans::No`: x has n elements, y m; `Trans::Yes`: x m, y n).
#[allow(clippy::too_many_arguments)]
pub fn gemv<T: Element>(trans: Trans, m: usize, n: usize, alpha: f64, a: &[T], lda: usize, x: &[T], beta: f64,
                        y: &mut [T]) -> Result<()> {
    mat(a, m, n, lda, "gemv A")?;
    let (nx, ny) = if trans == Trans::No { (n, m) } else { (m, n) };
    check_len(x.len(), nx, "gemv x")?;
    check_len(y.len(), ny, "gemv y")?;
    let (mi, ni, la) = (int(m, "m")?, int(n, "n")?, int(lda, "lda")?);
    call(|| unsafe {
        ffi::gen9blas_gemv(T::TYPE, trans.c(), mi, ni, alpha, a.as_ptr().cast(), la, x.as_ptr().cast(), 1, beta,
                           y.as_mut_ptr().cast(), 1)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip_and_rounding() {
        for &x in &[0.0f32, -0.0, 1.0, -2.5, 65504.0, 6.103_515_6e-5, 5.960_464_5e-8, 0.333_333_34] {
            let h = F16::from_f32(x);
            assert_eq!(F16::from_f32(h.to_f32()), h, "{x}");
        }
        assert_eq!(F16::from_f32(1.0).0, 0x3c00);
        assert_eq!(F16::from_f32(65520.0).0, 0x7c00);   // rounds to inf
        assert_eq!(F16::from_f32(1.0 + 1.0 / 2048.0).0, 0x3c00);   // tie -> even
        assert_eq!(F16::from_f32(1.0 + 3.0 / 2048.0).0, 0x3c02);   // tie -> even (up)
        assert!(F16::from_f32(f32::NAN).to_f32().is_nan());
    }

    #[test]
    fn f16_exhaustive_round_trip() {
        for h in 0..=u16::MAX {
            let x = F16(h).to_f32();
            if x.is_nan() { continue; }
            assert_eq!(F16::from_f32(x).0, h, "{h:#06x} -> {x}");
        }
    }

    #[test]
    fn shape_errors_before_the_gpu() {
        let a = vec![0f32; 10];
        let mut c = vec![0f32; 4];
        let e = gemm(Trans::No, Trans::No, 2, 2, 8, 1.0, &a, 8, &a, 2, 0.0, &mut c, 2).unwrap_err();
        assert_eq!(e.status, Status::InvalidArg);
        let e = gemm(Trans::No, Trans::No, 2, 2, 2, 1.0, &a, 1, &a, 2, 0.0, &mut c, 2).unwrap_err();
        assert_eq!(e.status, Status::InvalidArg);   // lda < k
        let mut y = vec![0f32; 3];
        assert_eq!(axpy(1.0, &a, &mut y).unwrap_err().status, Status::InvalidArg);
    }
}
