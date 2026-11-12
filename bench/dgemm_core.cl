// DGEMM (FP64) dla libgen9blas: pakowanie i główny kernel. Reszta kerneli FP64 (epilogi, TRSM, rozwijanie, sklejanie)
// powstaje z sgemm.cl przez podstawienie typów -- lib/gen_dgemm.py składa z tego bench/dgemm.cl.
// Układ panelowy (A: r = wiersz m, B: r = kolumna n): P[(r/16)·Kp·16 + k·16 + r%16] -- w panelu 16 wierszy element
// (k, r) pod k·16 + r. Przesunięcia używane przez bibliotekę (fragment od wiersza i0 i kolumny k0: i0·Kp + k0·16
// dla A, k0·16 dla B) są takie same jak w układzie asm FP32.
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define GMD 16
#define PD(r, k, Kp) ((size_t)((r) / 16) * (Kp) * 16 + (size_t)(k) * 16 + (r) % 16)
__kernel void pkd_a(__global const double *A, __global double *Ap, int M, int K, int lda, int Kp, long sA, int Mp) {
    const int k = get_global_id(0), m = get_global_id(1), b = get_global_id(2);
    Ap[(size_t)b * Mp * Kp + PD(m, k, Kp)] = (m < M && k < K) ? A[b * sA + (size_t)m * lda + k] : 0.0; }
__kernel void pkd_a_t(__global const double *A, __global double *Ap, int M, int K, int lda, int Kp, long sA, int Mp) {
    const int m = get_global_id(0), k = get_global_id(1), b = get_global_id(2);
    Ap[(size_t)b * Mp * Kp + PD(m, k, Kp)] = (m < M && k < K) ? A[b * sA + (size_t)k * lda + m] : 0.0; }
__kernel void pkd_b(__global const double *B, __global double *Bp, int K, int N, int ldb, int Kp, long sB, int Np) {
    const int n = get_global_id(0), k = get_global_id(1), b = get_global_id(2);
    Bp[(size_t)b * Np * Kp + PD(n, k, Kp)] = (n < N && k < K) ? B[b * sB + (size_t)k * ldb + n] : 0.0; }
__kernel void pkd_b_t(__global const double *B, __global double *Bp, int K, int N, int ldb, int Kp, long sB, int Np) {
    const int k = get_global_id(0), n = get_global_id(1), b = get_global_id(2);
    Bp[(size_t)b * Np * Kp + PD(n, k, Kp)] = (n < N && k < K) ? B[b * sB + (size_t)n * ldb + k] : 0.0; }

// Główny kernel, sygnatura jak asm „v”: (Ap, Bp, C, M, N, ldc, K, Kfull); M, N wielokrotności 16, K -- liczba kroków
// k od początku fragmentu, Kfull -- długość panelu. Grupa robocza 16 = kafelek C 16×16 z dwóch subgrup po 8 (natywna
// szerokość FP64 na Gen9): subgrupa h liczy wiersze 8h..8h+7, lane l kolumny l i l+8 -- jedno rozgłoszenie A na dwa
// FMA (wersja z subgrupą 16 i jednym FMA na rozgłoszenie: 26 GFLOPS). Kolejność kafelków: grupy GMD wierszy kafelków.
__attribute__((intel_reqd_sub_group_size(8)))
__kernel __attribute__((reqd_work_group_size(16, 1, 1)))
void dgemm_v(__global const double *Ap, __global const double *Bp, __global double *C, int M, int N, int ldc, int K, int Kfull) {
    const int pid0 = get_group_id(0);
    const int npm = M / 16, npn = N / 16;
    const int b = pid0 / (npm * npn), pid = pid0 % (npm * npn);
    const int grupa = pid / (GMD * npn), m_pierwszy = grupa * GMD, gsz = min(npm - m_pierwszy, GMD);
    const int pm = m_pierwszy + (pid % (GMD * npn)) % gsz, pn = (pid % (GMD * npn)) / gsz;
    const int l = get_sub_group_local_id(), h = get_sub_group_id();
    const __global double *pa = Ap + (size_t)b * M * Kfull + (size_t)pm * Kfull * 16 + 8 * h + l;
    const __global double *pb = Bp + (size_t)b * N * Kfull + (size_t)pn * Kfull * 16 + l;
    double acc0[8], acc1[8];
#pragma unroll
    for (int r = 0; r < 8; r++) { acc0[r] = 0.0; acc1[r] = 0.0; }
    for (int k = 0; k < K; k += 4) {
        double a[4], b0[4], b1[4];
#pragma unroll
        for (int q = 0; q < 4; q++) { a[q] = pa[(size_t)(k + q) * 16]; b0[q] = pb[(size_t)(k + q) * 16]; b1[q] = pb[(size_t)(k + q) * 16 + 8]; }
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const double ar = sub_group_broadcast(a[q], r);
                acc0[r] = fma(ar, b0[q], acc0[r]); acc1[r] = fma(ar, b1[q], acc1[r]);
            }
    }
    __global double *c = C + (size_t)b * M * ldc + (size_t)(pm * 16 + 8 * h) * ldc + pn * 16 + l;
#pragma unroll
    for (int r = 0; r < 8; r++) { c[(size_t)r * ldc] = acc0[r]; c[(size_t)r * ldc + 8] = acc1[r]; }
}
