// libgen9blas: BLAS (poziomy 1–3) na Intel UHD 620 (Gen9.5) z ręcznymi pętlami GEMM w asemblerze. Zob. lib/README.md.
// Wszystkie macierze row-major. Elementy: uint16_t (FP16) albo float (FP32), wskaźniki jako void*.
// Wynik 0 = sukces, -1 = błąd (opis: gen9blas_error()).
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

enum { GEN9BLAS_H = 0,    // FP16, sumowanie FP16 (najszybszy)
       GEN9BLAS_S = 1,    // FP32
       GEN9BLAS_HS = 2,   // FP16, sumowanie porcjami po GEN9BLAS_KC iloczynów do FP32 (dokładniejszy)
       GEN9BLAS_D = 3 };  // FP64
#define GEN9BLAS_KC 32
enum { GEN9BLAS_LEFT = 0, GEN9BLAS_RIGHT = 1 };          // side
enum { GEN9BLAS_LOWER = 0, GEN9BLAS_UPPER = 1 };         // uplo
                                               // trans / transA: 0 = bez, 1 = transpozycja; diag: 0 = zwykła, 1 = jedynki

int gen9blas_init(const char *katalog);             // katalog z gen_h.bin, gen_hd.bin, gen_s.bin, hgemm.cl, sgemm.cl, dgemm.cl, level12.cl

// Pamięć bez kopii (USM host, wyrównana do 4096 B): wskaźniki z gen9blas_alloc (także na wnętrze przydziału) GPU czyta
// i pisze bezpośrednio. Inna pamięć hosta jest kopiowana przez CPU w każdym wywołaniu. Po gen9blas_init.
void *gen9blas_alloc(size_t bajty);
void gen9blas_free(void *p);
int gen9blas_is_usm(const void *p, size_t bajty);   // 1: [p, p+bajty) w jednym przydziale z gen9blas_alloc

// GEMM: C = alpha·op(A)·op(B) + beta·C; op(A): M×K (transA: A zapisana K×M), op(B): K×N (transB: B zapisana N×K).
int gen9blas_gemm(int rodzaj, int transA, int transB, int M, int N, int K, double alpha, const void *A, int lda,
             const void *B, int ldb, double beta, void *C, int ldc);
// GEMM wsadowy (strided batched): dla b = 0..count-1 macierze A + b·sA, B + b·sB, C + b·sC (odstępy w elementach).
// Jedno uruchomienie kernela dla całej paczki.
int gen9blas_gemm_batched(int rodzaj, int transA, int transB, int M, int N, int K, double alpha,
                     const void *A, int lda, long sA, const void *B, int ldb, long sB,
                     double beta, void *C, int ldc, long sC, int count);
// SYMM: C = alpha·A·B + beta·C (lewa, A: M×M symetryczna) albo alpha·B·A + beta·C (prawa, A: N×N). Używany trójkąt uplo.
int gen9blas_symm(int rodzaj, int side, int uplo, int M, int N, double alpha, const void *A, int lda,
             const void *B, int ldb, double beta, void *C, int ldc);
// SYRK: C = alpha·op(A)·op(A)ᵀ + beta·C (C: N×N; op(A) = A: N×K albo Aᵀ, A: K×N). Zmieniany tylko trójkąt uplo.
int gen9blas_syrk(int rodzaj, int uplo, int trans, int N, int K, double alpha, const void *A, int lda, double beta, void *C, int ldc);
// SYR2K: C = alpha·(op(A)·op(B)ᵀ + op(B)·op(A)ᵀ) + beta·C. Zmieniany tylko trójkąt uplo.
int gen9blas_syr2k(int rodzaj, int uplo, int trans, int N, int K, double alpha, const void *A, int lda,
              const void *B, int ldb, double beta, void *C, int ldc);
// TRMM: B = alpha·op(A)·B (lewa, A: M×M) albo alpha·B·op(A) (prawa, A: N×N); A trójkątna uplo, diag = 1: jedynki.
int gen9blas_trmm(int rodzaj, int side, int uplo, int transA, int diag, int M, int N, double alpha,
             const void *A, int lda, void *B, int ldb);
// TRSM: rozwiązanie op(A)·X = alpha·B (lewa) albo X·op(A) = alpha·B (prawa); X nadpisuje B. FP32, FP64 i FP16 (GEN9BLAS_H, GEN9BLAS_HS).
int gen9blas_trsm(int rodzaj, int side, int uplo, int transA, int diag, int M, int N, double alpha,
             const void *A, int lda, void *B, int ldb);

// Skróty bez transpozycji.
int gen9blas_hgemm(int M, int N, int K, double alpha, const uint16_t *A, int lda, const uint16_t *B, int ldb,
              double beta, uint16_t *C, int ldc);
int gen9blas_sgemm(int M, int N, int K, double alpha, const float *A, int lda, const float *B, int ldb,
              double beta, float *C, int ldc);

// ---- Poziomy 1–2 (rodzaj: GEN9BLAS_S, GEN9BLAS_D albo GEN9BLAS_H; GEN9BLAS_HS = GEN9BLAS_H). FP16: liczenie w FP32, redukcje w FP64. ----
// Wektory: n elementów co inc (inc != 0; inc < 0 jak w BLAS: wskaźnik na najniższy adres).
int gen9blas_axpy(int rodzaj, int n, double a, const void *x, int incx, void *y, int incy);    // y = a·x + y
int gen9blas_scal(int rodzaj, int n, double a, void *x, int incx);                             // x = a·x
int gen9blas_copy(int rodzaj, int n, const void *x, int incx, void *y, int incy);             // y = x
int gen9blas_swap(int rodzaj, int n, void *x, int incx, void *y, int incy);                   // x <-> y
int gen9blas_rot(int rodzaj, int n, void *x, int incx, void *y, int incy, double c, double s);  // obrót Givensa
int gen9blas_rotm(int rodzaj, int n, void *x, int incx, void *y, int incy, const double *param);  // zmodyfikowany obrót
void gen9blas_rotg(float *a, float *b, float *c, float *s);                  // CPU: parametry obrotu Givensa (a = r, b = z)
void gen9blas_rotmg(float *d1, float *d2, float *x1, float y1, float *param); // CPU: parametry zmodyfikowanego obrotu
int gen9blas_dot(int rodzaj, int n, const void *x, int incx, const void *y, int incy, double *wynik);
int gen9blas_asum(int rodzaj, int n, const void *x, int incx, double *wynik);                // Σ|x|
int gen9blas_nrm2(int rodzaj, int n, const void *x, int incx, double *wynik);                // ‖x‖₂
int gen9blas_iamax(int rodzaj, int n, const void *x, int incx, long *wynik);                 // indeks od 0 (pierwszy max |x|)
// GEMV: y = alpha·op(A)·x + beta·y, A: M×N (trans: y długości N, x długości M). beta = 0: y nie jest czytany.
int gen9blas_gemv(int rodzaj, int trans, int M, int N, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy);
// SYMV: y = alpha·A·x + beta·y, A symetryczna n×n (używany trójkąt uplo).
int gen9blas_symv(int rodzaj, int uplo, int n, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy);
// TRMV: x = op(A)·x; TRSV: rozwiązanie op(A)·x = b (b podane w x). A trójkątna n×n (uplo; diag = 1: jedynki).
int gen9blas_trmv(int rodzaj, int uplo, int trans, int diag, int n, const void *A, int lda, void *x, int incx);
int gen9blas_trsv(int rodzaj, int uplo, int trans, int diag, int n, const void *A, int lda, void *x, int incx);
// SYR: A = alpha·x·xᵀ + A; SYR2: A = alpha·(x·yᵀ + y·xᵀ) + A. Zmieniany tylko trójkąt uplo.
int gen9blas_syr(int rodzaj, int uplo, int n, double alpha, const void *x, int incx, void *A, int lda);
int gen9blas_syr2(int rodzaj, int uplo, int n, double alpha, const void *x, int incx, const void *y, int incy, void *A, int lda);
// GER: A = alpha·x·yᵀ + A, A: M×N.
int gen9blas_ger(int rodzaj, int M, int N, double alpha, const void *x, int incx, const void *y, int incy, void *A, int lda);

// ---- Poziom 2: macierze pasmowe i spakowane, układ wierszowy jak CBLAS RowMajor ----
// Pasmo: A(i,j) = A[i·lda + KL + j − i] dla −KL ≤ j − i ≤ KU (lda ≥ KL + KU + 1); symetryczna/trójkątna pasmowa:
// górna KL = 0, KU = K, dolna KL = K, KU = 0. Spakowana górna: wiersz i (j = i..n−1) od i·n − i(i−1)/2; dolna:
// wiersz i (j = 0..i) od i(i+1)/2. TBSV/TPSV liczy jedna grupa robocza wiersz po wierszu (sekwencyjne z natury).
int gen9blas_gbmv(int rodzaj, int trans, int M, int N, int KL, int KU, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy);
int gen9blas_sbmv(int rodzaj, int uplo, int N, int K, double alpha, const void *A, int lda, const void *x, int incx, double beta, void *y, int incy);
int gen9blas_tbmv(int rodzaj, int uplo, int trans, int diag, int N, int K, const void *A, int lda, void *x, int incx);
int gen9blas_tbsv(int rodzaj, int uplo, int trans, int diag, int N, int K, const void *A, int lda, void *x, int incx);
int gen9blas_spmv(int rodzaj, int uplo, int N, double alpha, const void *Ap, const void *x, int incx, double beta, void *y, int incy);
int gen9blas_tpmv(int rodzaj, int uplo, int trans, int diag, int N, const void *Ap, void *x, int incx);
int gen9blas_tpsv(int rodzaj, int uplo, int trans, int diag, int N, const void *Ap, void *x, int incx);
int gen9blas_spr(int rodzaj, int uplo, int N, double alpha, const void *x, int incx, void *Ap);
int gen9blas_spr2(int rodzaj, int uplo, int N, double alpha, const void *x, int incx, const void *y, int incy, void *Ap);

int gen9blas_available(int rodzaj);         // 1, gdy kernel danego rodzaju został wczytany
double gen9blas_last_time_ms(void);     // GEN9BLAS_PROFILE=1: czas kerneli GEMM ostatniego wywołania; bez -- czas całego wywołania
int gen9blas_last_zero_copy(void);      // GEMM: bity 1 = A, 2 = B, 4 = C bez kopii (pamięć z gen9blas_alloc)
const char *gen9blas_error(void);

#ifdef __cplusplus
}
#endif
