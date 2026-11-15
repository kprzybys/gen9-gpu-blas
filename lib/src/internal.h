// internal.h: state and helpers shared by the library's translation units (internal, not installed).
// Split verbatim from lib/gen9blas.c (refactor stage 2b); everything here has hidden visibility.
#ifndef GEN9BLAS_INTERNAL_H
#define GEN9BLAS_INTERNAL_H
#include "gen9blas_ml.h"
#ifdef GEN9BLAS_NEO   // old NEO (OpenCL) path -- for comparisons only (libgen9blas_neo.so)
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>
#else            // default: kcl -- an OpenCL subset directly on i915 (kcl.c, docs/no-neo.md)
#include "kcl.h"
#endif
#include <limits.h>
#include <pthread.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef struct {
    cl_kernel main, pa, pb, pa_t, pb_t, pa_q, pb_w2,   // pa_q, pb_w2: faster packing without transpose (NULL: absent)
              epi, epi_tri, tL, tY, tZ, tI[4], expand, syr2k_merge;   // t*: TRSM; expand: full matrix; syr2k_merge: SYR2K join
    int es, tn, km, ready;
    int sg;   // work-items per C tile in the main kernel: its subgroup size (16; Gen8 sgemm8_v32: 8)
} type_t;
#define CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { snprintf(errbuf, sizeof errbuf, "%s:%d: %s = %d", __FILE__, __LINE__, #x, e_); return -1; } } while (0)
#define FAIL(...) do { snprintf(errbuf, sizeof errbuf, __VA_ARGS__); return -1; } while (0)
#define KER(pr, name, dst) do { cl_int e2_; dst = clCreateKernel(pr, name, &e2_); CHECK(e2_); } while (0)
typedef struct { cl_kernel axpy, scal, copy, copy16, rot, red, iamax, gemv_n, gemv_nw, gemv_t1, gemv_t1b, gemv_t2, ger, syr2, to_f, from_f, trsv_block, trsv_upd_n, trsv_upd_t, rotm, pmv, psv, spr2, symv1, symv2, skinny_k1, skinny_k4, skinny_k8, skinny_r4, skinny_r8, skinny_n4, skinny_n2; int es; } l12_t;
typedef struct { cl_mem m; void *p; } bf_t;
#define BM(x) ((bf_t){ (x), NULL })
#define ARGB(k, i, b) CHECK(set_arg(k, i, b))
#ifndef GEN9BLAS_NEO
#define NEW_OPERATION() kcl_new_operation()
#else
#define NEW_OPERATION() ((void)0)
#endif
typedef struct { void *p; size_t n; } work_t;
#define ARG(k, i, v) CHECK(clSetKernelArg(k, i, sizeof(v), &(v)))
#define ARGS(k, i, v) CHECK(arg_scalar(r, k, i, v))
struct gen9blas_packed_b { int type, K, N, Np, Kp; cl_mem buf; size_t bufsize; };
#define SPARES 4
typedef struct { const float *bias; int gelu; const float *R; int ldr; } epi_ext_t;
struct gen9blas_packed_a { cl_mem buf; size_t cap; int M, K, Kp; };
#define CHECK_TYPE(type) do { NEW_OPERATION(); if ((type) < 0 || (type) > 3 || !types[type].ready) FAIL("type %d unavailable", type); } while (0)
#define GROUPS 96         // work-groups (of 256) in reductions: 24 EUs × 4
#define ARGR(K, i, v) CHECK(argr(K, i, v, k->es))
#define RS(k) ((k)->es == 8 ? sizeof(double) : sizeof(float))   // size of the compute type
typedef struct { const void *p; size_t b; bf_t g; int zc; } op_t;
#define BEGIN(n, ...) l12_t *k; int es; if (l12_type(type, &k, &es)) return -1; if ((n) <= 0) return 0; \
                     if (__VA_ARGS__) FAIL("vector increment 0")
#define SKINNY_S 16   // subgroups per group in skinny_r (like SKINNY_S in level12.cl)
#pragma GCC visibility push(hidden)
double now_s(void);
extern cl_context ctx;
extern cl_command_queue q;
extern type_t types[4];
extern cl_mem dAp, dBp, dCp;
extern size_t sAp, sBp, sCp;
extern double last_ms;
extern int profile;
extern int last_zc;
extern char errbuf[1200];
extern clMemBlockingFreeINTEL_fn usm_free;
extern clSetKernelArgMemPointerINTEL_fn usm_arg;
extern l12_t L12[3];
extern int gen8;
extern cl_kernel k_rope_m, k_setrows, k_cpy_any, k_upscale, k_dec_part, k_dec_comb;
extern cl_kernel k_skp[3];
extern cl_kernel k_pa3_f32, k_epi_hf, k_swiglu, k_dequant, k_dequant2, k_pa3_v, k_epi_8, k_epi_8x;
extern cl_kernel k_rms, k_binop, k_rope, k_lnorm, k_gelu, k_lnorm_pack, k_gelu_erf, k_cvt_heads, k_im2col, k_relu;
int wait_gpu(void);
extern cl_event evs[256];
extern int nev;
double event_times(void);
int ensure_buf(cl_mem *b, size_t *size, size_t needed);
cl_int set_arg(cl_kernel k, int i, bf_t b);
extern work_t uA, uB, uC;
extern int force_copy;
int get_obj(const void *ptr, size_t bytes, work_t *u, int copy_in, int writes, bf_t *result, int *zc);
int get_obj_zc(const void *ptr, size_t bytes, int writes, bf_t *result);
cl_int arg_scalar(const type_t *r, cl_kernel k, int i, double v);
int to_host(bf_t x, int zc, size_t bytes, void *host);
cl_kernel pack_kernel(type_t *r, int roleB, int trans, size_t Rp, size_t Kp, size_t cnt, size_t g[3]);
int pack_dev(type_t *r, int roleB, int trans, bf_t xX, int ld, int Rr, int K, int Rp, int Kp, cl_mem target,
                     int k0, int kn);
int pack_host(type_t *r, int roleB, int trans, const void *X, int ld, int Rr, int K, int Rp, int Kp, cl_mem target);
cl_mem subbuffer(cl_mem b, size_t off, size_t size);
int epilogue_dev(type_t *r, cl_mem Cp, int ldp, cl_mem Cd, int ldc, int M, int N, double alpha, double beta);
double max_launch_ms(void);
double est_ms(type_t *r, double M, double N, double K);
int launch(type_t *r, cl_mem Ap, size_t oA, cl_mem Bp, size_t oB, cl_mem Cp, size_t oC,
                   int M, int N, int ldc, int K, int Kfull, int measure);
int epilogue_host(type_t *r, cl_mem Cp, int ldp, void *C, int M, int N, int ldc, double alpha, double beta, int uplo);
int l12_type(int type, l12_t **k, int *es);
cl_int argr(cl_kernel K, int i, double v, int es);
extern cl_mem dQ;
extern size_t sQ;
int skinny_threshold(int type);
int gemm_skinny(int type, int tA, int tB, int M, int N, int K, double alpha, const void *A, int lda,
                      const void *B, int ldb, double beta, void *C, int ldc);
#pragma GCC visibility pop
#endif
