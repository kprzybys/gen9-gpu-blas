// Etap 2: szczytowa przepustowość arytmetyki.
// T = typ, OP(x) = operacja na łańcuchu. CH niezależnych łańcuchów × U powtórzeń w pętli n razy.
// ops na work-item na iterację = CH * U * OPS_PER (przekazywane do clbench).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifndef T
#define T float
#endif
#ifndef CH
#define CH 8
#endif

#define R8(s) s s s s s s s s
#define R16(s) R8(s) R8(s)

// Łańcuchy x0..x7 niezależne: pozwalają ukryć opóźnienie potoku.
#define DECL T x0 = (T)gid, x1 = x0 + (T)1, x2 = x0 + (T)2, x3 = x0 + (T)3, \
               x4 = x0 + (T)4, x5 = x0 + (T)5, x6 = x0 + (T)6, x7 = x0 + (T)7;
#define STEP8(F) x0 = F(x0); x1 = F(x1); x2 = F(x2); x3 = F(x3); x4 = F(x4); x5 = F(x5); x6 = F(x6); x7 = F(x7);
#define STEP1(F) x0 = F(x0);
#define SUM (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7)

#ifdef INT
#define AB T a = (T)(n | 3), b = (T)(n >> 4), av = a + (T)(gid & 7), bv = b + (T)(gid & 3);
#else
#define AB T a = (T)(n & 1) * (T)0.5 + (T)0.999, b = (T)(n & 2) * (T)0.25 + (T)0.001; \
           T av = a + (T)(gid & 7) * (T)0.0001, bv = b + (T)(gid & 3) * (T)0.0001;
#endif

#define KERNEL(name, F) \
__kernel void name(__global T *out, int n) {                \
    int gid = get_global_id(0);                              \
    AB                                                       \
    DECL                                                     \
    for (int i = 0; i < n; i++) { R16(STEP8(F)) }            \
    out[gid] = SUM;                                          \
}                                                            \
__kernel void name##_lat(__global T *out, int n) {          \
    int gid = get_global_id(0);                              \
    AB                                                       \
    DECL                                                     \
    for (int i = 0; i < n; i++) { R16(STEP1(F)) }            \
    out[gid] = SUM;                                          \
}

// Wersje z operandem wektorowym (a, b różne w każdej linii), a nie rozgłaszanym skalarem.
#define F_MULV(x) ((x) * av)
#define F_ADDV(x) ((x) + bv)
#define F_MADV(x) mad(x, av, bv)
#define F_MAD(x) mad(x, a, b)
#define F_FMA(x) fma(x, a, b)
#define F_MUL(x) ((x) * a)
#define F_ADD(x) ((x) + b)
#define F_IMAD(x) ((x) * a + b)

#ifndef INT
KERNEL(k_mad, F_MAD)
KERNEL(k_fma, F_FMA)
#endif
KERNEL(k_mul, F_MUL)
KERNEL(k_add, F_ADD)
KERNEL(k_imad, F_IMAD)
KERNEL(k_mulv, F_MULV)
KERNEL(k_addv, F_ADDV)
#ifndef INT
KERNEL(k_madv, F_MADV)
#endif
