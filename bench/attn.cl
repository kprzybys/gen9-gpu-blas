// Fused attention (flash-attention style) for Gen9, stage 2 of docs/attention-gpu.md.
// Layouts as llama.cpp flash_attn_ext gives them:
//   Q  float  [H][n_q][D]       (d fastest)
//   K  half   [n_k][HKV][D]     (KV cache row per token)
//   V  half   [n_k][HKV][D]     (not transposed)
//   O  float  [n_q][H][D]
// Query head h uses KV head h / G. Non-causal, no mask.
//
// Build-time parameters: D (64 or 128), G (= H / HKV), HKV, RQ (query rows per subgroup), NK (8-key vectors per
// key block), SH (1: variant B, Q in FP16 and Q*K^T as FP16 partial sums over 8 dims), PF/PD (1: register prefetch
// of K/V PD steps ahead), ADV (0: diagnostic, K/V pointers do not advance; wrong results).
// Results and reading: results/attn-kernel-hd530-2026-11-13.txt, docs/attention-gpu.md section 8.
//
// One SIMD8 subgroup owns RQ query rows of one KV head (rows ordered token-major: row = t * G + head-in-group,
// so the RQ rows of a subgroup are query heads of the same KV head and K/V are read once for all of them).
// No SLM: K^T and V are read with subgroup block reads (128 B messages) through L3.
//   S = Q * K^T: lane = key (8 keys per vector); K^T pre-transposed by k_kt into [HKV][n_k/8][D][8] half, so a
//                block read gives packed halfs that fold into a mixed-mode mad; q[r][d] (FP32) is broadcast from
//                the lane-split Q registers (constant-index sub_group_broadcast = scalar register region).
//   O += P * V:  lane l owns dims l + 8c (c < D/8) of every row; V row read as one or two blocks of 64 halfs;
//                p[r][j] is broadcast from the lane holding key j.
// Softmax: online (running max per row, uniform across the subgroup; lane-partial sums), exp2 with the scale
// times log2(e) folded into Q; the O rescale runs only when the row maximum grew.
// Registers: Q and O take 16 GRF per row each at D = 128, so RQ = 2 (D = 128) / RQ = 4 (D = 64) are the largest
// spill-free choices (IGC 24.35).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef SH
#define SH 0
#endif

#if D == 128
typedef float16 OV;
#elif D == 64
typedef float8 OV;
#else
#error D must be 64 or 128
#endif
#if SH && D == 128
typedef half16 QV;
#elif SH
typedef half8 QV;
#elif D == 128
typedef float16 QV;
#else
typedef float8 QV;
#endif

#if NK == 1
typedef float SV;
#define SK(v, k) (v)
#elif NK == 2
typedef float2 SV;
#define SK(v, k) (v)[k]
#elif NK == 4
typedef float4 SV;
#define SK(v, k) (v)[k]
#endif

#if RQ == 1
#define ROWS(X) X(0)
#elif RQ == 2
#define ROWS(X) X(0) X(1)
#elif RQ == 3
#define ROWS(X) X(0) X(1) X(2)
#elif RQ == 4
#define ROWS(X) X(0) X(1) X(2) X(3)
#elif RQ == 6
#define ROWS(X) X(0) X(1) X(2) X(3) X(4) X(5)
#elif RQ == 8
#define ROWS(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
#endif

// K (half [n_k][HKV][D]) -> K^T [HKV][n_k/8][D][8] half (8 keys of one dim packed). One work-item per half.
__kernel void k_kt(__global const ushort *K, __global ushort *Kt, int nk) {
    int i = get_global_id(0);                 // i = (t * HKV + hkv) * D + d  (coalesced reads)
    int d = i % D, th = i / D, hkv = th % HKV, t = th / HKV;
    Kt[((hkv * (nk / 8) + t / 8) * D + d) * 8 + (t & 7)] = K[i];
}

#define DECL(r) QV q##r; OV o##r = 0; float m##r = -1e30f; float l##r = 0.f; SV s##r;
// Lane l holds dims l + 8c (c < D/8) of Q (scaled by scale * log2 e) and O.
#define LOADQ(r) { int row = row0 + r, t = row / G, hh = row - t * G; \
    const __global float *qp = Q + ((hkv * G + hh) * nq + t) * D; \
    _Pragma("unroll") for (int c = 0; c < D / 8; c++) q##r[c] = qp[lane + 8 * c] * qscale; }
// Variant B (SH = 1): Q in FP16, Q * K^T as FP16 multiply-adds; FP16 partial sum over the 8 dims of a chunk,
// added to the FP32 score once per chunk.
#define FMAHK(r, k, kvk) { half hs = 0; _Pragma("unroll") for (int i = 0; i < 8; i++) \
    hs = fma(intel_sub_group_shuffle(q##r[c], i), as_half(kvk[i]), hs); SK(s##r, k) += (float)hs; }
#define ZS(r) s##r = 0;
// S chunk: dims 8c + i, i < 8; kvk[i] = K[key = lane][8c + i]; q[r][8c + i] sits in lane i of q[r][c].
#define FMAK(r, k, kvk) SK(s##r, k) = fma(sub_group_broadcast(q##r[c], i), (float)as_half(kvk[i]), SK(s##r, k));
#define FMA0(r) FMAK(r, 0, kv0)
#define FMA1(r) FMAK(r, 1, kv1)
#define FMA2(r) FMAK(r, 2, kv2)
#define FMA3(r) FMAK(r, 3, kv3)

#if NK == 1
#define SMAX(v) (v)
#define SSUM(v) (v)
#elif NK == 2
#define SMAX(v) fmax((v).s0, (v).s1)
#define SSUM(v) ((v).s0 + (v).s1)
#else
#define SMAX(v) fmax(fmax((v).s0, (v).s1), fmax((v).s2, (v).s3))
#define SSUM(v) (((v).s0 + (v).s1) + ((v).s2 + (v).s3))
#endif
// Online softmax for row r: new running max (uniform), rescale O and the partial sum only if it grew.
#define SOFT(r) { float mn = sub_group_reduce_max(SMAX(s##r)); \
    if (mn > m##r) { float a = native_exp2(m##r - mn); o##r *= a; l##r *= a; m##r = mn; } \
    s##r = native_exp2(s##r - m##r); l##r += SSUM(s##r); }

// P * V for key j of key vector k: v0/v1 hold dims l + 8c of V row (c < 8 / c >= 8).
#define PVK(r, k) { float p = sub_group_broadcast(SK(s##r, k), j); \
    _Pragma("unroll") for (int c = 0; c < 8; c++) o##r[c] = fma(p, (float)as_half(v0[c]), o##r[c]); \
    if (D == 128) { _Pragma("unroll") for (int c = 0; c < 8; c++) o##r[8 + c] = fma(p, (float)as_half(v1[c]), o##r[8 + c]); } }
#define PV0(r) PVK(r, 0)
#define PV1(r) PVK(r, 1)
#define PV2(r) PVK(r, 2)
#define PV3(r) PVK(r, 3)

#define STORE(r) { float il = 1.f / sub_group_reduce_add(l##r); int row = row0 + r, t = row / G, hh = row - t * G; \
    __global float *op = O + (t * (HKV * G) + hkv * G + hh) * D; \
    _Pragma("unroll") for (int c = 0; c < D / 8; c++) op[lane + 8 * c] = o##r[c] * il; }

#if D == 128
#define READV ushort8 v0 = intel_sub_group_block_read_us8(vp), v1 = intel_sub_group_block_read_us8(vp + 64);
#else
#define READV ushort8 v0 = intel_sub_group_block_read_us8(vp), v1;
#endif
#define PVLOOP(k) _Pragma("unroll") for (int j = 0; j < 8; j++) { READV vp += ADV * HKV * D; ROWS(PV##k) }

// Register prefetch (PF = 1): K chunks and V rows are loaded PD steps before use from a 4-slot ring
// (D/8 and 8 * NK are multiples of 4, so chunk c and key j always map to slot c % 4 / j % 4).
// ADV = 0 (diagnostic only, wrong results): K/V pointers do not advance, every block re-reads the first one.
#ifndef ADV
#define ADV 1
#endif
#ifndef PF
#define PF 0
#endif
#ifndef PD
#define PD 2
#endif
#define NCH (D / 8)
#define KLD(slot, ptr, cc) _Pragma("unroll") for (int k = 0; k < NK; k++) \
    kr[(slot) * NK + k] = intel_sub_group_block_read_us8((ptr) + k * D * 8 + (cc) * 64);
#if D == 128
#define VLD(slot, ptr) { vr0[slot] = intel_sub_group_block_read_us8(ptr); vr1[slot] = intel_sub_group_block_read_us8((ptr) + 64); }
#else
#define VLD(slot, ptr) { vr0[slot] = intel_sub_group_block_read_us8(ptr); }
#endif
#define FMAR(r) _Pragma("unroll") for (int k = 0; k < NK; k++) \
    SK(s##r, k) = fma(sub_group_broadcast(q##r[c], i), (float)as_half(kr[(c % 4) * NK + k][i]), SK(s##r, k));
#define PVR(r) { float p = sub_group_broadcast(SK(s##r, jj / 8), jj % 8); \
    _Pragma("unroll") for (int c = 0; c < 8; c++) o##r[c] = fma(p, (float)as_half(vr0[jj % 4][c]), o##r[c]); \
    if (D == 128) { _Pragma("unroll") for (int c = 0; c < 8; c++) o##r[8 + c] = fma(p, (float)as_half(vr1[jj % 4][c]), o##r[8 + c]); } }

// Global size: HKV * (nq * G / RQ) subgroups of 8. Subgroup s works on KV head s / (nq * G / RQ).
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_attn(__global const float *Q, __global const ushort *Kt, __global const ushort *V, __global float *O,
                     int nq, int nk, float qscale) {
    int sg = get_group_id(0) * get_num_sub_groups() + get_sub_group_id(), lane = get_sub_group_local_id();   // uniform
    int per_head = nq * G / RQ;
    int hkv = sg / per_head, row0 = (sg - hkv * per_head) * RQ;
    ROWS(DECL)
    ROWS(LOADQ)
    const __global ushort *kp = Kt + (size_t)hkv * nk * D;
    const __global ushort *vp = V + hkv * D;
#if PF
    ushort8 kr[4 * NK], vr0[4], vr1[4];
    _Pragma("unroll") for (int cc = 0; cc < PD; cc++) { KLD(cc, kp, cc) }
    for (int kb = 0; kb < nk; kb += 8 * NK) {
        ROWS(ZS)
        _Pragma("unroll")
        for (int c = 0; c < NCH; c++) {
            if (c + PD < NCH) { KLD((c + PD) % 4, kp, c + PD) }
            else VLD((c + PD - NCH) % 4, vp + (c + PD - NCH) * HKV * D)
            _Pragma("unroll")
            for (int i = 0; i < 8; i++) { ROWS(FMAR) }
        }
        kp += NK * D * 8;
        ROWS(SOFT)
        _Pragma("unroll")
        for (int jj = 0; jj < 8 * NK; jj++) {
            if (jj + PD < 8 * NK) VLD((jj + PD) % 4, vp + (jj + PD) * HKV * D)
            else { KLD((jj + PD - 8 * NK) % 4, kp, jj + PD - 8 * NK) }   // next block (Kt is padded at the end)
            ROWS(PVR)
        }
        vp += 8 * NK * HKV * D;
    }
#else
    for (int kb = 0; kb < nk; kb += 8 * NK) {
        ROWS(ZS)
        _Pragma("unroll")
        for (int c = 0; c < D / 8; c++) {
            ushort8 kv0 = intel_sub_group_block_read_us8(kp + c * 64);
#if NK >= 2
            ushort8 kv1 = intel_sub_group_block_read_us8(kp + D * 8 + c * 64);
#endif
#if NK >= 4
            ushort8 kv2 = intel_sub_group_block_read_us8(kp + 2 * D * 8 + c * 64);
            ushort8 kv3 = intel_sub_group_block_read_us8(kp + 3 * D * 8 + c * 64);
#endif
#if SH
#define FH0(r) FMAHK(r, 0, kv0)
#define FH1(r) FMAHK(r, 1, kv1)
            ROWS(FH0)
#if NK >= 2
            ROWS(FH1)
#endif
#else
            _Pragma("unroll")
            for (int i = 0; i < 8; i++) {
                ROWS(FMA0)
#if NK >= 2
                ROWS(FMA1)
#endif
#if NK >= 4
                ROWS(FMA2)
                ROWS(FMA3)
#endif
            }
#endif
        }
        kp += ADV * NK * D * 8;
        if (!ADV) vp -= 8 * NK * HKV * D;
        ROWS(SOFT)
        PVLOOP(0)
#if NK >= 2
        PVLOOP(1)
#endif
#if NK >= 4
        PVLOOP(2)
        PVLOOP(3)
#endif
    }
#endif
    ROWS(STORE)
}
