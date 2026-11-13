// Template for the hand-scheduled attention kernel (asm/attn_gen.py, docs/attention-gpu.md section 9).
// Variant A of bench/attn.cl (FP32 accumulation, SIMD8, RQ rows per subgroup, lane = key for Q K^T,
// lane = dim for P V), reduced to what the generator needs, with Q loaded and O stored by subgroup block reads /
// writes (one a64 pointer per row instead of per-dim offsets, so IGC keeps fewer registers live across the loop).
// attn_gen.py keeps IGC's prologue (Q load and scale, pointers, O = 0) and epilogue (O stores) and replaces the
// whole loop; IGC's own loop only serves to locate Q, O, the K/V pointers and the counter (symbolic pass).
// Layouts and build parameters (D, G, HKV, RQ, NK) as in bench/attn.cl; k_kt is the same K transpose.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#if D == 128
typedef float16 OV;
#define BR(p) as_float16((uint16)(intel_sub_group_block_read8(p), intel_sub_group_block_read8((p) + 64)))
#define BW(p, v) { intel_sub_group_block_write8((p), as_uint8((v).lo)); intel_sub_group_block_write8((p) + 64, as_uint8((v).hi)); }
#elif D == 64
typedef float8 OV;
#define BR(p) as_float8(intel_sub_group_block_read8(p))
#define BW(p, v) intel_sub_group_block_write8((p), as_uint8(v));
#endif

#if NK == 1
typedef float SV;
#define SK(v, k) (v)
#define SMAX(v) (v)
#define SSUM(v) (v)
#elif NK == 2
typedef float2 SV;
#define SK(v, k) (v)[k]
#define SMAX(v) fmax((v).s0, (v).s1)
#define SSUM(v) ((v).s0 + (v).s1)
#endif

#if RQ == 1
#define ROWS(X) X(0)
#elif RQ == 2
#define ROWS(X) X(0) X(1)
#elif RQ == 4
#define ROWS(X) X(0) X(1) X(2) X(3)
#endif

__kernel void k_kt(__global const ushort *K, __global ushort *Kt, int nk) {
    int i = get_global_id(0);
    int d = i % D, th = i / D, hkv = th % HKV, t = th / HKV;
    Kt[((hkv * (nk / 8) + t / 8) * D + d) * 8 + (t & 7)] = K[i];
}

#define DECL(r) OV q##r; OV o##r = 0; float m##r = -1e30f; float l##r = 0.f; SV s##r;
// Q layout: default Q[h][t][d] (bench/attn.c); QTOK: Q[t][h][d], what llama.cpp's flash_attn_ext gets (the
// permuted view of Qcur [d_head, n_head, n_tokens]: nb1 = token stride H*D*4, nb2 = head stride D*4), same as O.
#ifdef QTOK
#define QOFF(t, hh) ((t) * (HKV * G) + hkv * G + (hh))
#else
#define QOFF(t, hh) ((hkv * G + (hh)) * nq + (t))
#endif
#ifdef ROWVIS   // rows clamped to the last one: with n_q*G odd the last subgroup computes that row twice (same store)
#define ROWOF(r) min(row0 + r, nq * G - 1)
#else
#define ROWOF(r) (row0 + r)
#endif
#define LOADQ(r) { int row = ROWOF(r), t = row / G, hh = row - t * G; \
    q##r = BR((const __global uint *)(Q + QOFF(t, hh) * D)) * qscale; }
#define ZS(r) s##r = 0;
#define FMAK(r, k, kvk) SK(s##r, k) = fma(sub_group_broadcast(q##r[c], i), (float)as_half(kvk[i]), SK(s##r, k));
#define FMA0(r) FMAK(r, 0, kv0)
#define FMA1(r) FMAK(r, 1, kv1)
#define SOFT(r) { float mn = sub_group_reduce_max(SMAX(s##r)); \
    if (mn > m##r) { float a = native_exp2(m##r - mn); o##r *= a; l##r *= a; m##r = mn; } \
    s##r = native_exp2(s##r - m##r); l##r += SSUM(s##r); }
#define PVK(r, k) { float p = sub_group_broadcast(SK(s##r, k), j); \
    _Pragma("unroll") for (int c = 0; c < 8; c++) o##r[c] = fma(p, (float)as_half(v0[c]), o##r[c]); \
    if (D == 128) { _Pragma("unroll") for (int c = 0; c < 8; c++) o##r[8 + c] = fma(p, (float)as_half(v1[c]), o##r[8 + c]); } }
#define PV0(r) PVK(r, 0)
#define PV1(r) PVK(r, 1)
#define STORE(r) { float il = 1.f / sub_group_reduce_add(l##r); int row = ROWOF(r), t = row / G, hh = row - t * G; \
    __global uint *op = (__global uint *)(O + (t * (HKV * G) + hkv * G + hh) * D); BW(op, o##r * il) }
#if D == 128
#define READV ushort8 v0 = intel_sub_group_block_read_us8(vp), v1 = intel_sub_group_block_read_us8(vp + 64);
#else
#define READV ushort8 v0 = intel_sub_group_block_read_us8(vp), v1;
#endif
#define PVLOOP(k) _Pragma("unroll") for (int j = 0; j < 8; j++) { READV vp += HKV * D; ROWS(PV##k) }

// CAUSAL (stage 3, docs/attention-gpu.md section 11): index-based masking. Query token t sees the first
// nv = min(nk, vis0 + dvis * t) keys (dvis 1: causal, 0: the same count for every token); the loop runs over the
// ceil(nv / (8*NK)) key blocks only (fully masked blocks are skipped), and attn_gen.py masks the keys >= nv of the
// one partial block in its generated loop (lim = nv). All RQ rows of a subgroup belong to one token (G % RQ == 0).
// ROWVIS (docs/attention-gpu.md section 13, Qwen2.5-7B G 7 with RQ 2): the rows of a subgroup may belong to two
// consecutive tokens. The loop runs to the last row's count nv; each row r has its own limit lim_r, handed to
// attn_gen.py through IGC's l register of that row: l##r starts at (float) lim_r (the generated loop zeroes its l sums
// itself and reads lim_r first; it ties the register to its row by data flow, the l += sum(p) of that row's scores).
// The generated mask then masks each row against its own limit in the last block (lim_r >= nv - dvis*(RQ-1) lies in
// it). per_head rounds up and row indices are clamped (ROWOF), so n_q*G need not be a multiple of RQ.
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_attn(__global const float *Q, __global const ushort *Kt, __global const ushort *V, __global float *O,
                     int nq, int nk, float qscale
#ifdef CAUSAL
                     , int vis0, int dvis
#endif
                     ) {
    int sg = get_group_id(0) * get_num_sub_groups() + get_sub_group_id(), lane = get_sub_group_local_id();
#ifdef ROWVIS
    int per_head = (nq * G + RQ - 1) / RQ;
#else
    int per_head = nq * G / RQ;
#endif
#ifdef HKV_INNER   // grid order experiment (docs/attention-gpu.md section 10): KV heads alternate between subgroups
    int hkv = sg % HKV, row0 = (sg / HKV) * RQ;
#else               // default: all query rows of KV head 0, then of head 1, ...
    int hkv = sg / per_head, row0 = (sg - hkv * per_head) * RQ;
#endif
    ROWS(DECL)
    ROWS(LOADQ)
    const __global ushort *kp = Kt + (size_t)hkv * nk * D;
    const __global ushort *vp = V + hkv * D;
#if defined(CAUSAL) && defined(ROWVIS)
#define LIMR(r) l##r = (float)min(nk, vis0 + dvis * (ROWOF(r) / G));
    ROWS(LIMR)
    const int nv = sub_group_broadcast(min(nk, vis0 + dvis * (ROWOF(RQ - 1) / G)), 0);   // the last row's (largest) count
#elif defined(CAUSAL)
    const int nv = sub_group_broadcast(min(nk, vis0 + dvis * (row0 / G)), 0);   // uniform for IGC: scalar loop limit
#else
    const int nv = nk;
#endif
    for (int kb = 0; kb < nv; kb += 8 * NK) {
        ROWS(ZS)
        _Pragma("unroll")
        for (int c = 0; c < D / 8; c++) {
            ushort8 kv0 = intel_sub_group_block_read_us8(kp + c * 64);
#if NK >= 2
            ushort8 kv1 = intel_sub_group_block_read_us8(kp + D * 8 + c * 64);
#endif
            _Pragma("unroll")
            for (int i = 0; i < 8; i++) {
                ROWS(FMA0)
#if NK >= 2
                ROWS(FMA1)
#endif
            }
        }
        kp += NK * D * 8;
        ROWS(SOFT)
        PVLOOP(0)
#if NK >= 2
        PVLOOP(1)
#endif
    }
    ROWS(STORE)
}
