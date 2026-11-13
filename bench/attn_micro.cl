// Microbenchmarks for attention on the GPU (docs/attention-gpu.md, section 7). Run by bench/attn_micro.sh via clbench:
// kernel(__global float *out, int n), rate = global * ops * n / time.

// 1. exp throughput (FP32). 8 independent chains x = EXP(x * c) with c < 0, so x stays in (0, 1].
//    -DEXP=exp or -DEXP=native_exp. One "op" = one exp plus the multiply feeding it (FP32 mul: 200 Gop/s, measured).
//    ops per work-item per iteration: 8 chains x 8 = 64.
#ifndef EXP
#define EXP exp
#endif
#define R8(s) s s s s s s s s
#define STEP8 x0 = EXP(x0 * c); x1 = EXP(x1 * c); x2 = EXP(x2 * c); x3 = EXP(x3 * c); \
              x4 = EXP(x4 * c); x5 = EXP(x5 * c); x6 = EXP(x6 * c); x7 = EXP(x7 * c);
__kernel void k_exp(__global float *out, int n) {
    const int gid = get_global_id(0);
    const float c = -0.5f - (float)(n & 1) * 0.001f;
    float x0 = (float)(gid & 255) * 1e-3f, x1 = x0 + 0.1f, x2 = x0 + 0.2f, x3 = x0 + 0.3f,
          x4 = x0 + 0.4f, x5 = x0 + 0.5f, x6 = x0 + 0.6f, x7 = x0 + 0.7f;
    for (int i = 0; i < n; i++) { R8(STEP8) }
    out[gid] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

// 2. SLM read bandwidth. Work-group 256, 32 KB of SLM (float4 x 2048). 8 loads of 16 B per work-item per iteration,
//    addresses (base + k * 256) & 2047 with base advancing every iteration; pass ops = 128 to clbench to get GB/s.
//    -DPAT=0: consecutive float4 per lane (lane l reads element l: whole 64 B lines per 4 lanes)
//    -DPAT=1: broadcast (all lanes of a subgroup read the same element)
//    -DPAT=2: stride 2 float4 per lane (half of every line wasted: bank-conflict / line-split case)
#ifndef PAT
#define PAT 0
#endif
#define LDK(k) acc += s[(base + (k) * 256) & 2047];
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void k_slm(__global float *out, int n) {
    __local float4 s[2048];
    const int l = get_local_id(0);
    for (int j = l; j < 2048; j += 256) s[j] = (float4)((float)j);
    barrier(CLK_LOCAL_MEM_FENCE);
#if PAT == 0
    int base = l;
#elif PAT == 1
    int base = get_sub_group_id() * 16;
#else
    int base = 2 * l;
#endif
    float4 acc = (float4)0.0f;
    for (int i = 0; i < n; i++) {
        LDK(0) LDK(1) LDK(2) LDK(3) LDK(4) LDK(5) LDK(6) LDK(7)
        base += 1;
    }
    out[get_global_id(0)] = acc.x + acc.y + acc.z + acc.w;
}

// 3. Mixed-mode multiply-add rate (SIMD8): a_j = fma(broadcast(q, i), k_j, a_j) with k_j a packed half vector
//    (-DMIX=1: FP16 operand converted inside mad, as in bench/attn.cl) or a float vector (-DMIX=0).
//    8 accumulators x 8 broadcast lanes = 64 multiply-adds = 128 FLOP per work-item per iteration (plus 8 operand
//    multiplies, not counted).
#ifndef MIX
#define MIX 1
#endif
#if MIX
#define KT half
#else
#define KT float
#endif
#define MJ(j) a##j = fma(s, (float)k##j, a##j);
#define MI(i) { float s = sub_group_broadcast(qv, i); MJ(0) MJ(1) MJ(2) MJ(3) MJ(4) MJ(5) MJ(6) MJ(7) }
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_mix(__global float *out, int n) {
    const int gid = get_global_id(0);
    float qv = (float)(gid & 7) * 0.01f + out[gid];
    KT k0 = (KT)(qv * 0.5f), k1 = k0 + (KT)0.1f, k2 = k0 + (KT)0.2f, k3 = k0 + (KT)0.3f,
       k4 = k0 + (KT)0.4f, k5 = k0 + (KT)0.5f, k6 = k0 + (KT)0.6f, k7 = k0 + (KT)0.7f;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    const KT c = (KT)(0.999f + out[gid + 1]);
    for (int i = 0; i < n; i++) {
        MI(0) MI(1) MI(2) MI(3) MI(4) MI(5) MI(6) MI(7)
        // change the operands every iteration (8 extra multiplies per 64 multiply-adds) so that a conversion
        // of the half operands cannot be hoisted out of the loop
        k0 *= c; k1 *= c; k2 *= c; k3 *= c; k4 *= c; k5 *= c; k6 *= c; k7 *= c;
    }
    out[gid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
