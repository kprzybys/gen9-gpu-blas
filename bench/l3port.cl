// L3 / data-port read ceiling in the access pattern of the Qwen attention loop (docs/attention-gpu.md section 8,
// "What limits it now"): SIMD8 subgroups stream 128 B OWord block reads that hit L3, each block feeding MADS
// FP32 SIMD8 mads (Qwen: 16 mads per 128 B block, 8 B per mad), with PF blocks kept in flight per thread
// (the hand loop keeps 1-3 steps ahead). Question: is ~150 GB/s at 1.0 GHz (~6.2 B/clk/EU) the port ceiling?
// Signature for bench/clbench: (out, n, buf); BUF must be >= MASK + 1. n = blocks per subgroup (multiple of PF).
// Build-time: MASK (region bytes - 1, power of 2; 64 KB stays in L3), MADS (0..32), PF (1, 2, 4),
// SHARE (subgroups reading the same block sequence, as the subgroups of one KV head do).
#ifndef MASK
#define MASK ((1u << 16) - 1)
#endif
#ifndef MADS
#define MADS 16
#endif
#ifndef PF
#define PF 2
#endif
#ifndef SHARE
#define SHARE 1
#endif


__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_port(__global float *out, int n, __global uint *buf) {
    const uint sg = (get_group_id(0) * get_num_sub_groups() + get_sub_group_id()) / SHARE;
    const uint nsg = max(1u, (uint)(get_num_groups(0) * get_num_sub_groups()) / SHARE);
    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};      // independent accumulators (mad latency hidden)
    uint x = 0;
    uint4 v[PF];
    // block b of this subgroup: byte offset ((sg + b * nsg) * 128) & MASK, advanced by adds only (the attention
    // loop also just adds to its K/V pointers): about 3 non-mad instructions per block, as in the real loop
    const uint step = nsg * 128u;
    uint off = sg * 128u;
    _Pragma("unroll") for (int j = 0; j < PF; j++) {
        v[j] = intel_sub_group_block_read4(buf + (off & MASK) / 4);
        off += step;
    }
    for (int i = 0; i < n; i += PF) {
        _Pragma("unroll") for (int j = 0; j < PF; j++) {
#if MADS == 0
            x ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w;
#else
            // operands from adjacent data registers (different parity: no src1/src2 bank conflict)
            float4 f = as_float4(v[j]);
            _Pragma("unroll") for (int k = 0; k < MADS; k++) a[k & 7] = fma(f[k & 3], f[(k + 1) & 3], a[k & 7]);
#endif
            v[j] = intel_sub_group_block_read4(buf + (off & MASK) / 4);   // refill after use: PF blocks in flight
            off += step;
        }
    }
    _Pragma("unroll") for (int j = 0; j < PF; j++) x ^= v[j].x;
    out[get_global_id(0)] = a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7] + (float)x;
}
