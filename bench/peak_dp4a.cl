// Peak INT8 dot-product throughput (DP4a on Gen12LP; Gen9 has no DP4a and emulates it).
// CH independent chains (8 or 16) x 16 per loop iteration; ops per work-item per iteration = CH * 16 * 8
// (one 4x8-bit dot product with accumulate = 4 multiplies + 4 adds).
// SAT=1: dot_acc_sat_4x8packed_ss_int (saturating accumulate); SAT=0: dot_4x8packed_ss_int(a, b) + x.
#ifndef CH
#define CH 8
#endif
#ifndef SAT
#define SAT 1
#endif
// The accumulator is also the first operand (its bytes), so nothing is loop-invariant and the compiler cannot fold
// the chain (with constant a, b the non-saturating form was hoisted out of the loop: 20 TOPS, invalid).
#if SAT
#define S(x) x = dot_acc_sat_4x8packed_ss_int(as_uint(x), b, x);
#else
#define S(x) x = dot_4x8packed_ss_int(as_uint(x), b) + x;
#endif
__kernel void k_dp4a(__global int *out, int n) {
    int gid = get_global_id(0);
    uint a = 0x01020304u ^ (uint)n, b = 0x05060708u + (uint)(gid & 7);
    int x0 = gid, x1 = gid + 1, x2 = gid + 2, x3 = gid + 3, x4 = gid + 4, x5 = gid + 5, x6 = gid + 6, x7 = gid + 7;
#if CH == 16
    int y0 = gid + 8, y1 = gid + 9, y2 = gid + 10, y3 = gid + 11, y4 = gid + 12, y5 = gid + 13, y6 = gid + 14, y7 = gid + 15;
#define S8 S(x0) S(x1) S(x2) S(x3) S(x4) S(x5) S(x6) S(x7) S(y0) S(y1) S(y2) S(y3) S(y4) S(y5) S(y6) S(y7)
#else
#define S8 S(x0) S(x1) S(x2) S(x3) S(x4) S(x5) S(x6) S(x7)
#endif
    for (int i = 0; i < n; i++) { S8 S8 S8 S8 S8 S8 S8 S8 S8 S8 S8 S8 S8 S8 S8 S8 }
    int r = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
#if CH == 16
    r += y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7;
#endif
    out[gid] = r;
}
// Latency: one chain.
__kernel void k_dp4a_lat(__global int *out, int n) {
    int gid = get_global_id(0);
    uint a = 0x01020304u ^ (uint)n, b = 0x05060708u + (uint)(gid & 7);
    int x0 = gid;
#define L16 S(x0) S(x0) S(x0) S(x0) S(x0) S(x0) S(x0) S(x0) S(x0) S(x0) S(x0) S(x0) S(x0) S(x0) S(x0) S(x0)
    for (int i = 0; i < n; i++) { L16 }
    out[gid] = x0;
}
