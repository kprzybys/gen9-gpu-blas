// Gen8 Cherryview bring-up probes (docs/gen8.md 6a), compiled only into kernels_gen8.c. FP64 copy (copy16_d) hung the
// GPU; its only instruction forms not seen in kernels that completed are A64 QWord scattered messages and DF-typed moves.
// probe_df_dword: DF data and DF arithmetic, memory through dword messages only (doubles as uint pairs).
// probe_qword: A64 QWord scattered read/write, integer QWord data, no DF type.
__kernel void probe_df_dword(__global const uint *x, __global uint *y, double s) {
    const size_t i = get_global_id(0);
    const double d = as_double(vload2(i, x)) * s;
    vstore2(as_uint2(d), i, y);
}
__kernel void probe_qword(__global const ulong *x, __global ulong *y) {
    const size_t i = get_global_id(0);
    y[i] = x[i] + 1;
}
// Ceilings for the Gen8 performance model (lib/bench_gen8.c): FMA peak and memory bandwidth.
#define FMA8(T) x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b); \
    x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b);
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void peak_f32(__global float *out, int iters, float a, float b) {
    const float s = get_global_id(0) * 1e-7f;
    float x0 = s, x1 = s + 1, x2 = s + 2, x3 = s + 3, x4 = s + 4, x5 = s + 5, x6 = s + 6, x7 = s + 7;
    for (int i = 0; i < iters; i++) { FMA8(float) FMA8(float) }
    out[get_global_id(0)] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void peak_f64(__global uint *out, int iters, double a, double b) {
    const double s = get_global_id(0) * 1e-7;
    double x0 = s, x1 = s + 1, x2 = s + 2, x3 = s + 3, x4 = s + 4, x5 = s + 5, x6 = s + 6, x7 = s + 7;
    for (int i = 0; i < iters; i++) { FMA8(double) }
    vstore2(as_uint2(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7), get_global_id(0), out);
}
__kernel void mem_read(__global const float4 *p, long n4, __global float *out) {
    float4 s = 0.0f;
    for (long i = get_global_id(0); i < n4; i += get_global_size(0)) s += p[i];
    out[get_global_id(0)] = s.x + s.y + s.z + s.w;
}
__kernel void mem_write(__global float4 *p, long n4) {
    for (long i = get_global_id(0); i < n4; i += get_global_size(0)) p[i] = (float4)(i);
}
__kernel void mem_copy(__global const float4 *p, __global float4 *q, long n4) {
    for (long i = get_global_id(0); i < n4; i += get_global_size(0)) q[i] = p[i];
}
// Work-group / subgroup ids as a SIMD8 kernel sees them (lib/bench_gen8.c ids).
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void dbg_ids8(__global int *out) {
    const int t = get_global_id(0) / 8;
    if (get_sub_group_local_id() == 0) {
        out[4 * t] = get_group_id(0); out[4 * t + 1] = get_num_sub_groups(); out[4 * t + 2] = get_sub_group_id();
        out[4 * t + 3] = get_local_size(0);
    }
}
