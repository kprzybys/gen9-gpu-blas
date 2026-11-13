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
