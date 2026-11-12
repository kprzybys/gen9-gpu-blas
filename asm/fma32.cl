// Template kernel for the full round-trip test: OpenCL C -> binary -> disasm -> (edit) -> asm -> run.
__kernel void k_fma(__global float *out, int n) {
    int gid = get_global_id(0);
    float a = (float)(n & 1) * 0.5f + 0.999f, b = (float)(n & 2) * 0.25f + 0.001f;
    float x0 = gid, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3, x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;
    for (int i = 0; i < n; i++) {
        #pragma unroll
        for (int u = 0; u < 16; u++) { x0 = fma(x0,a,b); x1 = fma(x1,a,b); x2 = fma(x2,a,b); x3 = fma(x3,a,b);
                                        x4 = fma(x4,a,b); x5 = fma(x5,a,b); x6 = fma(x6,a,b); x7 = fma(x7,a,b); }
    }
    out[gid] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
