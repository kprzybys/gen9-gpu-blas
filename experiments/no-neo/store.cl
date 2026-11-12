// Minimal kernel for the first experiment without NEO: p[i] = v + i (stateless, SIMD16).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void store(__global uint *p, uint v) {
    const uint i = get_global_id(0);
    p[i] = v + i;
}

// Step in a chain of dependent kernels: b[i] = a[i] + 1.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void copy(__global const uint *a, __global uint *b) {
    const uint i = get_global_id(0);
    b[i] = a[i] + 1;
}

// Step with a cross-wise dependency (race test between kernels): b[i] = a[n-1-i] + 1.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void reverse(__global const uint *a, __global uint *b, uint n) {
    const uint i = get_global_id(0);
    b[i] = a[n - 1 - i] + 1;
}
