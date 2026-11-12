// Minimalne jądro do pierwszego eksperymentu bez NEO: p[i] = v + i (stateless, SIMD16).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void zapis(__global uint *p, uint v) {
    const uint i = get_global_id(0);
    p[i] = v + i;
}

// Krok w łańcuchu zależnych jąder: b[i] = a[i] + 1.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void kopiuj(__global const uint *a, __global uint *b) {
    const uint i = get_global_id(0);
    b[i] = a[i] + 1;
}

// Krok z zależnością „na krzyż” (test wyścigu między jądrami): b[i] = a[n-1-i] + 1.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void odwroc(__global const uint *a, __global uint *b, uint n) {
    const uint i = get_global_id(0);
    b[i] = a[n - 1 - i] + 1;
}
