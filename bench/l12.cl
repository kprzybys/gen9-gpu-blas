// Poziomy 1-2 BLAS: proste kernele do pomiaru energii (ograniczone pamięcią).
__kernel void saxpy(__global const float *x, __global float *y, float a) {
    const size_t i = get_global_id(0);
    y[i] = fma(a, x[i], y[i]);
}
// y = A·x, A: M×N row-major; jedna subgrupa 16 na wiersz, odczyt blokowy wiersza.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void sgemv_n(__global const float *A, __global const float *x, __global float *y, int N) {
    const int row = get_group_id(0);
    const __global float *a = A + (size_t)row * N;
    float s = 0.0f;
    for (int j = get_sub_group_local_id(); j < N; j += 16) s = fma(a[j], x[j], s);
    s = sub_group_reduce_add(s);
    if (get_sub_group_local_id() == 0) y[row] = s;
}
