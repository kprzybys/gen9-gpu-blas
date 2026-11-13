// kcl_const.cl -- regression test for kcl: program-scope __constant data and library math that IGC reads through a
// binding-table slot (.ze_info: const_base with bti_value). Before the fix kcl left that slot empty -> zeros.
__constant float TAB[8] = { 1.5f, -2.25f, 3.0f, 0.125f, 7.0f, -0.5f, 2.0f, 9.75f };
__kernel void k_const(__global const float *x, __global float *y, int n)
{
    const int i = get_global_id(0);
    if (i < n) y[i] = TAB[i & 7] * log10(x[i]) + exp(-x[i]);
}
