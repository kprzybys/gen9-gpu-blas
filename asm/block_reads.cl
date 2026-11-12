// Test: how block read from cl_intel_subgroups maps to send messages (buffer -> OWord, image -> media block).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void buf_read(__global uint *in, __global uint *out) {
    uint4 v = intel_sub_group_block_read4(in + get_group_id(0) * 32);
    intel_sub_group_block_write4(out + get_group_id(0) * 32, v);
}
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void img_read(__read_only image2d_t im, __global uint *out) {
    uint8 v = intel_sub_group_block_read8(im, (int2)(get_group_id(0) * 32, 0));
    intel_sub_group_block_write8(out + get_group_id(0) * 64, v);
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hf_mad_bcast(__global half *o, __global half *b) {
    // HGEMM loop pattern: result += A(scalar broadcast from lane k) * B(vector)
    half16 acc = 0; half a = o[get_sub_group_local_id()];
    for (int k = 0; k < 16; k++) acc = fma((half16)sub_group_broadcast(a, k), vload16(k, b), acc);
    vstore16(acc, get_global_id(0), o);
}
