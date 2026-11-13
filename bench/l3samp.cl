// Does the texture sampler add read bandwidth to the L3 data port (bench/l3port.cl)? Gen9 subslices have a sampler
// with its own caches next to the data port. Image: RGBA uint32 (16 B per texel), W x H texels, 64 KB by default
// (L3-resident). One SIMD8 sampler message = 8 texels = 128 B, the size of one OWord block read.
// k_samp: sampler only; k_both: one block read (data port) + one sampler read per step; MADS FP32 mads per 128 B.
#ifndef W
#define W 64
#endif
#ifndef H
#define H 64
#endif
#ifndef MASK
#define MASK ((1u << 16) - 1)
#endif
#ifndef PF
#define PF 2
#endif
#ifndef RP
#define RP 1
#endif
__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

// block b of a subgroup -> 8 consecutive texels of one row (128 B), walking the image like l3port walks the buffer
#define TXY(b) (int2)((int)((((sg + (uint)(b) * nsg) * 8u) % (W * H)) % W) + lane, (int)((((sg + (uint)(b) * nsg) * 8u) % (W * H)) / W))

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_samp(__global uint *out, int n, __read_only image2d_t im) {
    const uint sg = get_group_id(0) * get_num_sub_groups() + get_sub_group_id(), nsg = get_num_groups(0) * get_num_sub_groups();
    const int lane = get_sub_group_local_id();
    uint4 x = 0, v[PF];
    _Pragma("unroll") for (int j = 0; j < PF; j++) v[j] = read_imageui(im, smp, TXY(j));
    for (int i = 0; i < n; i += PF) {
        _Pragma("unroll") for (int j = 0; j < PF; j++) {
            x ^= v[j];
            v[j] = read_imageui(im, smp, TXY(i + PF + j));
        }
    }
    _Pragma("unroll") for (int j = 0; j < PF; j++) x ^= v[j];
    out[get_global_id(0)] = x.x ^ x.y ^ x.z ^ x.w;
}

// n steps, each: RP 128 B block reads from buf (data port) and one 128 B sampler read
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_both(__global uint *out, int n, __global uint *buf, __read_only image2d_t im) {
    const uint sg = get_group_id(0) * get_num_sub_groups() + get_sub_group_id(), nsg = get_num_groups(0) * get_num_sub_groups();
    const int lane = get_sub_group_local_id();
    const uint step = nsg * 128u;
    uint off = sg * 128u;
    uint4 x = 0, v[PF * RP], t[PF];
    _Pragma("unroll") for (int j = 0; j < PF; j++) {
        _Pragma("unroll") for (int r = 0; r < RP; r++) { v[j * RP + r] = intel_sub_group_block_read4(buf + (off & MASK) / 4); off += step; }
        t[j] = read_imageui(im, smp, TXY(j));
    }
    for (int i = 0; i < n; i += PF) {
        _Pragma("unroll") for (int j = 0; j < PF; j++) {
            x ^= t[j];
            t[j] = read_imageui(im, smp, TXY(i + PF + j));
            _Pragma("unroll") for (int r = 0; r < RP; r++) {
                x ^= v[j * RP + r];
                v[j * RP + r] = intel_sub_group_block_read4(buf + (off & MASK) / 4); off += step;
            }
        }
    }
    _Pragma("unroll") for (int j = 0; j < PF; j++) { x ^= t[j]; _Pragma("unroll") for (int r = 0; r < RP; r++) x ^= v[j * RP + r]; }
    out[get_global_id(0)] = x.x ^ x.y ^ x.z ^ x.w;
}
