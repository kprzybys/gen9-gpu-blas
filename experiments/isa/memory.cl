// Step 1 of the task: memory read bandwidth and latency on the UHD 620.
// Signature compatible with clbench: (out, n, [BUF], [IMG]). MASK = buffer/image size in bytes - 1 (power of 2).
// The result (xor) is written to out so that the compiler does not remove the reads.
#ifndef MASK
#define MASK ((1u << 20) - 1)
#endif

// OWord block read: each SIMD8 subgroup reads 128 B (block_read4: 8 lanes × 4 uint) per step,
// the subgroups interleave so that together they walk the whole buffer sequentially.
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_owb(__global uint *out, int n, __global uint *buf) {
    uint sg = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
    uint nsg = get_num_groups(0) * get_num_sub_groups();
    uint4 acc = 0;
    for (int i = 0; i < n; i++) {
        uint off = ((sg + i * nsg) * 128u) & MASK;           // bytes
        acc ^= intel_sub_group_block_read4(buf + off / 4);
    }
    out[get_global_id(0)] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}

// The same, 2 independent reads in flight per step (256 B per step per subgroup).
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_owb2(__global uint *out, int n, __global uint *buf) {
    uint sg = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
    uint nsg = get_num_groups(0) * get_num_sub_groups();
    uint4 a = 0, b = 0;
    for (int i = 0; i < n; i++) {
        uint off = ((sg + i * nsg) * 256u) & MASK;
        a ^= intel_sub_group_block_read4(buf + off / 4);
        b ^= intel_sub_group_block_read4(buf + off / 4 + 32);
    }
    out[get_global_id(0)] = a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w;
}

// Media block read from a 2D image (uint, width W px = 4W B): block of 32 B × 8 rows = 256 B per step.
// Subgroups walk consecutive blocks row by row.
#ifndef W
#define W 1024
#endif
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_media(__global uint *out, int n, __read_only image2d_t im) {
    uint sg = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
    uint nsg = get_num_groups(0) * get_num_sub_groups();
    uint bpr = (W * 4) / 32;                                   // blocks in a band of 8 rows
    uint nblk = ((MASK + 1) / 256);                            // blocks in the image
    uint8 acc = 0;
    for (int i = 0; i < n; i++) {
        uint b = (sg + i * nsg) % nblk;
        int2 xy = (int2)((b % bpr) * 32, (b / bpr) * 8);       // x in bytes, y in rows
        acc ^= intel_sub_group_block_read8(im, xy);
    }
    out[get_global_id(0)] = acc.s0 ^ acc.s1 ^ acc.s2 ^ acc.s3 ^ acc.s4 ^ acc.s5 ^ acc.s6 ^ acc.s7;
}

// Latency: one thread, a chain of dependent block reads (the address depends on the previous value, which = 0).
// STRIDE bytes between consecutive reads; region size = MASK + 1.
#ifndef STRIDE
#define STRIDE 4160u
#endif
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_lat(__global uint *out, int n, __global uint *buf) {
    uint off = 0;
    uint4 v = 0;
    for (int i = 0; i < n; i++) {
        off = (off + STRIDE + v.x) & MASK & ~127u;
        v = intel_sub_group_block_read4(buf + off / 4);
    }
    out[get_global_id(0)] = v.x + off;
}
