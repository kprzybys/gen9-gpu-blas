// Krok 1 zlecenia: przepustowość i opóźnienie odczytów z pamięci na UHD 620.
// Sygnatura zgodna z clbench: (out, n, [BUF], [IMG]). MASK = rozmiar bufora/obrazu w bajtach - 1 (potęga 2).
// Wynik (xor) zapisywany do out, żeby kompilator nie usunął odczytów.
#ifndef MASK
#define MASK ((1u << 20) - 1)
#endif

// OWord block read: każda subgrupa SIMD8 czyta 128 B (block_read4: 8 linii × 4 uint) na krok,
// subgrupy przeplatają się tak, że wspólnie przechodzą cały bufor sekwencyjnie.
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_owb(__global uint *out, int n, __global uint *buf) {
    uint sg = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
    uint nsg = get_num_groups(0) * get_num_sub_groups();
    uint4 acc = 0;
    for (int i = 0; i < n; i++) {
        uint off = ((sg + i * nsg) * 128u) & MASK;           // bajty
        acc ^= intel_sub_group_block_read4(buf + off / 4);
    }
    out[get_global_id(0)] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}

// To samo, 2 niezależne odczyty w locie na krok (256 B na krok na subgrupę).
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

// Media block read z obrazu 2D (uint, szerokość W px = 4W B): blok 32 B × 8 wierszy = 256 B na krok.
// Subgrupy idą po kolejnych blokach wierszami.
#ifndef W
#define W 1024
#endif
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_media(__global uint *out, int n, __read_only image2d_t im) {
    uint sg = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
    uint nsg = get_num_groups(0) * get_num_sub_groups();
    uint bpr = (W * 4) / 32;                                   // bloków w pasie 8 wierszy
    uint nblk = ((MASK + 1) / 256);                            // bloków w obrazie
    uint8 acc = 0;
    for (int i = 0; i < n; i++) {
        uint b = (sg + i * nsg) % nblk;
        int2 xy = (int2)((b % bpr) * 32, (b / bpr) * 8);       // x w bajtach, y w wierszach
        acc ^= intel_sub_group_block_read8(im, xy);
    }
    out[get_global_id(0)] = acc.s0 ^ acc.s1 ^ acc.s2 ^ acc.s3 ^ acc.s4 ^ acc.s5 ^ acc.s6 ^ acc.s7;
}

// Opóźnienie: jeden wątek, łańcuch zależnych odczytów blokowych (adres zależy od poprzedniej wartości, która = 0).
// STRIDE bajtów między kolejnymi odczytami; rozmiar obszaru = MASK + 1.
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
