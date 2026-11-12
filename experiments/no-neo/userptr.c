// userptr microtest (DRM_IOCTL_I915_GEM_USERPTR): the GPU reads and writes plain process memory (malloc, like NumPy) without copies.
// Comparison for size S (bytes) of one copy kernel (b[i] = a[i] + 1, input S, output S):
//   copy      -- as gen9blas does today for arrays outside USM: memcpy to a GEM buffer, kernel, memcpy the result back;
//   up        -- userptr created on every call (create, execbuf: page pinning, close);
//   up_cached -- userptr created once, reused many times (pages pinned; cost like a GEM buffer?);
//   gem       -- the kernel alone on GEM buffers (lower bound).
// Checks correctness (also: the CPU changes the input between calls -> the GPU sees the new data) and the flags
// I915_USERPTR_READ_ONLY and PROBE. Usage: ./userptr [S_MB ...]
#include <time.h>
#include "kernels.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double cpu(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static struct g9i g;
static uint32_t j_copy;

// userptr over the pages covering [p, p+n); returns the object (GPU address = addr) and the offset of p within the object.
static int up_new(struct g9i_bo *b, void *p, uint64_t n, uint64_t addr, uint32_t flags, uint64_t *off) {
    const uintptr_t s = (uintptr_t)p & ~4095ul, e = ((uintptr_t)p + n + 4095) & ~4095ul;
    struct drm_i915_gem_userptr u = { .user_ptr = s, .user_size = e - s, .flags = flags };
    if (ioctl(g.fd, DRM_IOCTL_I915_GEM_USERPTR, &u)) return -errno;
    *b = (struct g9i_bo){ .h = u.handle, .addr = addr, .size = e - s, .map = NULL, .written = 1 };
    if (off) *off = (uintptr_t)p - s;
    return 0;
}
static void close_bo(uint32_t h) { struct drm_gem_close c = { .handle = h }; G9I_CHECK(ioctl(g.fd, DRM_IOCTL_GEM_CLOSE, &c)); }

static void run(struct g9i_bo *in_bo, uint64_t a_in, struct g9i_bo *out_bo, uint64_t a_out, uint64_t S) {
    g9i_begin(&g);
    uint64_t a[2] = { a_in, a_out };
    g9i_call(&g, j_copy, (uint32_t)(S / 4 / 16), 1, 1, a, NULL);
    struct g9i_bo *bo[2] = { in_bo, out_bo }; in_bo->written = 0; out_bo->written = 1;
    g9i_submit(&g, bo, 2);
    if (g9i_wait(&g, 10000000000ll)) { fprintf(stderr, "GPU did not finish in 10 s\n"); exit(2); }
}
static uint64_t count_bad(const uint32_t *in_bo, const uint32_t *out_bo, uint64_t n) {
    uint64_t z = 0; for (uint64_t i = 0; i < n; i++) z += out_bo[i] != in_bo[i] + 1; return z;
}

int main(int argc, char **argv) {
    g9i_init(&g);
    j_copy = g9i_add_kernel(&g, &copy_kernel);
    { int v = 0; struct drm_i915_getparam gp = { .param = I915_PARAM_HAS_USERPTR_PROBE, .value = &v };
      int r = ioctl(g.fd, DRM_IOCTL_I915_GETPARAM, &gp);
      printf("I915_PARAM_HAS_USERPTR_PROBE: %s\n", r ? strerror(errno) : v ? "yes" : "no"); }
    {   // flags on a small buffer
        void *p = aligned_alloc(4096, 8192); struct g9i_bo b; int r;
        r = up_new(&b, p, 8192, 0, I915_USERPTR_READ_ONLY, NULL);
        printf("I915_USERPTR_READ_ONLY: %s\n", r ? strerror(-r) : "accepted"); if (!r) close_bo(b.h);
        r = up_new(&b, p, 8192, 0, I915_USERPTR_PROBE, NULL);
        printf("I915_USERPTR_PROBE (valid address): %s\n", r ? strerror(-r) : "accepted"); if (!r) close_bo(b.h);
        r = up_new(&b, (void *)0x1000, 8192, 0, I915_USERPTR_PROBE, NULL);
        printf("I915_USERPTR_PROBE (bad address 0x1000): %s\n", r ? strerror(-r) : "accepted (!)"); if (!r) close_bo(b.h);
        free(p);
    }
    if (getenv("PHASES")) {   // per-call userptr cost split into phases (CPU in us), 64 MB input + 64 MB output
        const uint64_t S = 64ull << 20, n = S / 4; uint32_t *h = malloc(S), *h2 = malloc(S);
        for (uint64_t i = 0; i < n; i++) { h[i] = (uint32_t)i; h2[i] = 0; }
        for (int r = 0; r < 4; r++) {
            struct g9i_bo up_in, up_out; uint64_t off_in, off_out; double c0 = cpu(), t0 = now();
            G9I_CHECK(up_new(&up_in, h, S, 0x100000000ull, 0, &off_in)); G9I_CHECK(up_new(&up_out, h2, S, 0x200000000ull, 0, &off_out));
            double c1 = cpu();
            g9i_begin(&g); uint64_t a[2] = { 0x100000000ull + off_in, 0x200000000ull + off_out };
            g9i_call(&g, j_copy, (uint32_t)(n / 16), 1, 1, a, NULL);
            struct g9i_bo *bo[2] = { &up_in, &up_out }; up_in.written = 0; g9i_submit(&g, bo, 2);
            double c2 = cpu(), t2 = now(); g9i_wait(&g, 10000000000ll); double c3 = cpu(), t3 = now();
            close_bo(up_in.h); close_bo(up_out.h); double c4 = cpu();
            printf("PHASES: create %.0f, execbuf %.0f (wall %.0f), wait %.0f (wall %.0f), close %.0f us CPU\n",
                   (c1 - c0) * 1e6, (c2 - c1) * 1e6, (t2 - t0) * 1e6, (c3 - c2) * 1e6, (t3 - t2) * 1e6, (c4 - c3) * 1e6);
        }
        return 0;
    }
    int n_sizes = argc > 1 ? argc - 1 : 5; double mbs[] = { 0.0625, 1, 4, 16, 64 };
    const uint64_t A_IN = 0x100000000ull, A_OUT = 0x200000000ull;   // fixed GPU addresses for the userptr objects
    printf("\n%8s | %-26s | %-26s | %-26s | %-12s | chk.\n", "S", "copy: time / CPU [us]", "up per call", "up created once", "kernel only");
    for (int si = 0; si < n_sizes; si++) {
        const uint64_t S = (uint64_t)((argc > 1 ? atof(argv[si + 1]) : mbs[si]) * (1 << 20)) & ~63ull, n = S / 4;
        // memory as in NumPy: malloc (large blocks: mmap, start 16 B past the page boundary)
        uint32_t *h = malloc(S), *h2 = malloc(S);
        for (uint64_t i = 0; i < n; i++) { h[i] = (uint32_t)(i * 2654435761u); h2[i] = 0; }
        struct g9i_bo B_IN, B_OUT; g9i_bo_new(&g, &B_IN, S); g9i_bo_new(&g, &B_OUT, S);
        const int R = S >= (16 << 20) ? 15 : 60;
        double t[4][60], c[4] = { 0 }; uint64_t z = 0;
        for (int r = 0; r < R; r++) {
            double a = now(), ca = cpu();   // copy
            memcpy(B_IN.map, h, S); run(&B_IN, B_IN.addr, &B_OUT, B_OUT.addr, S); memcpy(h2, B_OUT.map, S);
            t[0][r] = now() - a; c[0] += cpu() - ca;
        }
        z += count_bad(h, h2, n); memset(h2, 0, S);
        for (int r = 0; r < R; r++) {
            double a = now(), ca = cpu();   // userptr per call
            struct g9i_bo up_in, up_out; uint64_t off_in, off_out;
            G9I_CHECK(up_new(&up_in, h, S, A_IN, 0, &off_in)); G9I_CHECK(up_new(&up_out, h2, S, A_OUT, 0, &off_out));
            run(&up_in, A_IN + off_in, &up_out, A_OUT + off_out, S);
            close_bo(up_in.h); close_bo(up_out.h);
            t[1][r] = now() - a; c[1] += cpu() - ca;
        }
        z += count_bad(h, h2, n); memset(h2, 0, S);
        struct g9i_bo up_in, up_out; uint64_t off_in, off_out;
        G9I_CHECK(up_new(&up_in, h, S, A_IN, 0, &off_in)); G9I_CHECK(up_new(&up_out, h2, S, A_OUT, 0, &off_out));
        run(&up_in, A_IN + off_in, &up_out, A_OUT + off_out, S);   // first use (pinning) outside the measurement
        for (int r = 0; r < R; r++) {
            if (r == R / 2) for (uint64_t i = 0; i < n; i++) h[i] ^= 0x5a5a;   // CPU changes the input: the GPU must see it
            double a = now(), ca = cpu();
            run(&up_in, A_IN + off_in, &up_out, A_OUT + off_out, S);
            t[2][r] = now() - a; c[2] += cpu() - ca;
        }
        z += count_bad(h, h2, n);
        close_bo(up_in.h); close_bo(up_out.h);
        for (int r = 0; r < R; r++) {
            double a = now(), ca = cpu(); run(&B_IN, B_IN.addr, &B_OUT, B_OUT.addr, S); t[3][r] = now() - a; c[3] += cpu() - ca;
        }
        for (int k = 0; k < 4; k++) qsort(t[k], R, sizeof(double), cmp_d);
        printf("%6.3g MB | %10.0f / %-13.0f | %10.0f / %-13.0f | %10.0f / %-13.0f | %10.0f   | %s (h%%4096 = %lu)\n",
               S / 1048576.0, t[0][R / 2] * 1e6, c[0] / R * 1e6, t[1][R / 2] * 1e6, c[1] / R * 1e6,
               t[2][R / 2] * 1e6, c[2] / R * 1e6, t[3][R / 2] * 1e6, z ? "BAD" : "OK", (uintptr_t)h % 4096);
        close_bo(B_IN.h); close_bo(B_OUT.h); munmap(B_IN.map, B_IN.size); munmap(B_OUT.map, B_OUT.size);
        free(h); free(h2);
    }
    return 0;
}
