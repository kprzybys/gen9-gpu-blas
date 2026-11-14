// Flash attention on the GPU for the ggml backend (docs/attention-gpu.md section 11): the hand-scheduled kernels of
// asm/attn_gen.py (*_tokc binaries: Q in llama.cpp's layout, index-based causal masking), run through kcl on the
// caller's memory (i915 userptr). Separate from gen9blas.c on purpose (own state, own error text); needs kcl (not
// built for the NEO variants of the library).
//
// Per call (one ggml FLASH_ATTN_EXT node): one kcl operation with userptr regions K, V (from the view's first row to
// the end of the whole KV tensor: the view grows every 256 tokens, the tensor stays, so its pages are pinned once),
// Q and O; k_kt transposes the first nk rows of K into an own GPU buffer (padded by NK*D*16 B: the kernel's ring
// prefetches past the last block, section 9); k_attn in launches of at most ~GEN9BLAS_MAX_LAUNCH_MS (default 100 ms
// at a conservative 200 GFLOPS = 2e10 FLOP: kcl has no mid-thread preemption and a launch cut by the 640 ms
// preemption timeout returns silently wrong results; kcl_wait reports a reset as an error); one submission, wait.
#include "gen9blas.h"
#include "kcl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// rowvis: binary built with -DROWVIS=1 (asm/attn.cl): per-row visible key counts (the RQ rows of a subgroup may span two
// consecutive tokens, so dvis 1 is right for any G when RQ <= 2) and clamped row indices (n_q*G need not divide by RQ)
typedef struct { int D, G, HKV, RQ, NK, rowvis; const char *bin; cl_kernel kt, attn; int state; } attn_bin_t;   // state: 0 untried, 1 ok, -1 failed
static attn_bin_t bins[] = {
    { 128, 6, 2, 2, 2, 0, "attn_qwen_tokc.bin", NULL, NULL, 0 },    // Qwen2.5-1.5B (sched4)
    { 64, 4, 8, 4, 2, 0, "attn_llama_tokc.bin", NULL, NULL, 0 },    // Llama-3.2-1B (sched2)
    { 64, 1, 20, 4, 2, 0, "attn_whisper_tokc.bin", NULL, NULL, 0 }, // whisper large-v3(-turbo) encoder (sched2); G < RQ: dvis 0 only
    { 128, 7, 4, 2, 2, 1, "attn_qwen7b_tokc.bin", NULL, NULL, 0 },  // Qwen2.5-7B (sched4, ROWVIS; docs/attention-gpu.md section 13)
    { 128, 2, 8, 2, 2, 0, "attn_qwen3_tokc.bin", NULL, NULL, 0 },   // Qwen3 0.6B/1.7B, Qwen3-Embedding-0.6B (sched4; section 14)
};
static cl_mem dKt; static size_t sKt;
static char aerr[512];
#define AFAIL(...) do { snprintf(aerr, sizeof aerr, __VA_ARGS__); return -1; } while (0)
#define ACHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) AFAIL("%s:%d: %s = %d", __FILE__, __LINE__, #x, e_); } while (0)

const char *gen9blas_attn_error(void) { return aerr; }

static attn_bin_t *find_bin(int D, int H, int HKV) {
    if (HKV <= 0 || H % HKV) return NULL;
    for (size_t i = 0; i < sizeof bins / sizeof bins[0]; i++) {
        attn_bin_t *b = &bins[i];
        if (b->D != D || b->HKV != HKV || b->G != H / HKV) continue;
        if (b->state == 0) {
            cl_program pr; cl_int e1 = CL_SUCCESS, e2 = CL_SUCCESS;
            b->state = -1;
            if (kcl_program(b->bin, NULL, &pr) == CL_SUCCESS) {
                b->kt = clCreateKernel(pr, "k_kt", &e1);
                b->attn = clCreateKernel(pr, "k_attn", &e2);
                if (e1 == CL_SUCCESS && e2 == CL_SUCCESS) b->state = 1;
            }
            if (b->state < 0) snprintf(aerr, sizeof aerr, "attention binary %s not in the kernel table", b->bin);
        }
        return b->state > 0 ? b : NULL;
    }
    return NULL;
}

int gen9blas_attn_supported(int D, int H, int HKV) { return find_bin(D, H, HKV) != NULL; }

static double max_flop(void) {   // per k_attn launch, see the header comment
    static double v = 0;
    if (v <= 0) { const char *e = getenv("GEN9BLAS_MAX_LAUNCH_MS"); v = (e && atof(e) > 0 ? atof(e) : 100) * 1e-3 * 200e9; }
    return v;
}

static size_t local_size(size_t global) {   // work-group 64 (section 10), smaller when 64 does not divide the grid
    for (size_t l = 64; l >= 8; l /= 2) if (global % l == 0) return l;
    return 0;
}

int gen9blas_attn_f16kv(int D, int H, int HKV, int nq, int nk, const float *Q, const uint16_t *K, size_t k_bytes,
                        const uint16_t *V, size_t v_bytes, float *O, float scale, int vis0, int dvis, int *launches) {
    attn_bin_t *b = find_bin(D, H, HKV);
    if (!b) return 1;
    const int G = H / HKV, NK = b->NK;
    // rows (token-major, row = t * G + head) split into whole RQ-row subgroups: every launch slice is a multiple of 8
    // tokens except the last, so 8 * G and nq * G must divide by RQ (G % RQ for the GQA shapes; whisper G 1, RQ 4)
    // (rowvis binaries: the last slice may have n_q*G odd; the kernel rounds the subgroup count up and clamps rows)
    if (nq <= 0 || nk <= 0 || nk % (8 * NK) || vis0 < 1 || (dvis != 0 && dvis != 1) || (8 * G) % b->RQ ||
        (!b->rowvis && ((long)nq * G) % b->RQ)) return 1;
    // asm/attn.cl without ROWVIS computes the visible key count once per subgroup, from the token of its first row:
    // right only when all RQ rows are one token (G % RQ == 0) or every token sees the same count (dvis 0); G 1 with
    // causal: unsupported. ROWVIS binaries take each row's own count (rows differ by at most dvis * (RQ - 1) keys,
    // all in the last block: RQ <= 2 asserted by attn_gen.py).
    if (G % b->RQ && dvis != 0 && !b->rowvis) return 1;
    const size_t row = (size_t)HKV * D * 2;
    if (k_bytes < (size_t)nk * row || v_bytes < (size_t)nk * row) return 1;
    if (((uintptr_t)Q | (uintptr_t)K | (uintptr_t)V | (uintptr_t)O) & 15) return 1;   // OWord block reads / writes
    const size_t ktb = (size_t)nk * row + (size_t)NK * D * 16;
    cl_int err;
    if (sKt < ktb) {
        if (dKt) clReleaseMemObject(dKt);
        dKt = clCreateBuffer(NULL, CL_MEM_READ_WRITE, ktb, NULL, &err);
        if (err != CL_SUCCESS) { dKt = NULL; sKt = 0; AFAIL("Kt buffer of %zu B: %d", ktb, err); }
        sKt = ktb;
    }
    cl_command_queue q = NULL;   // kcl: one implicit queue
    const size_t qrow = (size_t)H * D;   // floats per token in Q and O
    kcl_new_operation();
    cl_mem mK = kcl_userptr(K, k_bytes, 0), mV = kcl_userptr(V, v_bytes, 0);
    cl_mem mQ = kcl_userptr(Q, nq * qrow * 4, 0), mO = kcl_userptr(O, nq * qrow * 4, 1);
    if (!mK || !mV || !mQ || !mO) { snprintf(aerr, sizeof aerr, "kcl_userptr failed (K/V %zu B: over KCL_USERPTR_MB?)", k_bytes); return 1; }
    size_t gkt = (size_t)nk * HKV * D, lkt = 256;
    ACHECK(clSetKernelArg(b->kt, 0, sizeof mK, &mK)); ACHECK(clSetKernelArg(b->kt, 1, sizeof dKt, &dKt));
    ACHECK(clSetKernelArg(b->kt, 2, sizeof nk, &nk));
    ACHECK(clEnqueueNDRangeKernel(q, b->kt, 1, NULL, &gkt, &lkt, 0, NULL, NULL));
    const float qscale = scale * 1.4426950408889634f;   // the kernel uses exp2
    int n_launch = 0;
    for (int t0 = 0; t0 < nq; ) {
        // tokens of this launch: FLOP = 4 * H * D * sum of the visible key counts; whole multiples of 8 tokens
        double fl = 0; int t1 = t0;
        while (t1 < nq) {
            double add = 0;
            for (int t = t1; t < t1 + 8 && t < nq; t++) { long v = vis0 + (long)dvis * t; add += 4.0 * H * D * (v < nk ? v : nk); }
            if (t1 > t0 && fl + add > max_flop()) break;
            fl += add; t1 = t1 + 8 < nq ? t1 + 8 : nq;
        }
        const int nc = t1 - t0, v0 = vis0 + dvis * t0;
        cl_mem cQ = mQ, cO = mO;
        if (t0) {   // inside the objects of this operation (kcl_userptr after the first launch resolves to them)
            cQ = kcl_userptr(Q + t0 * qrow, nc * qrow * 4, 0); cO = kcl_userptr(O + t0 * qrow, nc * qrow * 4, 1);
            if (!cQ || !cO) AFAIL("kcl_userptr of a Q/O slice failed");
        }
        size_t global = (size_t)HKV * ((nc * G + b->RQ - 1) / b->RQ) * 8, local = local_size(global);
        if (!local) AFAIL("grid %zu", global);
        ACHECK(clSetKernelArg(b->attn, 0, sizeof cQ, &cQ)); ACHECK(clSetKernelArg(b->attn, 1, sizeof dKt, &dKt));
        ACHECK(clSetKernelArg(b->attn, 2, sizeof mV, &mV)); ACHECK(clSetKernelArg(b->attn, 3, sizeof cO, &cO));
        ACHECK(clSetKernelArg(b->attn, 4, sizeof nc, &nc)); ACHECK(clSetKernelArg(b->attn, 5, sizeof nk, &nk));
        ACHECK(clSetKernelArg(b->attn, 6, sizeof qscale, &qscale)); ACHECK(clSetKernelArg(b->attn, 7, sizeof v0, &v0));
        ACHECK(clSetKernelArg(b->attn, 8, sizeof dvis, &dvis));
        ACHECK(clEnqueueNDRangeKernel(q, b->attn, 1, NULL, &global, &local, 0, NULL, NULL));
        n_launch++;
        t0 = t1;
    }
    if (launches) *launches = n_launch;
    if (kcl_wait() != CL_SUCCESS) AFAIL("GPU execution error (kcl; a reset after the preemption timeout?)");
    return 0;
}
