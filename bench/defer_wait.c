// LD_PRELOAD shim for bench/token_gemv.py (2026-11-13): with defer on, kcl_wait() from libgen9blas returns at once, so
// launches accumulate in kcl batch buffers (submitted when full, 64 kernels) and the GPU is waited for only at a sync.
// Valid only for USM operands (zero copy, nothing copied back after the wait). Build:
//   gcc -O2 -shared -fPIC -o bench/defer_wait.so bench/defer_wait.c -ldl
static int defer;
static int (*real_wait)(void);
// libgen9blas is dlopen-ed RTLD_LOCAL by ctypes, so RTLD_NEXT does not see it: the caller passes the real kcl_wait
// (dlsym on the library handle finds the library's own definition).
void defer_wait_init(void *real) { real_wait = (int (*)(void))real; }
void defer_wait_set(int on) { defer = on; }
int kcl_wait(void) { return defer ? 0 : real_wait(); }
int defer_wait_sync(void) { return real_wait(); }
