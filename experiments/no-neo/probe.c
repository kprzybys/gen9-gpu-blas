// i915 interface probe (only reads parameters, submits nothing to the GPU).
#include <drm/i915_drm.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

static int fd;
static void par(const char *n, int p) {
    int v = -1; struct drm_i915_getparam g = { .param = p, .value = &v };
    int r = ioctl(fd, DRM_IOCTL_I915_GETPARAM, &g);
    printf("%-36s %s%d\n", n, r ? "(error) " : "", v);
}
int main(void) {
    fd = open("/dev/dri/renderD128", O_RDWR);
    if (fd < 0) { perror("renderD128"); return 1; }
    par("CHIPSET_ID", I915_PARAM_CHIPSET_ID);
    par("REVISION", I915_PARAM_REVISION);
    par("EU_TOTAL", I915_PARAM_EU_TOTAL);
    par("SUBSLICE_TOTAL", I915_PARAM_SUBSLICE_TOTAL);
    par("HAS_EXEC_SOFTPIN", I915_PARAM_HAS_EXEC_SOFTPIN);
    par("HAS_EXEC_NO_RELOC", I915_PARAM_HAS_EXEC_NO_RELOC);
    par("HAS_EXEC_HANDLE_LUT", I915_PARAM_HAS_EXEC_HANDLE_LUT);
    par("HAS_EXEC_FENCE", I915_PARAM_HAS_EXEC_FENCE);
    par("HAS_EXEC_FENCE_ARRAY", I915_PARAM_HAS_EXEC_FENCE_ARRAY);
    par("HAS_EXEC_TIMELINE_FENCES", I915_PARAM_HAS_EXEC_TIMELINE_FENCES);
    par("HAS_USERPTR_PROBE", I915_PARAM_HAS_USERPTR_PROBE);
    par("MMAP_VERSION", I915_PARAM_MMAP_VERSION);
    par("MMAP_GTT_VERSION", I915_PARAM_MMAP_GTT_VERSION);
    par("HAS_LLC", I915_PARAM_HAS_LLC);
    par("CMD_PARSER_VERSION", I915_PARAM_CMD_PARSER_VERSION);
    par("HAS_CONTEXT_ISOLATION", I915_PARAM_HAS_CONTEXT_ISOLATION);
    par("CS_TIMESTAMP_FREQUENCY", I915_PARAM_CS_TIMESTAMP_FREQUENCY);
    par("HAS_SCHEDULER", I915_PARAM_HAS_SCHEDULER);
    par("HAS_WAIT_TIMEOUT", I915_PARAM_HAS_WAIT_TIMEOUT);
    par("HAS_SECURE_BATCHES", I915_PARAM_HAS_SECURE_BATCHES);
    par("HAS_EXEC_BATCH_FIRST", I915_PARAM_HAS_EXEC_BATCH_FIRST);
    par("HAS_EXEC_ASYNC", I915_PARAM_HAS_EXEC_ASYNC);
    struct drm_i915_gem_context_param cp = { .ctx_id = 0, .param = I915_CONTEXT_PARAM_GTT_SIZE };
    int r = ioctl(fd, DRM_IOCTL_I915_GEM_CONTEXT_GETPARAM, &cp);
    printf("%-36s %s%llu (2^%d)\n", "CONTEXT GTT_SIZE", r ? "(error) " : "", (unsigned long long)cp.value, 63 - __builtin_clzll(cp.value ? cp.value : 1));
    struct drm_i915_gem_get_aperture ap = { 0 };
    r = ioctl(fd, DRM_IOCTL_I915_GEM_GET_APERTURE, &ap);
    printf("%-36s %s%llu MB\n", "APERTURE (GGTT)", r ? "(error) " : "", (unsigned long long)ap.aper_size >> 20);
    return 0;
}
