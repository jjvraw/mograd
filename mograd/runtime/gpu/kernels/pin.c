/*
 * Mark this .so as RTLD_NODELETE at load time.
 *
 * mojo run KGEN_CompilerRT_DestroyGlobals holds function pointers into this
 * .so and is called after user code cleanup (including OwnedDLHandle::dlclose).
 * If dlclose has already unmapped the library, KGEN crashes with SIGSEGV.
 * RTLD_NODELETE prevents dlclose from unmapping the library even when the refcount
 * hits zero, so it stays mapped until process exit.
 *
 */
#define _GNU_SOURCE
#include <dlfcn.h>

__attribute__((constructor))
static void _mograd_pin(void) {
    Dl_info info;
    if (dladdr(&_mograd_pin, &info) && info.dli_fname) {
        void *h = dlopen(info.dli_fname, RTLD_NOW | RTLD_NODELETE);
        if (h) dlclose(h);
    }
}
