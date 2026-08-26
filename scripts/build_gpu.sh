#!/usr/bin/env bash
set -euo pipefail

log() {
    echo "[mograd] $*"
}

die() {
    echo "[mograd] ERROR: $*" >&2
    exit 1
}

: "${CONDA_PREFIX:?CONDA_PREFIX is not set. Are you inside a Pixi/conda environment?}"

MOGRAD_SO="${MOGRAD_SO:-$CONDA_PREFIX/lib/mograd/libmograd_gpu.so}"
SO_PATH="$MOGRAD_SO"
FORCE_BUILD="${MOGRAD_FORCE_BUILD:-0}"

MOGRAD_LIB_DIR="$(dirname "$SO_PATH")"
mkdir -p "$MOGRAD_LIB_DIR"

DEV_ROOT="${PIXI_PROJECT_ROOT:-}"
PKG_ROOT="$CONDA_PREFIX/share/mograd"

if [[ -n "$DEV_ROOT" && -f "$DEV_ROOT/mograd_gpu/exports.mojo" ]]; then
    SRC_ROOT="$DEV_ROOT"
    MODE="dev"
elif [[ -f "$PKG_ROOT/mograd_gpu/exports.mojo" ]]; then
    SRC_ROOT="$PKG_ROOT"
    MODE="installed"
else
    die "Could not find mograd GPU kernel sources. Checked:
  - \$PIXI_PROJECT_ROOT/mograd_gpu/exports.mojo
  - $PKG_ROOT/mograd_gpu/exports.mojo"
fi

KERNEL_SRC="$SRC_ROOT/mograd_gpu/exports.mojo"
PIN_SRC="$SRC_ROOT/mograd/runtime/gpu/kernels/pin.c"
PIN_OBJ="$MOGRAD_LIB_DIR/pin.o"
RUNTIME_DIR="$SRC_ROOT/mograd/runtime"

[[ -f "$KERNEL_SRC" ]] || die "Missing kernel source: $KERNEL_SRC"
[[ -f "$PIN_SRC" ]] || die "Missing pin source: $PIN_SRC"
[[ -d "$RUNTIME_DIR" ]] || die "Missing runtime dir: $RUNTIME_DIR"

# abi("Mojo") requires host and .so from the same toolchain: rebuild when the
# compiler version changes, not only when sources do.
BUILD_STAMP="$SO_PATH.mojo-version"
CURRENT_MOJO_VERSION="$(mojo --version 2>&1 | head -1)"

if [[ "$FORCE_BUILD" != "1" && -f "$SO_PATH" ]]; then
    NEWER_INPUT="$(
        find "$RUNTIME_DIR" \
            \( -name '*.mojo' -o -name '*.🔥' -o -name '*.c' -o -name '*.h' -o -name '*.hpp' -o -name '*.cpp' \) \
            -newer "$SO_PATH" \
            -print \
            -quit
    )"

    STAMPED_MOJO_VERSION=""
    [[ -f "$BUILD_STAMP" ]] && STAMPED_MOJO_VERSION="$(cat "$BUILD_STAMP")"

    if [[ -z "$NEWER_INPUT" && "$STAMPED_MOJO_VERSION" == "$CURRENT_MOJO_VERSION" ]]; then
        log "GPU library already up to date: $SO_PATH"
        exit 0
    fi

    if [[ "$STAMPED_MOJO_VERSION" != "$CURRENT_MOJO_VERSION" ]]; then
        log "Rebuilding because toolchain changed: '$STAMPED_MOJO_VERSION' -> '$CURRENT_MOJO_VERSION'"
    else
        log "Rebuilding because runtime input changed: $NEWER_INPUT"
    fi
elif [[ "$FORCE_BUILD" == "1" ]]; then
    log "Force rebuild requested"
fi

PLATFORM="$(uname -s)"

if [[ "$PLATFORM" == "Darwin" ]]; then
    CC="xcrun clang"

    if ! xcrun metal --version >/dev/null 2>&1; then
        die "Metal toolchain not installed. Install with: xcodebuild -downloadComponent MetalToolchain"
    fi

    METAL_VERSION="$(xcrun metal --version 2>&1 | head -1)"
    log "Metal toolchain OK: $METAL_VERSION"
else
    CC="${CC:-gcc}"

    if ! command -v "$CC" >/dev/null 2>&1; then
        die "C compiler not found: $CC"
    fi
fi

if ! command -v mojo >/dev/null 2>&1; then
    die "mojo not found in PATH"
fi

log "$CURRENT_MOJO_VERSION"
log "Building libmograd_gpu.so"
log "mode: $MODE"

START="$(date +%s)"

$CC -c -fPIC \
    -o "$PIN_OBJ" \
    "$PIN_SRC"

# TODO: Investigate further. Cold-cache builds on macOS overflow compiler worker
# thread stack (AsyncRT recursion) and die with SIGILL.
# Single-threaded compilation avoids.
MOJO_BUILD_THREADS=""
if [[ "$PLATFORM" == "Darwin" && "${CI:-}" == "true" ]]; then
    MOJO_BUILD_THREADS="--num-threads 1"
fi

# MODULAR_DISABLE_VENDOR_FALLBACK: the cuBLAS fallback allocates a fresh 32MB
# workspace per matmul; the native matmul kernels avoid it.
mojo build --emit shared-lib \
    $MOJO_BUILD_THREADS \
    -D MODULAR_DISABLE_VENDOR_FALLBACK=True \
    -I "$SRC_ROOT" \
    "$KERNEL_SRC" \
    -o "$SO_PATH" \
    -Xlinker "$PIN_OBJ"

END="$(date +%s)"

printf '%s' "$CURRENT_MOJO_VERSION" > "$BUILD_STAMP"

if [[ "$PLATFORM" == "Darwin" ]]; then
    SO_SIZE="$(stat -f%z "$SO_PATH")"
else
    SO_SIZE="$(stat -c%s "$SO_PATH")"
fi

SO_SIZE_MB="$(awk "BEGIN { printf \"%.1f\", $SO_SIZE / 1048576 }")"

log "Build complete in $((END - START))s (${SO_SIZE_MB}MB / $SO_SIZE bytes)"
