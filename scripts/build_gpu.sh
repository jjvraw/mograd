#!/usr/bin/env bash
set -euo pipefail
PLATFORM="$(uname -s)"
if [[ "$PLATFORM" == "Darwin" ]]; then
    CC="xcrun clang"
    if ! xcrun --find metal &>/dev/null 2>&1; then
        echo "[mograd] ERROR: Metal toolchain not found."
        exit 1
    fi
    METAL_VERSION=$(xcrun metal --version 2>&1 | head -1)
    echo "[mograd] Metal toolchain OK: $METAL_VERSION"
else
    CC="gcc"
fi
START=$(date +%s)
$CC -c -fPIC \
    -o mograd/runtime/gpu/kernels/pin.o \
    mograd/runtime/gpu/kernels/pin.c
mojo build --emit shared-lib \
    mograd/runtime/gpu/kernels/__init__.mojo \
    -o "$PIXI_PROJECT_ROOT/$MOGRAD_SO" \
    -Xlinker "$PIXI_PROJECT_ROOT/mograd/runtime/gpu/kernels/pin.o"
END=$(date +%s)

SO_PATH="$PIXI_PROJECT_ROOT/$MOGRAD_SO"
if [[ "$PLATFORM" == "Darwin" ]]; then
    SO_SIZE=$(stat -f%z "$SO_PATH")
else
    SO_SIZE=$(stat -c%s "$SO_PATH")
fi

SO_SIZE_MB=$(echo "scale=1; $SO_SIZE / 1048576" | bc)
echo "[mograd] Build complete in $((END - START))s — ${SO_SIZE_MB}MB ($SO_SIZE bytes)"
