#!/usr/bin/env bash
set -euo pipefail

PLATFORM="$(uname -s)"

if [[ "$PLATFORM" == "Darwin" ]]; then
    CC="xcrun clang"
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
echo "[mograd] Build complete in $((END - START))s"
