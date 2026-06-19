#!/usr/bin/env bash

export MOGRAD_SO="${MOGRAD_SO:-$CONDA_PREFIX/lib/mograd/libmograd_gpu.so}"

bash "$PIXI_PROJECT_ROOT/scripts/build_gpu.sh"
