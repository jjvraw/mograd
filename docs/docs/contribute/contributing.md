---
title: Contributing
sidebar_label: Contributing
sidebar_position: 1
description: Set up a mograd checkout, run the tests and benchmarks, and find your way around the tree.
---

# Contributing

## Status

`mograd` is too young to be a released package, and there will not be one until an
_undecided_ milestone has been reached. Thus, this page outlines the general direction of
the project, should you want to contribute.

## Developer Environment and Pixi Tasks

Firstly, everything runs through [pixi](https://pixi.sh). Mojo and MAX versions are pinned
in `pixi.toml`.

```bash
git clone https://github.com/jjvraw/mograd.git && cd mograd
pixi install
```

Currently, only GPUs are supported. The GPU kernel library, `libmograd_gpu.so`, is a
shared object built from `mograd/runtime/gpu`, see [available kernels](../api/internals/runtime/gpu/kernels/index.md).
All kernels are pre-compiled, enabling eager dispatch / reducing compile time:

```bash
pixi run ensure-gpu     # build if missing
pixi run build-gpu      # force a rebuild
```

`pixi task list` displays the full list of tasks, inclusive of tests. There are currently
three broad _testable_ categories:

```bash
pixi run test-gpu        # autograd, nn, ops, data, non-contiguous, scheduler (GPU backend)
pixi run test-graph      # pattern matcher, rewrites, layout, custom rules (no device)
pixi run test-kernels    # mograd written kernels (in contrast from upstream MAX kernels)

pixi run test-all        # test-gpu + test-graph + examples
```

Running tests is probably a good place to start.

If benchmarking, kernel engineering, tuning, is your thing, `libmograd_gpu` consists
of (1) Kernels composed of upstream MAX packages ([linalg](https://docs.modular.com/api/mojo/linalg/)
and [nn](https://docs.modular.com/api/mojo/nn/)), (2) As for any _training-related_ 
kernels, we have to author our own. Kernel implementations live in [mograd/runtime/gpu](https://github.com/jjvraw/mograd/tree/main/mograd/runtime/gpu).
All benchmark tasks can be found via:
```
pixi task list 2>&1 | tr ',' '\n' | grep "bench" | sed 's/^ *//'
```
Though, benchmark coverage is currently under developed.

## Design

As a brief motivation, `mograd` very much started as (perhaps still is) a project to
tinker with. However, the workflow is motivated by the extensive use of external kernels 
for both serving and training alongside SOTA framework compilers. This leads to 
tension[^tension] between external kernels and compilers, where graph IR is partitioned
into optimised regions. In turn, what's seen is custom pattern-matching passes and 
benchmarking between generated and external kernels. If you view this as a problem, and 
approach said problem from first principles, you'd probably end up in the direction of
[MAX](https://www.modular.com/open-source/max). `mograd` is more so of a naive approach. 

If a spectrum exists between eagerly dispatched, generated, and handwritten models, and
assuming position along that spectrum scales with performance, the window where fully
generated models suffice appears to be narrower than the window where generated and
handwritten kernels coexist. If we prioritise eager dispatch and imperative ergonomics, 
have a thin transformable IR sitting between tensors and kernels, and put equal effort 
into authoring common rewrite passes as into compiler engineering, perhaps that’s a way
forward. The last variable, hardware heterogeneity, is then handled by Mojo. 

Much of the following decisions, stated below, stem from this.

### Laziness
A [`Tensor`](../api/public/tensor) holds no data. It is a refcounted handle to a node in a 
graph. It follows, `mograd` is lazy, and execution is deferred until the graph is 
realised.

For example:

```mojo
from mograd import Tensor, Device

def main() raises: 
    device = Device()
    a, b = Tensor.randn(device, (8,8)), Tensor.randn(device, (8, 8))
    c = a + b # or a.add(b)
    print(c)
```

```
Op(ADD, float32, shape=(8,8), src=(
  Op(RANDN, float32, shape=(8,8), src=()),
  Op(RANDN, float32, shape=(8,8), src=())
))
```

`Tensor` holds no data, for instance, consider the implementation of [`add`](../api/public/tensor#tensor-add).

```mojo
def add(self, other: Self) -> Self:
    return Self(self.device, self.op + other.op, self.requires_grad or other.requires_grad)
```

An [`Op`](../api/internals/op#op) is our node: what to compute (`Op.op_type`), what it
produces (`Op.layout`, `Op.dtype`), what from (`Op.srcs`), any parameters (`Op.attrs`),
and where the results lands (`Op.buf`).

[`OpRef`](../api/internals/op#opref) is a refcounted pointer and the identity 
(hashed by address) of an `Op`. Both `Tensor` and `Op.srcs` hold `OpRef`, so the graph
is a DAG of shared nodes and idenity is sharing.

An [`OpType`](../api/internals/op#optype) is a list of `mograd` primitives. Ideally, we 
want this to be as granular as possible. However, in some cases it doesn't make sense to
express computations as a chain of primitives just to simplify, or fuse, during runtime.
When we introduce CPU runtime, the granularity may change, proceeded with hardware-target
specific simplifications. This rough idea is already present, see [GPU-specific rewrites](../api/internals/runtime/gpu/rewrites).

As hinted already, laziness enables us to mimick eager dispatch, with the slight deviation
of calling a realisation operation on tensors, while also perfoming graph transformations.

### Graph Transformations




### Runtime


[^tension]: [vllm-project/vllm#36066](https://github.com/vllm-project/vllm/issues/36066),
        [vllm-project/vllm#24629](https://github.com/vllm-project/vllm/issues/24629),
        [sgl-project/sglang#21855](https://github.com/sgl-project/sglang/issues/21855),
        [sgl-project/sglang#21855](https://github.com/sgl-project/sglang/issues/10118)
