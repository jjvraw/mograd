<p align="center">
  <img src="./assets/logo.png" alt="Project Logo" width="200"/>
</p>

<h1 align="center">mograd</h1>

> [!WARNING]
> Early-stage development.

Mograd is an *Autograd Tensor Library* that leverages Mojo's zero-cost abstractions,
providing: 

- **Imperative Tensor API**
    - Eager-style composition, lazy realization
- **Thin Graph IR** with pattern-matching rewrite semantics
    - Enables built-in graph transformations, fusions, and extensible through user-defined
    rewrites for custom kernels
- **Precompiled Kernel Library** for eager dispatch
    - Custom Mojo, and [Modular MAX](https://www.modular.com/open-source-max) powered 
    kernels

---

Define a Neural Network, (see [examples/](./examples) for complete models with training): 

```mojo
from mograd import Tensor
import mograd.nn as nn
from mograd.nn import Module, Parameter

struct MLP(Module):
    var l1: nn.Linear
    var norm: nn.LayerNorm
    var l2: nn.Linear
    var l3: nn.Linear

    def __init__(out self):
        self.l1 = nn.Linear(4096, 4096)
        self.norm = nn.LayerNorm(4096)
        self.l2 = nn.Linear(4096, 4096)
        self.l3 = nn.Linear(4096, 10)

    def __call__(mut self, x: Tensor) raises -> Tensor:
        var h = self.norm(self.l1(x).relu())
        h = self.l2(h).relu()
        return self.l3(h)

    def parameters(mut self) -> List[Parameter]:
        var ps = self.l1.parameters()
        ps += self.norm.parameters()
        ps += self.l2.parameters()
        ps += self.l3.parameters()
        return ps^
```

Benchmark a single forward pass with `DEBUG_MOGRAD` flag:

```
 mojo -D MOGRAD_DEBUG=3 demo.mojo # =4 reveals OpGraph

    kernel                                time        tput
RANDN (2048,4096)
  ↳ mograd_randn                       859.8µs   39.0 GB/s
UNIFORM (4096,4096)
  ↳ mograd_uniform                      92.2µs  727.3 GB/s
UNIFORM (4096)
  ↳ mograd_uniform                       9.7µs    1.6 GB/s
MATMUL_BIAS_BT (2048,4096)
  ↳ mograd_matmul_bias_bt               58.7ms    2.2 GB/s
RELU (2048,4096)
  ↳ mograd_relu                        123.0µs  545.5 GB/s
FULL (4096)
  ↳ mograd_full                         29.5µs    0.5 GB/s
FULL (4096)
  ↳ mograd_full                          7.3µs    2.2 GB/s
LAYER_NORM (2048,4096)
  ↳ mograd_layer_norm_fwd              163.8µs  409.7 GB/s
UNIFORM (4096,4096)
  ↳ mograd_uniform                      65.0µs 1031.7 GB/s
UNIFORM (4096)
  ↳ mograd_uniform                       8.8µs    1.8 GB/s
MATMUL_BIAS_BT (2048,4096)
  ↳ mograd_matmul_bias_bt                1.0ms  122.0 GB/s
RELU (2048,4096)
  ↳ mograd_relu                         69.9µs  959.2 GB/s
UNIFORM (10,4096)
  ↳ mograd_uniform                       9.6µs   16.9 GB/s
UNIFORM (10)
  ↳ mograd_uniform                      42.0µs    0.0 GB/s
MATMUL_BIAS_BT (2048,10)
  ↳ mograd_matmul_bias_bt                2.5ms   13.4 GB/s

Σ run: 15 kernels  63.9ms  = 63.8ms gpu + 98.7µs dispatch
```

## Design

The intention is to keep the graph IR between the Tensor API and kernels thin such that
`PatternMatcher` drives all graph transformations through rewrite passes. For instance, 
for gradient computation, we may define rewrite rules as: 

```mojo
var pm = PatternMatcher[GradFn]([
    Rule(Pat(OpType.MUL), mul_grad),
    Rule(Pat(OpType.ADD), add_grad),
    Rule(Pat(OpType.SOFTMAX), softmax_grad),
    ...
])

def mul_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [node.src(1) * upstream, node.src(0) * upstream]


def add_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [upstream] * len(node.srcs())

...
```

A similar approach is used to enable fusions, simplifications and runtime dispatching.
Rewrite rules are designed to be extensible through external, user-defined rules. This 
allows everything to remain in Mojo, from model definition, to kernels, model profiling, 
and graph rewrites. A larger initiative is for Mograd to support rule sets for common
architectures. This approach is infamously known for its scaling issues, but it is
motivated by evident tension between framework compilers, external operators, and custom
rewrite passes in current inference and training engines. The spirit is close to
[TVM Unity](https://tvm.apache.org/2021/12/15/tvm-unity). That being, remove boundaries 
between the operator graph, transformations, and the kernels. Mograd chases that idea in
pure Mojo.
