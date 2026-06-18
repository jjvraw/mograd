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

### Design

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
Rewrite rules are designed to be extensible through external, user-defined rules. A larger
initiative is to support rule sets for common architectures. This approach is infamously
known for its scaling issues, but it is motivated by evident tension between framework
compilers, external operators, and custom rewrite passes in current inference and training
engines. The spirit is close to [TVM Unity](https://tvm.apache.org/2021/12/15/tvm-unity).
That being, remove boundaries between the operator graph, transformations, and the
kernels. Mograd chases that idea in pure Mojo, with eager PyTorch-like ergonomics.
