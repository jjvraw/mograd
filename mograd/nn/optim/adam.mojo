from mograd.tensor import Tensor
from mograd.nn.modules import Parameter
from mograd.nn.optim import Optimizer


def _adam_step[
    decoupled_wd: Bool
](
    mut weights: List[Parameter],
    mut m: List[Optional[Tensor]],
    mut v: List[Optional[Tensor]],
    grads: List[Tensor],
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    t: Int,
) raises:
    var bc1 = Float32(1.0) - beta1**t
    var bc2 = Float32(1.0) - beta2**t

    var new_ms = List[Tensor]()
    var new_vs = List[Tensor]()
    var updates = List[Tensor]()
    var indices = List[Int]()

    var grad_idx = 0
    for i in range(len(weights)):
        if not weights[i]:
            continue

        var w = weights[i].tensor()
        var g = grads[grad_idx]
        grad_idx += 1

        comptime if not decoupled_wd:
            if weight_decay > 0.0:
                g = g + w * weight_decay

        if not m[i]:
            m[i] = w.zeros_like()
            v[i] = w.zeros_like()

        var new_m = m[i].value() * beta1 + g * (Float32(1.0) - beta1)
        var new_v = v[i].value() * beta2 + (g * g) * (Float32(1.0) - beta2)
        var m_hat = new_m * (Float32(1.0) / bc1)
        var v_hat = new_v * (Float32(1.0) / bc2)
        var adam_update = m_hat / (v_hat.sqrt() + eps) * lr

        comptime if decoupled_wd:
            var w_decayed = w * (Float32(1.0) - lr * weight_decay) if weight_decay > 0.0 else w
            updates.append(w_decayed - adam_update)
        else:
            updates.append(w - adam_update)

        new_ms.append(new_m)
        new_vs.append(new_v)
        indices.append(i)

    if len(updates) == 0:
        return

    # Single GPU round-trip: [w0..., m0..., v0...]
    var n = len(indices)
    var all_tensors = List[Tensor]()
    for j in range(n):
        all_tensors.append(updates[j])
    for j in range(n):
        all_tensors.append(new_ms[j])
    for j in range(n):
        all_tensors.append(new_vs[j])
    var bufs = Tensor.values(all_tensors)
    for j in range(n):
        var idx = indices[j]
        weights[idx].set(Tensor.from_buffer(updates[j].device.value(), updates[j].op.layout().copy(), bufs[j].copy()))
        m[idx] = Tensor.from_buffer(new_ms[j].device.value(), new_ms[j].op.layout().copy(), bufs[n + j].copy())
        v[idx] = Tensor.from_buffer(new_vs[j].device.value(), new_vs[j].op.layout().copy(), bufs[2 * n + j].copy())


struct Adam(Optimizer):
    var lr: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var weight_decay: Float32
    var weights: List[Parameter]
    var m: List[Optional[Tensor]]
    var v: List[Optional[Tensor]]
    var t: Int

    def __init__(
        out self,
        var params: List[Parameter],
        lr: Float32 = 1e-3,
        beta1: Float32 = 0.9,
        beta2: Float32 = 0.999,
        eps: Float32 = 1e-8,
        weight_decay: Float32 = 0.0,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.weights = params^
        self.m = List[Optional[Tensor]]()
        self.v = List[Optional[Tensor]]()
        for _ in range(len(self.weights)):
            self.m.append(Optional[Tensor](None))
            self.v.append(Optional[Tensor](None))

    def parameters(self) -> List[Tensor]:
        var ps = List[Tensor]()
        for w in self.weights:
            if w:
                ps.append(w.tensor())
        return ps^

    def step(mut self, grads: List[Tensor]) raises:
        self.t += 1
        _adam_step[False](
            self.weights,
            self.m,
            self.v,
            grads,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.weight_decay,
            self.t,
        )

    def set_lr(mut self, lr: Float32):
        self.lr = lr


struct AdamW(Optimizer):
    var lr: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var weight_decay: Float32
    var weights: List[Parameter]
    var m: List[Optional[Tensor]]
    var v: List[Optional[Tensor]]
    var t: Int

    def __init__(
        out self,
        var params: List[Parameter],
        lr: Float32 = 1e-3,
        beta1: Float32 = 0.9,
        beta2: Float32 = 0.999,
        eps: Float32 = 1e-8,
        weight_decay: Float32 = 0.01,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.weights = params^
        self.m = List[Optional[Tensor]]()
        self.v = List[Optional[Tensor]]()
        for _ in range(len(self.weights)):
            self.m.append(Optional[Tensor](None))
            self.v.append(Optional[Tensor](None))

    def parameters(self) -> List[Tensor]:
        var ps = List[Tensor]()
        for w in self.weights:
            if w:
                ps.append(w.tensor())
        return ps^

    def step(mut self, grads: List[Tensor]) raises:
        self.t += 1
        _adam_step[True](
            self.weights,
            self.m,
            self.v,
            grads,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.weight_decay,
            self.t,
        )

    def set_lr(mut self, lr: Float32):
        self.lr = lr
