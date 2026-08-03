from mograd.tensor import Tensor
from mograd.nn.modules import Parameter
from mograd.nn.optim import Optimizer


struct SGD(Optimizer):
    var lr: Float32
    var weights: List[Parameter]

    def __init__(out self, var params: List[Parameter], lr: Float32):
        self.lr = lr
        self.weights = params^

    def parameters(self) -> List[Tensor]:
        var ps = List[Tensor]()
        for ref w in self.weights:
            if w:
                ps.append(w.tensor())
        return ps^

    def set_lr(mut self, lr: Float32):
        self.lr = lr

    def step(mut self, grads: List[Tensor]) raises:
        var updates = List[Tensor]()
        var indices = List[Int]()
        # `grads` lines up with `parameters()`, which skips unset slots, so it is
        # indexed by its own running counter rather than the slot index.
        var grad_idx = 0
        for i in range(len(self.weights)):
            if self.weights[i]:
                var w = self.weights[i].tensor()
                updates.append(w - grads[grad_idx] * self.lr)
                grad_idx += 1
                indices.append(i)

        if len(updates) == 0:
            return

        var bufs = Tensor.values(updates)
        for j in range(len(indices)):
            var t = updates[j]
            self.weights[indices[j]].set(Tensor.from_buffer(t.device.value(), t.op.layout().copy(), bufs[j].copy()))
