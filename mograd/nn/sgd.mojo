from mograd.tensor import Tensor
from mograd.nn.module import ModuleParam
from mograd.nn.optimizer import Optimizer


struct SGD(Optimizer):
    var lr: Float32
    var _weights: List[ModuleParam]

    def __init__(out self, var params: List[ModuleParam], lr: Float32):
        self.lr = lr
        self._weights = params^

    def parameters(self) -> List[Tensor]:
        var ps = List[Tensor]()
        for i in range(len(self._weights)):
            if self._weights[i][]:
                ps.append(self._weights[i][].value())
        return ps^

    def set_lr(mut self, lr: Float32):
        self.lr = lr

    def step(mut self, grads: List[Tensor]) raises:
        var updates = List[Tensor]()
        var indices = List[Int]()
        for i in range(len(self._weights)):
            if self._weights[i][]:
                var w = self._weights[i][].value()
                updates.append(w - grads[i] * self.lr)
                indices.append(i)

        if len(updates) == 0:
            return

        var bufs = Tensor.values(updates)
        for j in range(len(indices)):
            var t = updates[j]
            self._weights[indices[j]][] = Tensor.from_buffer(t.device.value(), t.op.layout().copy(), bufs[j].copy())
