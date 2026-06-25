from std.memory import ArcPointer

from mograd.tensor import Tensor
from mograd.nn.linear import Linear

# ===-------------------------------------------------------------------===#
# SGD
# ===-------------------------------------------------------------------===#


struct SGD(Copyable, Movable):
    var lr: Float32
    var _weights: List[ArcPointer[Optional[Tensor]]]

    def __init__(out self, layers: List[Linear], lr: Float32):
        self.lr = lr
        self._weights = []
        for layer in layers:
            self._weights.append(layer._weight)
            if layer.use_bias:
                self._weights.append(layer._bias)

    def params(self) -> List[Tensor]:
        var ps = List[Tensor]()
        for i in range(len(self._weights)):
            if self._weights[i][]:
                ps.append(self._weights[i][].value())
        return ps^

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
