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

    def params(self) -> List[Tensor]:
        var ps = List[Tensor]()
        for i in range(len(self._weights)):
            if self._weights[i][]:
                ps.append(self._weights[i][].value())
        return ps^

    def step(mut self, grads: List[Tensor]) raises:
        for i in range(len(self._weights)):
            if self._weights[i][]:
                var w = self._weights[i][].value()
                var updated = (w - grads[i] * self.lr).value()
                self._weights[i][] = Tensor.from_buffer(w.ctx.value(), updated^)
