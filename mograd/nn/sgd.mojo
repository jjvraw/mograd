from std.memory import ArcPointer

from mograd.tensor import Tensor
from mograd.nn.linear import Linear

# ===-------------------------------------------------------------------===#
# SGD
# ===-------------------------------------------------------------------===#


struct SGD[dtype: DType = DType.float32](Copyable, Movable) where dtype.is_floating_point():
    var lr: Scalar[Self.dtype]
    var _weights: List[ArcPointer[Optional[Tensor[Self.dtype]]]]

    def __init__(out self, layers: List[Linear[Self.dtype]], lr: Scalar[Self.dtype]):
        self.lr = lr
        self._weights = []
        for layer in layers:
            self._weights.append(layer._weight)

    def params(self) -> List[Tensor[Self.dtype]]:
        var ps = List[Tensor[Self.dtype]]()
        for i in range(len(self._weights)):
            if self._weights[i][]:
                ps.append(self._weights[i][].value())
        return ps^

    def step(mut self, grads: List[Tensor[Self.dtype]]) raises:
        for i in range(len(self._weights)):
            if self._weights[i][]:
                var w = self._weights[i][].value()
                var updated = (w - grads[i] * self.lr).value()
                self._weights[i][] = Tensor[Self.dtype].from_buffer(w.device.value(), updated^)
