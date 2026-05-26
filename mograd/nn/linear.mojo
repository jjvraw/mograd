from std.math import sqrt

from mograd.tensor import Tensor

# ===-------------------------------------------------------------------===#
# Linear
# ===-------------------------------------------------------------------===#


struct Linear(Copyable, Movable):
    var in_features: Int
    var out_features: Int
    var _weight: Optional[Tensor]  # [out_features, in_features], lazy init

    def __init__(out self, in_features: Int, out_features: Int):
        self.in_features = in_features
        self.out_features = out_features
        self._weight = None

    def __call__(mut self, x: Tensor) raises -> Tensor:
        if not self._weight:
            if not x.ctx:
                raise Error("Linear requires a device context on first call")
            var bound = sqrt(Float32(6) / Float32(self.in_features))
            self._weight = Tensor.uniform(
                x.ctx.value(),
                [self.out_features, self.in_features],
                low=-bound,
                high=bound,
                seed=UInt32(self.out_features * self.in_features),
                requires_grad=True,
            )
        return x @ self._weight.value().transpose()
