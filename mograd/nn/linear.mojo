from std.math import sqrt
from std.memory import ArcPointer

from mograd.tensor import Tensor, Device

# ===-------------------------------------------------------------------===#
# Linear
# ===-------------------------------------------------------------------===#


struct Linear(Copyable, ImplicitlyCopyable, Movable):
    var in_features: Int
    var out_features: Int
    var _weight: ArcPointer[Optional[Tensor]]

    def __init__(out self, in_features: Int, out_features: Int):
        self.in_features = in_features
        self.out_features = out_features
        self._weight = ArcPointer(Optional[Tensor](None))

    def __call__(mut self, x: Tensor) raises -> Tensor:
        if not self._weight[]:
            if not x.device:
                raise Error("Linear requires a device context on first call")
            var bound = Float32(sqrt(Float32(6) / Float32(self.in_features)))
            var seed = UInt32(self.out_features * self.in_features)
            self._weight[] = Tensor.uniform(
                x.device.value(),
                (self.out_features, self.in_features),
                low=-bound,
                high=bound,
                seed=seed,
                requires_grad=True,
            )
        return x @ self._weight[].value().transpose()
