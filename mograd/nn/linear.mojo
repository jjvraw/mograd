from std.math import sqrt
from std.memory import ArcPointer

from mograd.tensor import Tensor

# ===-------------------------------------------------------------------===#
# Linear
# ===-------------------------------------------------------------------===#


struct Linear[dtype: DType = DType.float32](Copyable, ImplicitlyCopyable, Movable) where dtype.is_floating_point():
    var in_features: Int
    var out_features: Int
    var _weight: ArcPointer[Optional[Tensor[Self.dtype]]]  # [out_features, in_features], lazy init

    def __init__(out self, in_features: Int, out_features: Int):
        self.in_features = in_features
        self.out_features = out_features
        self._weight = ArcPointer(Optional[Tensor[Self.dtype]](None))

    def __call__(mut self, x: Tensor[Self.dtype]) raises -> Tensor[Self.dtype]:
        if not self._weight[]:
            if not x.ctx:
                raise Error("Linear requires a device context on first call")
            var bound = Float32(sqrt(Float32(6) / Float32(self.in_features)))
            var seed = UInt32(self.out_features * self.in_features)
            self._weight[] = Tensor[Self.dtype].uniform(
                x.ctx.value(),
                (self.out_features, self.in_features),
                low=-bound,
                high=bound,
                seed=seed,
                requires_grad=True,
            )
        return x @ self._weight[].value().transpose()
