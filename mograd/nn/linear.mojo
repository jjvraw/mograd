from std.math import sqrt
from std.memory import ArcPointer

from mograd.tensor import Tensor, Device
from mograd.nn.module import Module, ModuleParam

# ===-------------------------------------------------------------------===#
# Linear
# ===-------------------------------------------------------------------===#


struct Linear(Module):
    var in_features: Int
    var out_features: Int
    var use_bias: Bool
    var _weight: ArcPointer[Optional[Tensor]]
    var _bias: ArcPointer[Optional[Tensor]]

    def __init__(out self, in_features: Int, out_features: Int, bias: Bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self._weight = ArcPointer(Optional[Tensor](None))
        self._bias = ArcPointer(Optional[Tensor](None))

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
            if self.use_bias:
                self._bias[] = Tensor.uniform(
                    x.device.value(),
                    (self.out_features,),
                    low=-bound,
                    high=bound,
                    seed=seed + 1,
                    requires_grad=True,
                )
        var rank = x.rank()
        var x2d = x.reshape(x.op.layout().flatten(0, rank - 2))
        var out2d = x2d @ self._weight[].value().transpose()
        var out = out2d.reshape(x.op.layout().with_last_dim(self.out_features))
        if self.use_bias:
            out = out + self._bias[].value().expand(out.shape())
        return out

    def parameters(mut self) -> List[ModuleParam]:
        var ps = List[ModuleParam]()
        ps.append(self._weight)
        if self.use_bias:
            ps.append(self._bias)
        return ps^
