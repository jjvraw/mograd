from std.math import sqrt

from mograd.tensor import Tensor, Device
from mograd.nn.modules import Module, Parameter

# ===-------------------------------------------------------------------===#
# Linear
# ===-------------------------------------------------------------------===#


struct Linear(Module):
    var in_features: Int
    var out_features: Int
    var use_bias: Bool
    var weight: Parameter
    var bias: Parameter

    def __init__(out self, in_features: Int, out_features: Int, bias: Bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.weight = Parameter()
        self.bias = Parameter()

    def __call__(mut self, x: Tensor) raises -> Tensor:
        if not self.weight:
            if not x.device:
                raise Error("Linear requires a device context on first call")
            var bound = Float32(sqrt(Float32(6) / Float32(self.in_features)))
            self.weight.set(
                Tensor.uniform(
                    x.device.value(),
                    (self.out_features, self.in_features),
                    low=-bound,
                    high=bound,
                    requires_grad=True,
                )
            )
            if self.use_bias:
                self.bias.set(
                    Tensor.uniform(
                        x.device.value(),
                        (self.out_features,),
                        low=-bound,
                        high=bound,
                        requires_grad=True,
                    )
                )
        var rank = x.rank()
        var x2d = x.reshape(x.op.layout().flatten(0, rank - 2))
        var out2d = x2d @ self.weight.tensor().transpose()
        var out = out2d.reshape(x.op.layout().with_last_dim(self.out_features))
        if self.use_bias:
            out = out + self.bias.tensor().expand(out.shape())
        return out

    def parameters(mut self) -> List[Parameter]:
        var ps = List[Parameter]()
        ps.append(self.weight)
        if self.use_bias:
            ps.append(self.bias)
        return ps^
