from mograd.tensor import Tensor, Device
from mograd.nn.modules import Module, Parameter


struct LayerNorm(Module):
    var d_model: Int
    var eps: Float32
    var weight: Parameter
    var bias: Parameter

    def __init__(out self, d_model: Int, eps: Float32 = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.weight = Parameter()
        self.bias = Parameter()

    def __call__(mut self, x: Tensor) raises -> Tensor:
        if not self.weight:
            var device = x.device.value()
            self.weight.set(Tensor.ones(device, (self.d_model,), requires_grad=True))
            self.bias.set(Tensor.zeros(device, (self.d_model,), requires_grad=True))
        var shape = x.shape()
        var mean = x.mean(axis=-1, keepdim=True).expand(shape)
        var diff = x - mean
        var var_ = (diff * diff).mean(axis=-1, keepdim=True).expand(shape)
        var x_norm = diff / (var_ + self.eps).sqrt()
        return x_norm * self.weight.tensor().expand(shape) + self.bias.tensor().expand(shape)

    def parameters(mut self) -> List[Parameter]:
        var ps = List[Parameter]()
        ps.append(self.weight)
        ps.append(self.bias)
        return ps^
