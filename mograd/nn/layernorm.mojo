from std.memory import ArcPointer

from mograd.tensor import Tensor, Device
from mograd.nn.module import Module, ModuleParam


struct LayerNorm(Module):
    var d_model: Int
    var eps: Float32
    var _weight: ArcPointer[Optional[Tensor]]
    var _bias: ArcPointer[Optional[Tensor]]

    def __init__(out self, d_model: Int, eps: Float32 = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self._weight = ArcPointer(Optional[Tensor](None))
        self._bias = ArcPointer(Optional[Tensor](None))

    def __call__(mut self, x: Tensor) raises -> Tensor:
        if not self._weight[]:
            var device = x.device.value()
            self._weight[] = Tensor.ones(device, (self.d_model,), requires_grad=True)
            self._bias[] = Tensor.zeros(device, (self.d_model,), requires_grad=True)
        var shape = x.shape()
        var mean = x.mean(axis=-1, keepdim=True).expand(shape)
        var diff = x - mean
        var var_ = (diff * diff).mean(axis=-1, keepdim=True).expand(shape)
        var x_norm = diff / (var_ + self.eps).sqrt()
        return x_norm * self._weight[].value().expand(shape) + self._bias[].value().expand(shape)

    def parameters(mut self) -> List[ModuleParam]:
        var ps = List[ModuleParam]()
        ps.append(self._weight)
        ps.append(self._bias)
        return ps^
