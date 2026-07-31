from mograd import Tensor
import mograd.nn as nn
from mograd.nn import Module, Parameter


struct MLP(Module):
    var l1: nn.Linear
    var l2: nn.Linear
    var l3: nn.Linear

    def __init__(out self):
        self.l1 = nn.Linear(784, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 10)

    def __call__(mut self, x: Tensor) raises -> Tensor:
        var h = x.reshape((x.shape(0), -1))
        h = self.l1(h).relu()
        h = self.l2(h).relu()
        return self.l3(h)

    def parameters(mut self) -> List[Parameter]:
        var ps = self.l1.parameters()
        ps += self.l2.parameters()
        ps += self.l3.parameters()
        return ps^
