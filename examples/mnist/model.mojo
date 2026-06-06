from mograd import Tensor, Device
import mograd.nn as nn


struct MLP:
    var l1: nn.Linear[]
    var l2: nn.Linear[]
    var l3: nn.Linear[]

    def __init__(out self, device: Device):
        self.l1 = nn.Linear(784, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 10)

    def __call__(mut self, x: Tensor[]) raises -> Tensor[]:
        var h = x.reshape((x.shape(0), -1))
        h = self.l1(h).relu()
        h = self.l2(h).relu()
        return self.l3(h)
