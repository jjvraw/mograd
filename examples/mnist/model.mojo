from mograd import Tensor
import mograd.nn as nn


struct MLP:
    var l1: nn.Linear
    var l2: nn.Linear
    var l3: nn.Linear

    def __init__(out self):
        self.l1 = nn.Linear(784, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 10)

    def __call__(mut self, mut x: Tensor) raises -> Tensor:
        x = x.reshape((x.shape(0), -1))
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        return self.l3(x)
