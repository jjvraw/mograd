from mograd import Tensor, DeviceContext
from mograd.data import mnist
from model import MLP


def main() raises:
    var ctx = DeviceContext()

    var data = mnist(ctx)

    var batch_size = 4
    var dummy = List[Float32]()
    for i in range(batch_size * 784):
        dummy.append(Float32(i % 256) / Float32(255))
    var x = Tensor(ctx, dummy, [batch_size, 784])

    var model = MLP()
    var logits = model(x)
    print(
        "logits shape:", logits.value().shape[0], "x", logits.value().shape[1]
    )

    var loss = logits.sum()
    var grads = loss.gradient(
        model.l1._weight.value(),
        model.l2._weight.value(),
        model.l3._weight.value(),
    )
    for i in range(len(grads)):
        var g = grads[i].value()
        print("grad", i, "shape:", g.shape[0], "x", g.shape[1])
