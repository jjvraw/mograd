from mograd import Tensor, DeviceContext
from model import MLP


def main() raises:
    var ctx = DeviceContext()

    var batch_size = 4
    var pixels = 784
    var data = List[Float32]()
    for i in range(batch_size * pixels):
        data.append(Float32(i % 256) / Float32(255))
    var x = Tensor(ctx, data, [batch_size, pixels])

    var model = MLP()
    var logits = model(x)
    var result = logits.value()

    print("logits shape:", result.shape[0], "x", result.shape[1])

    var loss = logits.sum()
    var grads = loss.gradient(
        model.l1._weight.value(),
        model.l2._weight.value(),
        model.l3._weight.value(),
    )
    for i in range(len(grads)):
        var g = grads[i].value()
        print("grad", i, "shape:", g.shape[0], "x", g.shape[1])
