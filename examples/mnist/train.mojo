from mograd import Tensor, Device
from mograd.data import mnist
import mograd.nn as nn
from model import MLP


def accuracy(mut model: MLP, x: Tensor[DType.float32], y: Tensor[DType.float32], batch_size: Int) raises -> Float32:
    var n = x.shape(0)
    var avg_acc: Float32 = 0.0
    for step in range(n // batch_size):
        var xb = x[step * batch_size : (step + 1) * batch_size]
        var yb = y[step * batch_size : (step + 1) * batch_size]
        avg_acc += (model(xb).argmax() == yb).mean().item()
    return avg_acc / Float32(n // batch_size)


def main() raises:
    var device = Device()
    var data = mnist(ctx)

    var model = MLP(ctx)
    var opt = nn.SGD([model.l1, model.l2, model.l3], lr=Float32(0.01))

    var batch_size = 32
    var n_steps = 60000 // batch_size

    for step in range(n_steps):
        var x = data.x_train[step * batch_size : (step + 1) * batch_size]
        var y = data.y_train[step * batch_size : (step + 1) * batch_size]

        var logits = model(x)
        var loss = logits.cross_entropy(y.one_hot(10).cast[DType.float32]())

        var params = opt.params()
        var grads = loss.gradient(params)
        opt.step(grads)

        if step % 100 == 0:
            print("step", step, "loss:", loss.item())

        if step % 500 == 0:
            var acc = accuracy(model, data.x_test, data.y_test, batch_size)
            print("  val_acc:", acc, "\n")

    var acc = accuracy(model, data.x_test, data.y_test, batch_size)
    print("\n" + "=" * 10)
    print("\nfinal val_acc:", acc)
