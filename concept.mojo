import mograd as mg
from mograd.nn import Linear
from std.gpu.host import DeviceContext
from std.sys import has_accelerator


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()

    # weight: [out=2, in=3], x: [batch=2, in=3]
    var weight = mg.Tensor(
        ctx, [1.0, 0.0, -1.0, 0.0, 1.0, 0.0], [2, 3], requires_grad=True
    )
    var layer = Linear(weight)

    var x = mg.Tensor(
        ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True
    )

    var y = layer(x)
    print("y = x @ W.T  (shape [2,2]):")
    _print_buffer(y.value())

    var grads = y.gradient(x, weight)
    print("\nd_y/dx:")
    _print_buffer(grads[0].value())
    print("\nd_y/dW:")
    _print_buffer(grads[1].value())


def _print_buffer(b: mg.Buffer) raises:
    with b.buf().map_to_host() as mapped:
        var ptr = mapped.unsafe_ptr()
        for i in range(b.size):
            print(ptr[i])
