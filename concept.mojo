import mograd as mg
from std.gpu.host import DeviceContext
from std.sys import has_accelerator


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"

    var ctx = DeviceContext()
    var x = mg.Tensor(ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
    var w = mg.Tensor(ctx, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [2, 3], requires_grad=True)
    var out = x * w + w

    print("Graph:")
    print(out)

    print("\nResult (x * w + w):")
    _print_buffer(out.value())

    var grads = out.gradient(x, w)
    print("\nd_out/dx:")
    _print_buffer(grads[0].value())
    print("\nd_out/dw:")
    _print_buffer(grads[1].value())


def _print_buffer(b: mg.Buffer) raises:
    with b.buf().map_to_host() as mapped:
        var ptr = mapped.unsafe_ptr()
        for i in range(b.size):
            print(ptr[i])
