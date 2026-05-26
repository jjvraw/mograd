import mograd as mg
from std.gpu.host import DeviceContext
from std.sys import has_accelerator


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()

    # [2, 3] @ [3, 2] -> [2, 2]
    var a = mg.Tensor(
        ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True
    )
    var b = mg.Tensor(
        ctx, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2], requires_grad=True
    )

    var c = a @ b
    print("a @ b ([2,3] @ [3,2] -> [2,2], expected [58,64,139,154]):")
    _print_buffer(c.value())

    print("\ntranspose(a) ([2,3] -> [3,2], expected [1,4,2,5,3,6]):")
    _print_buffer(a.transpose().value())

    var grads = c.gradient(a, b)
    print("\nd_c/da (expected [[15,19,23],[15,19,23]]):")
    _print_buffer(grads[0].value())

    print("\nd_c/db (expected [[5,5],[7,7],[9,9]]):")
    _print_buffer(grads[1].value())


def _print_buffer(b: mg.Buffer) raises:
    with b.buf().map_to_host() as mapped:
        var ptr = mapped.unsafe_ptr()
        for i in range(b.size):
            print(ptr[i])
