import mograd as mg
from std.gpu.host import DeviceContext
from std.sys import has_accelerator


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()

    var x = mg.Tensor(
        ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True
    )
    var w = mg.Tensor(
        ctx, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [2, 3], requires_grad=True
    )

    var e = x.exp()
    print("exp([1..6]):")
    _print_buffer(e.value())

    var l = x.log()
    print("\nlog([1..6]):")
    _print_buffer(l.value())

    var n = x.neg()
    print("\nneg([1..6]):")
    _print_buffer(n.value())

    var d = x / w
    print("\n[1..6] / [0.5,1,1.5,2,2.5,3]:")
    _print_buffer(d.value())

    var s = x.sum()
    print("\nsum([1..6]):")
    _print_buffer(s.value())

    var r = x.reshape([6])
    print("\nreshape([2,3] -> [6]):")
    _print_buffer(r.value())

    var exp_grads = e.gradient(x)
    print("\nd_exp/dx (should equal exp([1..6])):")
    _print_buffer(exp_grads[0].value())

    var log_grads = l.gradient(x)
    print("\nd_log/dx (should be 1/[1..6]):")
    _print_buffer(log_grads[0].value())

    var div_grads = d.gradient(x, w)
    print("\nd_div/dx (should be 1/w = [2,1,0.667,0.5,0.4,0.333]):")
    _print_buffer(div_grads[0].value())
    print("\nd_div/dw (should be -x/w^2):")
    _print_buffer(div_grads[1].value())

    var sum_grads = s.gradient(x)
    print("\nd_sum/dx (should be all 1s):")
    _print_buffer(sum_grads[0].value())

    var reshape_grads = r.gradient(x)
    print("\nd_reshape/dx (should be all 1s, shape [2,3]):")
    _print_buffer(reshape_grads[0].value())


def _print_buffer(b: mg.Buffer) raises:
    with b.buf().map_to_host() as mapped:
        var ptr = mapped.unsafe_ptr()
        for i in range(b.size):
            print(ptr[i])
