import mograd as mg


def main() raises:
    var x = mg.Tensor.empty([2, 3], requires_grad=True)
    var w = mg.Tensor.empty([2, 3], requires_grad=True)
    var out = x * w + w

    print(out)
    var grads = out.gradient(x, w)

    for t in grads:
        print(t)
