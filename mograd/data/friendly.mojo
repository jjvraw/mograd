from std.gpu.host import DeviceContext

from mograd.tensor import Tensor
from mograd.data.lazy_loader import LazyLoader


@fieldwise_init
struct MNISTData(Copyable, Movable):
    var x_train: Tensor  # [60000, 784]
    var y_train: Tensor  # [60000]
    var x_test: Tensor  # [10000, 784]
    var y_test: Tensor  # [10000]


comptime _BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def mnist(ctx: DeviceContext, cache_dir: String = "~/.mograd/datasets") raises -> MNISTData:
    var loader = LazyLoader(cache_dir)

    var x_train_path = loader.ensure_cached_idx(
        _BASE + "train-images-idx3-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "mnist_x_train.bin",
        is_images=True,
    )
    var y_train_path = loader.ensure_cached_idx(
        _BASE + "train-labels-idx1-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "mnist_y_train.bin",
        is_images=False,
    )
    var x_test_path = loader.ensure_cached_idx(
        _BASE + "t10k-images-idx3-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "mnist_x_test.bin",
        is_images=True,
    )
    var y_test_path = loader.ensure_cached_idx(
        _BASE + "t10k-labels-idx1-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
        "mnist_y_test.bin",
        is_images=False,
    )

    return MNISTData(
        x_train=Tensor.disk(ctx, x_train_path, [60000, 784]),
        y_train=Tensor.disk(ctx, y_train_path, [60000]),
        x_test=Tensor.disk(ctx, x_test_path, [10000, 784]),
        y_test=Tensor.disk(ctx, y_test_path, [10000]),
    )
