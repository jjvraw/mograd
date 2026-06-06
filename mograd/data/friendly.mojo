from mograd.tensor import Tensor, Device
from mograd.data.lazy_loader import LazyLoader


@fieldwise_init
struct MNISTData[dtype: DType = DType.float32](Copyable, Movable) where dtype.is_floating_point():
    var x_train: Tensor[Self.dtype]  # [60000, 784]
    var y_train: Tensor[Self.dtype]  # [60000]
    var x_test: Tensor[Self.dtype]  # [10000, 784]
    var y_test: Tensor[Self.dtype]  # [10000]


comptime _BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def mnist[
    dtype: DType = DType.float32
](device: Device, cache_dir: String = "~/.mograd/datasets") raises -> MNISTData[dtype] where dtype.is_floating_point():
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

    return MNISTData[dtype](
        x_train=Tensor[dtype].disk(device, x_train_path, (60000, 784)),
        y_train=Tensor[dtype].disk(device, y_train_path, (60000,)),
        x_test=Tensor[dtype].disk(device, x_test_path, (10000, 784)),
        y_test=Tensor[dtype].disk(device, y_test_path, (10000,)),
    )
