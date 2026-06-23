from mograd.tensor import Tensor, Device
from mograd.data.lazy_loader import LazyLoader


# ===-------------------------------------------------------------------===#
# MNIST
# ===-------------------------------------------------------------------===#


@fieldwise_init
struct MNISTData(Copyable, Movable):
    var x_train: Tensor  # [60000, 784]
    var y_train: Tensor  # [60000]
    var x_test: Tensor  # [10000, 784]
    var y_test: Tensor  # [10000]


comptime MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def mnist(
    device: Device,
    cache_dir: String = "~/.mograd/datasets",
    dtype: DType = DType.float32,
) raises -> MNISTData:
    var loader = LazyLoader(cache_dir)

    var x_train_path = loader.ensure_cached_idx(
        MNIST_BASE + "train-images-idx3-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "mnist_x_train.bin",
        scale=1.0 / 255.0,
    )
    var y_train_path = loader.ensure_cached_idx(
        MNIST_BASE + "train-labels-idx1-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "mnist_y_train.bin",
    )
    var x_test_path = loader.ensure_cached_idx(
        MNIST_BASE + "t10k-images-idx3-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "mnist_x_test.bin",
        scale=1.0 / 255.0,
    )
    var y_test_path = loader.ensure_cached_idx(
        MNIST_BASE + "t10k-labels-idx1-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
        "mnist_y_test.bin",
    )

    return MNISTData(
        x_train=Tensor.disk(device, x_train_path, (60000, 784), dtype),
        y_train=Tensor.disk(device, y_train_path, (60000,), dtype),
        x_test=Tensor.disk(device, x_test_path, (10000, 784), dtype),
        y_test=Tensor.disk(device, y_test_path, (10000,), dtype),
    )


# ===-------------------------------------------------------------------===#
# TinyShakespear
# ===-------------------------------------------------------------------===#


@fieldwise_init
struct TinyShakespeareData(Copyable, Movable):
    var data: Tensor
    var vocab_size: Int
    var train_size: Int
    var val_size: Int


comptime SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def tiny_shakespeare(
    device: Device,
    cache_dir: String = "~/.mograd/datasets",
) raises -> TinyShakespeareData:
    var loader = LazyLoader(cache_dir)

    var data_path = loader.ensure_cached_text_encoded(
        SHAKESPEARE_URL,
        "tinyshakespeare.txt",
        "tinyshakespeare_tokens.bin",
        "tinyshakespeare_vocab.txt",
    )

    var n = loader.file_len(data_path) // 8
    var train_n = (n * 9) // 10
    var val_n = n - train_n

    return TinyShakespeareData(
        data=Tensor.disk(device, data_path, (n,), DType.int64),
        vocab_size=65,
        train_size=train_n,
        val_size=val_n,
    )
