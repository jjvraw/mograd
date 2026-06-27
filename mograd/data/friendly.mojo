from std.collections.string import Codepoint

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


struct TinyShakespeareData(Copyable, Movable):
    var data: Tensor
    var vocab_size: Int
    var train_size: Int
    var val_size: Int
    var _vocab: List[Int]  # token id → unicode codepoint

    def __init__(out self, data: Tensor, vocab_size: Int, train_size: Int, val_size: Int, var vocab: List[Int]):
        self.data = data
        self.vocab_size = vocab_size
        self.train_size = train_size
        self.val_size = val_size
        self._vocab = vocab^

    def decode(self, tokens: List[Int64]) raises -> String:
        var s = String("")
        for id in tokens:
            s += String(Codepoint.from_u32(UInt32(self._vocab[Int(id)])).value())
        return s^

    def get_batch(self, seq_len: Int, batch_size: Int, seed: Int = 42) raises -> Tuple[Tensor, Tensor]:
        """Samples a random batch of (x, y) windows from the training split.

        x[b, t] = data[offset[b] + t]
        y[b, t] = data[offset[b] + t + 1]   (next-token targets)
        """
        var device = self.data.device.value()

        var offsets = Tensor.randint(device, (batch_size,), 0, self.train_size - seq_len, seed=seed)

        var off = offsets.to_list[DType.int64]()
        var x_idx = List[Int64]()
        var y_idx = List[Int64]()
        for b in range(batch_size):
            for t in range(seq_len):
                x_idx.append(off[b] + Int64(t))
                y_idx.append(off[b] + Int64(t) + 1)

        var x_t = Tensor(device, x_idx, (batch_size, seq_len))
        var y_t = Tensor(device, y_idx, (batch_size, seq_len))

        # gather requires a rank-2 source: reshape (n,) → (n, 1), gather to
        # (batch_size, seq_len, 1), then squeeze the trailing dim.
        var data2d = self.data.reshape((self.data.shape(0), 1))
        var x = data2d[x_t].squeeze(2)
        var y = data2d[y_t].squeeze(2)
        return (x^, y^)


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

    var vocab_path = loader.cache_dir + "/tinyshakespeare_vocab.txt"
    var vocab = List[Int]()
    with open(vocab_path, "r") as f:
        for line in f.read().splitlines():
            if line:
                vocab.append(Int(line))

    var n = loader.file_len(data_path) // 8
    var train_n = (n * 9) // 10
    var val_n = n - train_n

    return TinyShakespeareData(
        data=Tensor.disk(device, data_path, (n,), DType.int64),
        vocab_size=len(vocab),
        train_size=train_n,
        val_size=val_n,
        vocab=vocab^,
    )
