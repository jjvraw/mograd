from mograd.tensor import Tensor

# ===-------------------------------------------------------------------===#
# Linear
# ===-------------------------------------------------------------------===#


struct Linear(Copyable, Movable):
    var weight: Tensor  # [out_features, in_features]

    def __init__(out self, var weight: Tensor):
        self.weight = weight^

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.weight.transpose()
