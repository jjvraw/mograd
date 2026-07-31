from std.memory import ArcPointer

from mograd.tensor import Tensor, Device
from mograd.nn.modules import Module, ModuleParam

# ===-------------------------------------------------------------------===#
# Embedding
# ===-------------------------------------------------------------------===#


struct Embedding(Module):
    var num_embeddings: Int
    var embedding_dim: Int
    var _weight: ArcPointer[Optional[Tensor]]

    def __init__(out self, num_embeddings: Int, embedding_dim: Int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._weight = ArcPointer(Optional[Tensor](None))

    def __call__(mut self, indices: Tensor) raises -> Tensor:
        if not self._weight[]:
            if not indices.device:
                raise Error("Embedding requires a device context on first call")
            var seed = UInt32(self.num_embeddings * self.embedding_dim)
            self._weight[] = Tensor.randn(
                indices.device.value(),
                (self.num_embeddings, self.embedding_dim),
                seed=seed,
                requires_grad=True,
            )
        return self._weight[].value().gather(indices)

    def parameters(mut self) -> List[ModuleParam]:
        var ps = List[ModuleParam]()
        ps.append(self._weight)
        return ps^
