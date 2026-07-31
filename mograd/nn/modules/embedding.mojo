from mograd.tensor import Tensor, Device
from mograd.nn.modules import Module, Parameter

# ===-------------------------------------------------------------------===#
# Embedding
# ===-------------------------------------------------------------------===#


struct Embedding(Module):
    var num_embeddings: Int
    var embedding_dim: Int
    var weight: Parameter

    def __init__(out self, num_embeddings: Int, embedding_dim: Int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter()

    def __call__(mut self, indices: Tensor) raises -> Tensor:
        if not self.weight:
            if not indices.device:
                raise Error("Embedding requires a device context on first call")
            var seed = UInt32(self.num_embeddings * self.embedding_dim)
            self.weight.set(
                Tensor.randn(
                    indices.device.value(),
                    (self.num_embeddings, self.embedding_dim),
                    seed=seed,
                    requires_grad=True,
                )
            )
        return self.weight.tensor().gather(indices)

    def parameters(mut self) -> List[Parameter]:
        var ps = List[Parameter]()
        ps.append(self.weight)
        return ps^
