from mograd.tensor import Tensor


trait Optimizer(Movable):
    def parameters(self) -> List[Tensor]:
        ...

    def step(mut self, grads: List[Tensor]) raises:
        ...

    def set_lr(mut self, lr: Float32):
        ...
