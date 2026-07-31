from std.memory import ArcPointer

from mograd.tensor import Tensor

comptime ModuleParam = ArcPointer[Optional[Tensor]]


trait Module(Movable):
    def parameters(mut self) -> List[ModuleParam]:
        ...

    def __call__(mut self, x: Tensor) raises -> Tensor:
        ...
