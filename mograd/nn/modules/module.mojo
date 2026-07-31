from std.memory import ArcPointer

from mograd.tensor import Tensor


struct Parameter(Boolable, Copyable, ImplicitlyCopyable, Movable):
    """A trainable tensor slot. Copies share the slot, so a module and its
    optimizer observe each other's updates.

    `Parameter()` creates an empty slot for modules that initialize weights
    lazily on first call. `Parameter(t)` wraps an existing tensor and marks
    it trainable.
    """

    var slot: ArcPointer[Optional[Tensor]]

    def __init__(out self):
        self.slot = ArcPointer(Optional[Tensor](None))

    def __init__(out self, var t: Tensor):
        t.requires_grad = True
        self.slot = ArcPointer(Optional[Tensor](t^))

    def __bool__(self) -> Bool:
        return Bool(self.slot[])

    def set(mut self, var t: Tensor):
        self.slot[] = t^

    def tensor(self) -> Tensor:
        return self.slot[].value()


trait Module(Movable):
    def parameters(mut self) -> List[Parameter]:
        ...

    def __call__(mut self, x: Tensor) raises -> Tensor:
        ...
