comptime MAX_RANK = 8

# ===-------------------------------------------------------------------===#
# Shape
# ===-------------------------------------------------------------------===#


struct Shape(Copyable, ImplicitlyCopyable, Movable, Sized, Writable):
    var _dims: InlineArray[Int, MAX_RANK]
    var _rank: Int

    def __init__(out self, *dims: Int):
        self._rank = len(dims)
        self._dims = InlineArray[Int, MAX_RANK](fill=0)
        for i in range(self._rank):
            self._dims[i] = dims[i]

    @implicit
    def __init__(out self, t: Tuple[Int]):
        self._rank = 1
        self._dims = InlineArray[Int, MAX_RANK](fill=0)
        self._dims[0] = t[0]

    @implicit
    def __init__(out self, t: Tuple[Int, Int]):
        self._rank = 2
        self._dims = InlineArray[Int, MAX_RANK](fill=0)
        self._dims[0] = t[0]
        self._dims[1] = t[1]

    @implicit
    def __init__(out self, t: Tuple[Int, Int, Int]):
        self._rank = 3
        self._dims = InlineArray[Int, MAX_RANK](fill=0)
        self._dims[0] = t[0]
        self._dims[1] = t[1]
        self._dims[2] = t[2]

    @implicit
    def __init__(out self, t: Tuple[Int, Int, Int, Int]):
        self._rank = 4
        self._dims = InlineArray[Int, MAX_RANK](fill=0)
        self._dims[0] = t[0]
        self._dims[1] = t[1]
        self._dims[2] = t[2]
        self._dims[3] = t[3]

    @implicit
    def __init__(out self, t: Tuple[Int, Int, Int, Int, Int]):
        self._rank = 5
        self._dims = InlineArray[Int, MAX_RANK](fill=0)
        self._dims[0] = t[0]
        self._dims[1] = t[1]
        self._dims[2] = t[2]
        self._dims[3] = t[3]
        self._dims[4] = t[4]

    @implicit
    def __init__(out self, dims: List[Int]):
        self._rank = len(dims)
        self._dims = InlineArray[Int, MAX_RANK](fill=0)
        for i in range(self._rank):
            self._dims[i] = dims[i]

    def __getitem__(self, i: Int) -> Int:
        return self._dims[i]

    def __setitem__(mut self, i: Int, val: Int):
        self._dims[i] = val

    def __len__(self) -> Int:
        return self._rank

    def numel(self) -> Int:
        var n = 1
        for i in range(self._rank):
            n *= self._dims[i]
        return n

    def to_list(self) -> List[Int]:
        var result = List[Int]()
        for i in range(self._rank):
            result.append(self._dims[i])
        return result^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("[")
        for i in range(self._rank):
            writer.write(String(self._dims[i]))
            if i < self._rank - 1:
                writer.write(", ")
        writer.write("]")
