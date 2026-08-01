from std.sys import get_defined_int
from std.time import perf_counter_ns

from mograd.device_ctx import Device
from mograd.op import OpRef, OpType
from mograd.buffer import AnyBuffer

# -D MOGRAD_DEBUG=N:
#   1: Σ per-run line (kernel count, wall time)
#   2: 1 + per-op trace (op, shape, kernel names)
#   3: 2 + per-kernel timing & throughput over device-buffer sizes
#      synchronizes around every kernel, (inter-kernel pipelining is lost)
#   4: 3 + op graph printed before each run
comptime DEBUG = get_defined_int["MOGRAD_DEBUG", 0]()

# ===-------------------------------------------------------------------===#
# DebugStats
# ===-------------------------------------------------------------------===#


struct DebugStats(Copyable, Movable):
    """Run counters, populated only when MOGRAD_DEBUG >= 1. The Tracer fills
    one instance per run and folds it into the cumulative instance shared
    through Device.stats."""

    var runs: Int
    var kernels: Int
    var gpu_ns: Int
    var dispatch_ns: Int
    var bytes_moved: Int

    def __init__(out self):
        self.runs = 0
        self.kernels = 0
        self.gpu_ns = 0
        self.dispatch_ns = 0
        self.bytes_moved = 0

    def __iadd__(mut self, other: Self):
        self.runs += other.runs
        self.kernels += other.kernels
        self.gpu_ns += other.gpu_ns
        self.dispatch_ns += other.dispatch_ns
        self.bytes_moved += other.bytes_moved


# ===-------------------------------------------------------------------===#
# Tracer
# ===-------------------------------------------------------------------===#


struct Tracer:
    """Scheduler instrumentation."""

    comptime KERNEL_COL = 32

    var stats: DebugStats
    var run_t0: Int
    var node_t0: Int

    def __init__(out self):
        self.stats = DebugStats()
        self.run_t0 = 0
        self.node_t0 = 0

    # ===-------------------------------------------------------------------===#
    # Hooks
    # ===-------------------------------------------------------------------===#

    @always_inline
    def run_begin(mut self, root: OpRef):
        comptime if DEBUG >= 4:
            print(root)
        comptime if DEBUG >= 3:
            print("    " + Self.rpad("kernel", Self.KERNEL_COL) + Self.lpad("time", 10) + Self.lpad("tput", 12))
        comptime if DEBUG >= 1:
            self.run_t0 = Int(perf_counter_ns())

    @always_inline
    def node_begin(mut self, node: OpRef, device: Device) raises:
        comptime if DEBUG >= 2:
            print(node.op_type()._name, Self.shape_str(node))
        comptime if DEBUG >= 3:
            if not Self.is_view_op(node):
                device.ctx.synchronize()
                self.node_t0 = Int(perf_counter_ns())

    @always_inline
    def node_end(mut self, node: OpRef, inputs: List[AnyBuffer], results: List[AnyBuffer], device: Device) raises:
        comptime if DEBUG >= 1:
            if not Self.is_view_op(node):
                self.stats.kernels += 1
        comptime if DEBUG >= 3:
            if not Self.is_view_op(node):
                device.ctx.synchronize()
                var dt = Int(perf_counter_ns()) - self.node_t0
                var bytes_touched = 0
                for i in range(len(inputs)):
                    bytes_touched += inputs[i].size_bytes()
                for i in range(len(results)):
                    bytes_touched += results[i].size_bytes()
                self.stats.gpu_ns += dt
                self.stats.bytes_moved += bytes_touched
                var tput = Self.fmt1(Float64(bytes_touched) / Float64(dt)) + " GB/s" if dt > 0 else String("")
                print(Self.lpad(Self.time_str(dt), 10) + Self.lpad(tput, 12))

    @always_inline
    def run_end(mut self, device: Device):
        comptime if DEBUG >= 1:
            var run_ns = Int(perf_counter_ns()) - self.run_t0
            self.stats.runs = 1
            comptime if DEBUG >= 3:
                self.stats.dispatch_ns = run_ns - self.stats.gpu_ns
            elif DEBUG >= 1:
                self.stats.dispatch_ns = run_ns
            var line = "\nΣ run: " + String(self.stats.kernels) + " kernels  " + Self.time_str(run_ns)
            comptime if DEBUG >= 3:
                line += (
                    "  = "
                    + Self.time_str(self.stats.gpu_ns)
                    + " gpu + "
                    + Self.time_str(self.stats.dispatch_ns)
                    + " dispatch"
                )
            print(line)
            device.stats[] += self.stats

    # ===-------------------------------------------------------------------===#
    # Helpers
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def rpad(s: String, n: Int) -> String:
        var out = s.copy()
        while out.count_codepoints() < n:
            out += " "
        return out^

    @staticmethod
    def lpad(s: String, n: Int) -> String:
        var out = s.copy()
        while out.count_codepoints() < n:
            out = " " + out
        return out^

    @staticmethod
    def shape_str(node: OpRef) -> String:
        var l = node.layout()
        var s = String("(")
        for i in range(l.rank()):
            if i > 0:
                s += ","
            s += String(l.shape(i))
        return s + ")"

    @staticmethod
    def is_view_op(node: OpRef) -> Bool:
        var t = node.op_type()
        return (
            t == OpType.EXPAND
            or t == OpType.VIEW
            or t == OpType.SLICE
            or t == OpType.RESHAPE
            or t == OpType.TRANSPOSE
            or t == OpType.SQUEEZE
            or t == OpType.UNSQUEEZE
        )

    @staticmethod
    def fmt1(x: Float64) -> String:
        """Formats to one decimal place."""
        var scaled = Int(x * 10.0)
        return String(scaled // 10) + "." + String(scaled % 10)

    @staticmethod
    def time_str(ns: Int) -> String:
        if ns < 1_000_000:
            return Self.fmt1(Float64(ns) / 1000.0) + "µs"
        if ns < 1_000_000_000:
            return Self.fmt1(Float64(ns) / 1000000.0) + "ms"
        return Self.fmt1(Float64(ns) / 1000000000.0) + "s"
