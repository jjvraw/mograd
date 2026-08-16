from max.gpu.host import DeviceContext

from mograd.buffer import AnyBuffer, BufferArm
from mograd.device_ctx import Device
from mograd.layout import Layout
from mograd.op import OpRef
from mograd.runtime.gpu.kernels.strided import (
    binary_strided_map,
    binary_strided_scalar_map,
    unary_strided_map,
)

# ===-------------------------------------------------------------------===#
# Kernel registry
# ===-------------------------------------------------------------------===#


struct KernelRegistry:
    comptime randn = KernelSym[RandomFactoryKernel]("mograd_randn")
    comptime uniform = KernelSym[RandomFactoryKernel]("mograd_uniform")
    comptime full = KernelSym[FactoryKernel]("mograd_full")
    comptime one_hot = KernelSym[OneHotOp]("mograd_one_hot")
    comptime gather = KernelSym[BinaryStrided]("mograd_gather")
    comptime scatter_add = KernelSym[BinaryStrided]("mograd_scatter_add")
    comptime neg = KernelSym[UnaryStrided]("mograd_neg")
    comptime log = KernelSym[UnaryStrided]("mograd_log")
    comptime exp = KernelSym[UnaryStrided]("mograd_exp")
    comptime sqrt = KernelSym[UnaryStrided]("mograd_sqrt")
    comptime relu = KernelSym[UnaryStrided]("mograd_relu")
    comptime relu_grad = KernelSym[BinaryStrided]("mograd_relu_grad")
    comptime slice_grad = KernelSym[UnaryStrided]("mograd_slice_grad")
    comptime cast = KernelSym[CastOp]("mograd_cast")
    comptime contiguous = KernelSym[UnaryStrided]("mograd_contiguous")
    comptime strided_copy = KernelSym[StridedCopy]("mograd_strided_copy")
    comptime transpose_last2 = KernelSym[UnaryStrided]("mograd_transpose_last2")
    comptime transpose = KernelSym[BinaryOp]("mograd_transpose")
    comptime add = KernelSym[BinaryElementWise]("mograd_add")
    comptime add_strided = KernelSym[BinaryStrided]("mograd_add_strided")
    comptime mul = KernelSym[BinaryStrided]("mograd_mul")
    comptime div = KernelSym[BinaryStrided]("mograd_div")
    comptime eq = KernelSym[BinaryStrided]("mograd_eq")
    comptime scale = KernelSym[BinaryScalarElementWiseStrided]("mograd_scale")
    comptime matmul = KernelSym[MatmulStrided]("mograd_matmul")
    comptime matmul_bt = KernelSym[MatmulStrided]("mograd_matmul_bt")
    comptime matmul_bias_bt = KernelSym[MatmulBiasStrided]("mograd_matmul_bias_bt")
    comptime flash_attn_fwd = KernelSym[FlashAttnFwdKernel]("mograd_flash_attn_fwd")
    comptime flash_attn_bwd = KernelSym[FlashAttnBwdKernel]("mograd_flash_attn_bwd")
    comptime softmax = KernelSym[UnaryStrided]("mograd_softmax")
    comptime softmax_strided = KernelSym[UnaryStrided]("mograd_softmax_strided")
    comptime softmax_grad = KernelSym[BinaryStrided]("mograd_softmax_grad")
    comptime sum = KernelSym[UnaryStrided]("mograd_sum")
    comptime sum_axis = KernelSym[AxisReduceKernel]("mograd_sum_axis")
    comptime mean = KernelSym[UnaryStrided]("mograd_mean")
    comptime mean_axis = KernelSym[AxisReduceKernel]("mograd_mean_axis")
    comptime argmax = KernelSym[UnaryStrided]("mograd_argmax")
    comptime argmax_axis = KernelSym[AxisReduceKernel]("mograd_argmax_axis")
    comptime layer_norm_fwd = KernelSym[LayerNormFwdKernel]("mograd_layer_norm_fwd")
    comptime layer_norm_bwd = KernelSym[LayerNormBwdKernel]("mograd_layer_norm_bwd")
    comptime cross_entropy = KernelSym[BinaryStrided]("mograd_cross_entropy")
    comptime cross_entropy_strided = KernelSym[BinaryStrided]("mograd_cross_entropy_strided")
    comptime cross_entropy_grad = KernelSym[TernaryStrided]("mograd_cross_entropy_grad")
    comptime cross_entropy_grad_strided = KernelSym[TernaryStrided]("mograd_cross_entropy_grad_strided")
    comptime triu = KernelSym[TriuOp]("mograd_triu")


@fieldwise_init
struct KernelSym[T: TrivialRegisterPassable](ImplicitlyCopyable, Movable):
    """An exported kernel symbol paired with its ABI signature."""

    var name: StaticString

    def load(self, device: Device) raises -> Self.T:
        return device.get_function[Self.T](String(self.name))


# ===-------------------------------------------------------------------===#
# Host dispatch
# Erased ABI signature (dtype as a runtime argument) paired with the helper
# that loads its registry symbol and marshals node buffers into it.
# ===-------------------------------------------------------------------===#


comptime FactoryKernel = def(
    params: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    numel: Int,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


comptime RandomFactoryKernel = def(
    params: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    numel: Int,
    dtype: DType,
    seed: UInt64,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


comptime UnaryStrided = def(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    layout: Layout,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def unary_strided(
    sym: KernelSym[UnaryStrided], imm node: OpRef, imm inputs: List[AnyBuffer], imm device: Device
) raises -> AnyBuffer:
    var layout = node.src(0).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    sym.load(device)(
        inputs[0].data_ptr(),
        out.data_ptr(),
        layout,
        node.dtype(),
        device.ctx,
    )
    return out^


comptime BinaryStrided = def(
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    la: Layout,
    lb: Layout,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def binary_strided(
    sym: KernelSym[BinaryStrided], imm node: OpRef, imm inputs: List[AnyBuffer], imm device: Device
) raises -> AnyBuffer:
    var la = node.src(0).layout()
    var lb = node.src(1).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    sym.load(device)(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        out.data_ptr(),
        la,
        lb,
        node.dtype(),
        device.ctx,
    )
    return out^


comptime StridedCopy = def(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    la: Layout,
    ld: Layout,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def strided_copy(
    sym: KernelSym[StridedCopy],
    imm src_layout: Layout,
    imm dst_layout: Layout,
    imm src: AnyBuffer,
    imm dst: AnyBuffer,
    imm dtype: DType,
    imm device: Device,
) raises:
    sym.load(device)(
        src.data_ptr(),
        dst.data_ptr(),
        src_layout,
        dst_layout,
        dtype,
        device.ctx,
    )


comptime AxisReduceKernel = def(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    layout: Layout,
    axis: Int,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def axis_reduce_strided(
    sym: KernelSym[AxisReduceKernel], imm node: OpRef, imm inputs: List[AnyBuffer], imm device: Device
) raises -> AnyBuffer:
    var layout = node.src(0).layout()
    var axis = node.attr_int("axis")
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    sym.load(device)(
        inputs[0].data_ptr(),
        out.data_ptr(),
        layout,
        axis,
        node.dtype(),
        device.ctx,
    )
    return out^


comptime MatmulStrided = def(
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    la: Layout,
    lb: Layout,
    N: Int,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def matmul_strided(
    sym: KernelSym[MatmulStrided], imm node: OpRef, imm inputs: List[AnyBuffer], imm device: Device
) raises -> AnyBuffer:
    var la = node.src(0).layout()
    var lb = node.src(1).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    sym.load(device)(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        out.data_ptr(),
        la,
        lb,
        node.shape(node.layout().rank() - 1),
        node.dtype(),
        device.ctx,
    )
    return out^


comptime MatmulBiasStrided = def(
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    bias: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    la: Layout,
    lb: Layout,
    N: Int,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def matmul_bias_strided(
    sym: KernelSym[MatmulBiasStrided], imm node: OpRef, imm inputs: List[AnyBuffer], imm device: Device
) raises -> AnyBuffer:
    var la = node.src(0).layout()
    var lb = node.src(1).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    sym.load(device)(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        inputs[2].data_ptr(),
        out.data_ptr(),
        la,
        lb,
        node.shape(node.layout().rank() - 1),
        node.dtype(),
        device.ctx,
    )
    return out^


comptime LayerNormFwdKernel = def(
    x: Pointer[NoneType, ImmutAnyOrigin],
    gamma: Pointer[NoneType, ImmutAnyOrigin],
    beta: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    rows: Int,
    cols: Int,
    eps: Float32,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def layer_norm_fwd_dispatch(
    sym: KernelSym[LayerNormFwdKernel], imm node: OpRef, imm inputs: List[AnyBuffer], imm device: Device
) raises -> AnyBuffer:
    var x_layout = node.src(0).layout()
    var cols = x_layout.shape(x_layout.rank() - 1)
    var rows = x_layout.numel() // cols
    var eps = node.attrs()["eps"][Float32]
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    sym.load(device)(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        inputs[2].data_ptr(),
        out.data_ptr(),
        rows,
        cols,
        eps,
        node.dtype(),
        device.ctx,
    )
    return out^


comptime LayerNormBwdKernel = def(
    dy: Pointer[NoneType, ImmutAnyOrigin],
    x: Pointer[NoneType, ImmutAnyOrigin],
    gamma: Pointer[NoneType, ImmutAnyOrigin],
    dx: Pointer[NoneType, MutAnyOrigin],
    dgamma: Pointer[NoneType, MutAnyOrigin],
    dbeta: Pointer[NoneType, MutAnyOrigin],
    rows: Int,
    cols: Int,
    eps: Float32,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def layer_norm_bwd_dispatch(
    sym: KernelSym[LayerNormBwdKernel], imm node: OpRef, imm inputs: List[AnyBuffer], imm device: Device
) raises -> List[AnyBuffer]:
    # inputs: [dy, x, gamma]  attrs: eps, axis  srcs: [dy, x, gamma]
    var x_layout = node.src(1).layout()
    var gamma_layout = node.src(2).layout()
    var cols = x_layout.shape(x_layout.rank() - 1)
    var rows = x_layout.numel() // cols
    var eps = node.attrs()["eps"][Float32]
    var dx = AnyBuffer.create(node.dtype(), device, x_layout.numel())
    var dgamma = AnyBuffer.create(node.dtype(), device, gamma_layout.numel(), fill=0.0)
    var dbeta = AnyBuffer.create(node.dtype(), device, gamma_layout.numel(), fill=0.0)
    sym.load(device)(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        inputs[2].data_ptr(),
        dx.data_ptr(),
        dgamma.data_ptr(),
        dbeta.data_ptr(),
        rows,
        cols,
        eps,
        node.dtype(),
        device.ctx,
    )
    return [dx^, dgamma^, dbeta^]


comptime FlashAttnFwdKernel = def(
    q: Pointer[NoneType, ImmutAnyOrigin],
    k: Pointer[NoneType, ImmutAnyOrigin],
    v: Pointer[NoneType, ImmutAnyOrigin],
    mask: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    lse: Pointer[NoneType, MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
    causal: Int,
    has_bias: Int,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def flash_attn_fwd_dispatch(
    sym: KernelSym[FlashAttnFwdKernel], imm node: OpRef, imm inputs: List[AnyBuffer], imm device: Device
) raises -> List[AnyBuffer]:
    # node.srcs = [Q, K, V, mask], all BSHD (B, S, H, D)
    # Returns [O_buf (BHSD), LSE_buf (BHS, float32)]
    var q_layout = node.src(0).layout()
    var B = q_layout.shape(0)
    var S = q_layout.shape(1)
    var H = q_layout.shape(2)
    var D = q_layout.shape(3)
    var scale = node.attrs()["scale"][Float32]
    var is_causal = node.attrs()["is_causal"][Bool]
    var has_bias = True
    if "has_bias" in node.attrs():
        has_bias = node.attrs()["has_bias"][Bool]
    var dst = AnyBuffer.create(node.dtype(), device, node.numel())
    var lse = AnyBuffer.create(DType.float32, device, B * H * S)
    sym.load(device)(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        inputs[2].data_ptr(),
        inputs[3].data_ptr(),
        dst.data_ptr(),
        lse.data_ptr(),
        B,
        S,
        H,
        D,
        scale,
        Int(is_causal),
        Int(has_bias),
        node.dtype(),
        device.ctx,
    )
    return [dst^, lse^]


comptime FlashAttnBwdKernel = def(
    dy: Pointer[NoneType, ImmutAnyOrigin],
    o: Pointer[NoneType, ImmutAnyOrigin],
    q: Pointer[NoneType, ImmutAnyOrigin],
    k: Pointer[NoneType, ImmutAnyOrigin],
    v: Pointer[NoneType, ImmutAnyOrigin],
    mask: Pointer[NoneType, ImmutAnyOrigin],
    lse: Pointer[NoneType, ImmutAnyOrigin],
    dq: Pointer[NoneType, MutAnyOrigin],
    dk: Pointer[NoneType, MutAnyOrigin],
    dv: Pointer[NoneType, MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
    causal: Int,
    has_bias: Int,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def flash_attn_bwd_dispatch(
    sym: KernelSym[FlashAttnBwdKernel], imm node: OpRef, imm inputs: List[AnyBuffer], imm device: Device
) raises -> List[AnyBuffer]:
    # node.srcs = [dO, O, Q, K, V, mask, LSE]  attrs: scale
    var q_layout = node.src(2).layout()
    var B = q_layout.shape(0)
    var S = q_layout.shape(1)
    var H = q_layout.shape(2)
    var D = q_layout.shape(3)
    var scale = node.attrs()["scale"][Float32]
    var is_causal = node.attrs()["is_causal"][Bool]
    var has_bias = not is_causal
    if "has_bias" in node.attrs():
        has_bias = node.attrs()["has_bias"][Bool]
    var numel = q_layout.numel()
    var dq = AnyBuffer.create(node.dtype(), device, numel)
    var dk = AnyBuffer.create(node.dtype(), device, numel)
    var dv = AnyBuffer.create(node.dtype(), device, numel)
    sym.load(device)(
        inputs[0].data_ptr(),  # dO
        inputs[1].data_ptr(),  # O
        inputs[2].data_ptr(),  # Q
        inputs[3].data_ptr(),  # K
        inputs[4].data_ptr(),  # V
        inputs[5].data_ptr(),  # mask
        inputs[6].data_ptr(),  # LSE (float32, BHS)
        dq.data_ptr(),
        dk.data_ptr(),
        dv.data_ptr(),
        B,
        S,
        H,
        D,
        scale,
        Int(is_causal),
        Int(has_bias),
        node.dtype(),
        device.ctx,
    )
    return [dq^, dk^, dv^]


comptime OneHotOp = def(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    in_dtype: DType,
    out_dtype: DType,
    ld: Layout,
    la: Layout,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


comptime BinaryElementWise = def(
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    numel: Int,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


comptime BinaryScalarElementWiseStrided = def(
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    layout: Layout,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


comptime CastOp = def(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    layout: Layout,
    in_dtype: DType,
    out_dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


comptime TriuOp = def(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    layout: Layout,
    diagonal: Int,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


comptime BinaryOp = def(
    Pointer[NoneType, ImmutAnyOrigin],
    Pointer[NoneType, ImmutAnyOrigin],
    Pointer[NoneType, MutAnyOrigin],
    Int,
    DType,
    DeviceContext,
) thin abi("Mojo") raises -> None


comptime TernaryStrided = def(
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    c: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    la: Layout,
    lb: Layout,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


# ===-------------------------------------------------------------------===#
# Dtype dispatch (shared-library side)
# Typed `*Impl` signatures (dtype as a comptime parameter) paired with the
# dispatch_* that erases the runtime dtype into per-dtype instantiations.
# ===-------------------------------------------------------------------===#


def dispatch_dtype[
    body: def[d: DType]() capturing raises -> None,
    float_only: Bool = False,
](dtype: DType) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.length):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        comptime if (not float_only) or d.is_floating_point():
            if dtype == d:
                body[d]()
                return
    raise Error("unsupported dtype")


comptime UnaryStridedImpl = def[dtype: DType](
    a: Pointer[Scalar[dtype], ImmutAnyOrigin],
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    rank: Int,
    inner: Pointer[Int64, ImmutAnyOrigin],
    sa: Pointer[Int64, ImmutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises thin -> None


def dispatch_unary[
    kernel: UnaryStridedImpl,
    float_only: Bool = False,
](
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: Pointer[mut=False, Int64, _],
    sa: Pointer[mut=False, Int64, _],
    dtype: DType,
    ctx: DeviceContext,
) raises:
    @always_inline
    def body[d: DType]() raises capturing:
        kernel[d](
            a.unsafe_bitcast[Scalar[d]](),
            dst.unsafe_bitcast[Scalar[d]](),
            rank,
            inner.as_unsafe_any_origin(),
            sa.as_unsafe_any_origin(),
            numel,
            ctx,
        )

    dispatch_dtype[body, float_only](dtype)


comptime BinaryContigImpl = def[dtype: DType](
    a: Pointer[Scalar[dtype], ImmutAnyOrigin],
    b: Pointer[Scalar[dtype], ImmutAnyOrigin],
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises thin -> None


def dispatch_binary_contiguous[
    kernel: BinaryContigImpl,
    float_only: Bool = False,
](
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    numel: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    @always_inline
    def body[d: DType]() capturing raises:
        kernel[d](
            a.unsafe_bitcast[Scalar[d]](), b.unsafe_bitcast[Scalar[d]](), dst.unsafe_bitcast[Scalar[d]](), numel, ctx
        )

    dispatch_dtype[body, float_only](dtype)


def dispatch_unary_map[
    op: def[d: DType](x: Scalar[d]) thin -> Scalar[d],
    float_only: Bool = False,
](
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: Pointer[mut=False, Int64, _],
    sa: Pointer[mut=False, Int64, _],
    dtype: DType,
    ctx: DeviceContext,
) raises:
    @always_inline
    def body[d: DType]() capturing raises:
        unary_strided_map[d, op[d]](
            a.unsafe_bitcast[Scalar[d]](),
            dst.unsafe_bitcast[Scalar[d]](),
            rank,
            inner,
            sa,
            numel,
            ctx,
        )

    dispatch_dtype[body, float_only](dtype)


def dispatch_binary_map[
    op: def[d: DType](x: Scalar[d], y: Scalar[d]) thin -> Scalar[d],
    float_only: Bool = False,
](
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: Pointer[mut=False, Int64, _],
    sa: Pointer[mut=False, Int64, _],
    sb: Pointer[mut=False, Int64, _],
    dtype: DType,
    ctx: DeviceContext,
) raises:
    @always_inline
    def body[d: DType]() capturing raises:
        binary_strided_map[d, op[d]](
            a.unsafe_bitcast[Scalar[d]](),
            b.unsafe_bitcast[Scalar[d]](),
            dst.unsafe_bitcast[Scalar[d]](),
            rank,
            inner,
            sa,
            sb,
            numel,
            ctx,
        )

    dispatch_dtype[body, float_only](dtype)


def dispatch_binary_scalar_map[
    op: def[d: DType](x: Scalar[d], y: Scalar[d]) thin -> Scalar[d],
    float_only: Bool = False,
](
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: Pointer[mut=False, Int64, _],
    sa: Pointer[mut=False, Int64, _],
    dtype: DType,
    ctx: DeviceContext,
) raises:
    @always_inline
    def body[d: DType]() capturing raises:
        # The scheduler always materialises the scalar operand as Float32
        # (see runtime scale()). Convert to the tensor dtype rather than
        # bitcasting, which reads garbage for any non-f32 dtype.
        var scalar = Scalar[d](b.unsafe_bitcast[Float32]()[unsafe_offset=0])
        binary_strided_scalar_map[d, op[d]](
            a.unsafe_bitcast[Scalar[d]](),
            Pointer(to=scalar),
            dst.unsafe_bitcast[Scalar[d]](),
            rank,
            inner,
            sa,
            numel,
            ctx,
        )

    dispatch_dtype[body, float_only](dtype)
