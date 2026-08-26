from std.math import sqrt
from std.sys import has_accelerator
from std.sys.info import has_apple_gpu_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal, assert_raises, assert_true, assert_false
from std.utils.numerics import neg_inf

from mograd import Tensor, Device
from mograd.testing import assert_allclose, assert_close


def test_basic_mixed_add_sub_neg() raises:
    var device = Device()
    var a = Tensor.ones(device, (8, 16))
    var b = Tensor.ones(device, (8, 16))
    var c = ((a + b) * 2) - (3 * a) + (2 + b) + a - a
    assert_allclose(c, Tensor.full(device, (8, 16), 4))


def test_basic_mixed_mul_div() raises:
    var device = Device()
    var a = Tensor.ones(device, (8, 16))
    var b = Tensor.full(device, (8, 16), 2)
    var c = (((((a * b) + 1) * -b) * a) / a) / b
    assert_allclose(c, Tensor.full(device, (8, 16), -3))


def test_matmul() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2, 3, 4], (2, 2))
    var identity = Tensor(device, [Float32(1), 0, 0, 1], (2, 2))
    assert_allclose(a @ identity, [Float32(1), 2, 3, 4])


def test_slice_preserves_shape_for_matmul() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6], (3, 2))
    var identity = Tensor(device, [Float32(1), 0, 0, 1], (2, 2))
    assert_allclose(x[1:3] @ identity, [Float32(3), 4, 5, 6])


def test_batched_matmul_rank3() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8], (2, 2, 2))
    var identity = Tensor(device, [Float32(1), 0, 0, 1, 1, 0, 0, 1], (2, 2, 2))
    assert_allclose(a @ identity, [Float32(1), 2, 3, 4, 5, 6, 7, 8])


def test_batched_matmul_transpose_b_rank3() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8], (2, 2, 2))
    var identity = Tensor(device, [Float32(1), 0, 0, 1, 1, 0, 0, 1], (2, 2, 2))
    assert_allclose(a.matmul(identity.transpose()), [Float32(1), 2, 3, 4, 5, 6, 7, 8])


def test_batched_matmul_rank4() raises:
    var device = Device()
    var a = Tensor(device, [Float32(i) for i in range(16)], (2, 2, 2, 2))
    var identity = Tensor(device, [Float32(1), 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], (2, 2, 2, 2))
    assert_allclose(a @ identity, [Float32(i) for i in range(16)])


def test_batched_matmul_non_square() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (2, 2, 3))
    var b = Tensor(device, [Float32(1), 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1], (2, 3, 2))
    assert_allclose(a @ b, [Float32(1), 2, 4, 5, 8, 9, 11, 12])


def test_reshape_then_matmul() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6], (6,)).reshape((2, 3))
    var identity = Tensor(device, [Float32(1), 0, 0, 0, 1, 0, 0, 0, 1], (3, 3))
    assert_allclose(x @ identity, [Float32(1), 2, 3, 4, 5, 6])


def test_view_then_batched_matmul() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8], (8,)).view((2, 2, 2))
    var identity = Tensor(device, [Float32(1), 0, 0, 1, 1, 0, 0, 1], (2, 2, 2))
    assert_allclose(x @ identity, [Float32(1), 2, 3, 4, 5, 6, 7, 8])


def test_transpose_single_batch_axis_then_matmul() raises:
    var device = Device()
    var a = Tensor(device, [Float32(i) for i in range(12)], (2, 3, 2)).transpose(0, 1)
    var identity = Tensor(device, [Float32(1), 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], (3, 2, 2))
    assert_allclose(a @ identity, a.contiguous())


def test_attention_style_middle_transpose_matmul() raises:
    var device = Device()
    var a = Tensor(device, [Float32(i) for i in range(16)], (2, 2, 2, 2)).transpose(1, 2)
    var identity = Tensor(device, [Float32(1), 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], (2, 2, 2, 2))
    assert_allclose(a @ identity, a.contiguous())


def test_softmax_sums_to_one() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4], (4,))
    assert_close(x.softmax().sum(), Float32(1.0))


def test_softmax_rank3_sums_to_one_per_row() raises:
    # Regression test: rows used to be computed as just shape(0), which is
    # wrong for rank >= 3 (e.g. 2 instead of 2*3=6 independent rows here).
    var device = Device()
    var x = Tensor.randn(device, (2, 3, 4), seed=7)
    var sums = x.softmax().sum(axis=-1)
    assert_allclose(sums, Tensor.ones(device, (2, 3)), tol=1e-4)


def test_softmax_axis0() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4], (2, 2))
    var out = x.softmax(axis=0)
    assert_allclose(out.sum(axis=0), [Float32(1), 1], tol=1e-5)


def test_softmax_axis0_matches_transpose_compose() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6], (2, 3))
    assert_allclose(x.softmax(axis=0), x.transpose().softmax().transpose())


def test_softmax_middle_axis_rank3() raises:
    var device = Device()
    var x = Tensor.randn(device, (2, 3, 4), seed=11)
    var out = x.softmax(axis=1)
    assert_allclose(out.sum(axis=1), Tensor.ones(device, (2, 4)), tol=1e-4)
    assert_allclose(out, x.transpose(1, 2).softmax().transpose(1, 2))


def test_one_hot_identity() raises:
    var device = Device()
    var labels = Tensor(device, [Int64(0), 1, 2], (3,))
    assert_allclose(labels.one_hot(3), [Int64(1), 0, 0, 0, 1, 0, 0, 0, 1])


def test_one_hot_shape() raises:
    var device = Device()
    var labels = Tensor(device, [Int64(0), 1, 2, 3], (4,))
    var oh = labels.one_hot(10)
    assert_true(oh.shape(0) == 4 and oh.shape(1) == 10)


def test_one_hot_values() raises:
    var device = Device()
    var labels = Tensor(device, [Int64(2), 0, 1], (3,))
    assert_allclose(
        labels.one_hot(4),
        [Int64(0), 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    )


def test_cross_entropy_uniform() raises:
    var device = Device()
    var logits = Tensor.full(device, (4, 10), Float32(0))
    var labels = Tensor(device, [Int64(0), 1, 2, 3], (4,))
    assert_close(logits.cross_entropy(labels.one_hot(10).cast(DType.float32)), Float32(2.302585), tol=1e-3)


def test_cross_entropy_soft_labels() raises:
    var device = Device()
    var logits = Tensor.full(device, (2, 4), Float32(0))
    var labels = Tensor.full(device, (2, 4), Float32(0.25))
    assert_close(logits.cross_entropy(labels), Float32(1.386294), tol=1e-3)


def test_cross_entropy_certain_prediction() raises:
    var device = Device()
    var logits = Tensor(device, [Float32(100), 0, 0, 0, 100, 0], (2, 3))
    var labels = Tensor(device, [Int64(1), 0, 0, 0, 1, 0], (2, 3))
    assert_close(logits.cross_entropy(labels.cast(DType.float32)), Float32(0.0), tol=1e-3)


def test_cross_entropy_rank3_matches_reshape() raises:
    var device = Device()
    var logits = Tensor.randn(device, (2, 3, 4))
    var label_ids = Tensor(device, [Int64(0), 2, 1, 3, 0, 1], (6,))
    var labels = label_ids.one_hot(4).cast(DType.float32).reshape((2, 3, 4))
    var loss_rank3 = logits.cross_entropy(labels)
    var loss_reshaped = logits.reshape((6, 4)).cross_entropy(labels.reshape((6, 4)))
    assert_close(loss_rank3, loss_reshaped.item(), tol=1e-4)


def test_slice_rows() raises:
    var device = Device()
    var data: List[Float32] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    var x = Tensor(device, data, (4, 3))
    assert_allclose(x[1:3], [Float32(4), 5, 6, 7, 8, 9])


def test_argmax() raises:
    var device = Device()
    var x = Tensor(device, [Float32(0.1), 0.9, 0.2, 0.3], (1, 4))
    assert_allclose(x.argmax(), [Float32(1)])


def test_argmax_axis1() raises:
    var device = Device()
    var x = Tensor(
        device,
        [
            Float32(1),
            3,
            2,
            0,
            Float32(4),
            0,
            1,
            2,
            Float32(0),
            2,
            3,
            1,
        ],
        (3, 4),
    )
    var out = x.argmax(axis=1)
    assert_true(out.shape(0) == 3)
    assert_allclose(out, [Float32(1), 0, 2])


def test_argmax_axis0() raises:
    var device = Device()
    var x = Tensor(
        device,
        [
            Float32(1),
            3,
            2,
            0,
            Float32(4),
            0,
            1,
            2,
            Float32(0),
            2,
            3,
            1,
        ],
        (3, 4),
    )
    var out = x.argmax(axis=0)
    assert_true(out.shape(0) == 4)
    assert_allclose(out, [Float32(1), 0, 2, 1])


def test_argmax_keepdim() raises:
    var device = Device()
    var x = Tensor(
        device,
        [
            Float32(1),
            3,
            2,
            0,
            Float32(4),
            0,
            1,
            2,
            Float32(0),
            2,
            3,
            1,
        ],
        (3, 4),
    )
    var out = x.argmax(axis=1, keepdim=True)
    assert_true(out.shape(0) == 3 and out.shape(1) == 1)
    assert_allclose(out, [Float32(1), 0, 2])


def test_argmax_axis1_3d() raises:
    var device = Device()
    var x = Tensor(
        device,
        [
            Float32(4),
            3,
            2,
            1,
            Float32(1),
            4,
            3,
            2,
            Float32(2),
            1,
            4,
            3,
            Float32(1),
            2,
            3,
            4,
            Float32(4),
            1,
            2,
            3,
            Float32(3),
            4,
            1,
            2,
        ],
        (2, 3, 4),
    )
    var out = x.argmax(axis=1)
    assert_true(out.shape(0) == 2 and out.shape(1) == 4)
    assert_allclose(out, [Float32(0), 1, 2, 2, 1, 2, 0, 0])


def test_argmax_negative_axis() raises:
    var device = Device()
    var x = Tensor(
        device,
        [
            Float32(1),
            3,
            2,
            0,
            Float32(4),
            0,
            1,
            2,
            Float32(0),
            2,
            3,
            1,
        ],
        (3, 4),
    )
    var out_neg = x.argmax(axis=-1)
    var out_pos = x.argmax(axis=1)
    assert_allclose(out_neg, out_pos)


def test_log_exp_inverse() raises:
    var device = Device()
    var data: List[Float32] = [0.5, 1.0, 2.0]
    var x = Tensor(device, data, (3,))
    assert_allclose(x.exp().log(), data)


def test_sqrt_values() raises:
    var device = Device()
    var x = Tensor(device, [Float32(4), 9, 16], (3,))
    assert_allclose(x.sqrt(), [Float32(2), 3, 4], tol=1e-4)


def test_relu_zeros_negatives() raises:
    var device = Device()
    var x = Tensor(device, [Float32(-2), -1, 0, 1, 2], (5,))
    assert_allclose(x.relu(), [Float32(0), 0, 0, 1, 2])


def test_eq_produces_zero_one() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2, 3], (3,))
    var b = Tensor(device, [Float32(1), 0, 3], (3,))
    assert_allclose(a == b, [Float32(1), 0, 1])


def test_scale() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4], (4,))
    assert_allclose(x * Float32(3.0), [Float32(3), 6, 9, 12])


def test_scale_float16() raises:
    # Regression test. The SCALE scalar operand is materialised as Float32
    # by the scheduler and used to be bitcast (not converted) to the tensor
    # dtype, which silently produced zeros for float16 tensors.
    var device = Device()
    var x = Tensor(device, [Float16(2), 4, 6, 8], (4,))
    assert_allclose(x * Float32(0.5), [Float16(1), 2, 3, 4])


def test_sum() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4], (4,))
    assert_close(x.sum(), Float32(10.0))


def test_sum_axis0() raises:
    var device = Device()
    var x = Tensor.full(device, (3, 4), Float32(1))
    var out = x.sum(axis=0)
    assert_true(out.shape(0) == 4)
    assert_allclose(out, [Float32(3), 3, 3, 3])


def test_sum_axis1() raises:
    var device = Device()
    var x = Tensor.full(device, (3, 4), Float32(1))
    var out = x.sum(axis=1)
    assert_true(out.shape(0) == 3)
    assert_allclose(out, [Float32(4), 4, 4])


def test_sum_axis_keepdim() raises:
    var device = Device()
    var x = Tensor.full(device, (3, 4), Float32(1))
    var out = x.sum(axis=1, keepdim=True)
    assert_true(out.shape(0) == 3 and out.shape(1) == 1)
    assert_allclose(out, [Float32(4), 4, 4])


def test_mean_axis1() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6], (2, 3))
    var out = x.mean(axis=1)
    assert_true(out.shape(0) == 2)
    assert_allclose(out, [Float32(2), 5])


def test_mean_axis_keepdim() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6], (2, 3))
    var out = x.mean(axis=1, keepdim=True)
    assert_true(out.shape(0) == 2 and out.shape(1) == 1)
    assert_allclose(out, [Float32(2), 5])


def test_mean_negative_axis() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6], (2, 3))
    assert_allclose(x.mean(axis=-1), [Float32(2), 5])


def test_sum_axis_3d() raises:
    var device = Device()
    var x = Tensor.full(device, (2, 3, 4), Float32(1))
    var out = x.sum(axis=1)
    assert_true(out.shape(0) == 2 and out.shape(1) == 4)
    assert_allclose(out, [Float32(3), 3, 3, 3, 3, 3, 3, 3])


def test_sum_axis_negative() raises:
    var device = Device()
    var x = Tensor.full(device, (3, 4), Float32(2))
    var out_neg = x.sum(axis=-1)
    var out_pos = x.sum(axis=1)
    assert_allclose(out_neg, out_pos)


def test_sum_large() raises:
    var device = Device()
    var n = 4096
    var x = Tensor.ones(device, (n,))
    assert_close(x.sum(), Float32(n))


def test_sum_of_softmax_is_one() raises:
    var device = Device()
    var x = Tensor.randn(device, (1024,))
    assert_close(x.softmax().sum(), Float32(1.0), tol=1e-4)


def test_randn_mean_and_std() raises:
    var device = Device()
    var x = Tensor.randn(device, (10000,), mean=0.0, std=1.0, seed=42)
    var vals = x.to_list()
    var mean = Float32(0.0)
    var sq = Float32(0.0)
    var n = Float32(len(vals))
    for i in range(len(vals)):
        mean += vals[i]
    mean /= n
    for i in range(len(vals)):
        sq += (vals[i] - mean) * (vals[i] - mean)
    var std = (sq / n) ** Float32(0.5)
    assert_almost_equal(mean, Float32(0.0), atol=0.05)
    assert_almost_equal(std, Float32(1.0), atol=0.05)


def test_randn_reproducible() raises:
    var device = Device()
    var a = Tensor.randn(device, (64,), seed=42)
    var b = Tensor.randn(device, (64,), seed=42)
    assert_allclose(a, b)


def test_randn_different_seeds_differ() raises:
    var device = Device()
    var a = Tensor.randn(device, (64,), seed=1).to_list()
    var b = Tensor.randn(device, (64,), seed=2).to_list()
    var any_diff = False
    for i in range(len(a)):
        if a[i] != b[i]:
            any_diff = True
            break
    assert_true(any_diff)


def test_transpose_2x3() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6], (2, 3))
    assert_allclose(x.transpose(), [Float32(1), 4, 2, 5, 3, 6])


def test_transpose_shape() raises:
    var device = Device()
    var x = Tensor.ones(device, (4, 7))
    var t = x.transpose()
    assert_true(t.shape(0) == 7 and t.shape(1) == 4)


def test_transpose_square() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4], (2, 2))
    assert_allclose(x.transpose(), [Float32(1), 3, 2, 4])


def test_mean_multi_batch_slice() raises:
    var device = Device()
    var x = Tensor.ones(device, (512, 8))
    var n = x.shape(0)
    var batch_size = 32
    for step in range(n // batch_size):
        var xb = x[step * batch_size : (step + 1) * batch_size]
        var m = xb.mean().item()
        assert_true(m > 0.99 and m < 1.01)


def test_transpose_tranpose() raises:
    var device = Device()
    var x = Tensor.randn(device, (113, 257), seed=7)
    assert_allclose(x.transpose().transpose(), x)


def test_cast_int64_to_float32() raises:
    var device = Device()
    var x = Tensor(device, [Int64(0), 1, 2, 3], (4,))
    var y = x.cast(DType.float32)
    assert_allclose(y, [Float32(0), 1, 2, 3])


def test_cast_one_hot_to_float32() raises:
    var device = Device()
    var labels = Tensor(device, [Int64(0), 1, 2], (3,))
    var oh = labels.one_hot(3).cast(DType.float32)
    assert_allclose(oh, [Float32(1), 0, 0, 0, 1, 0, 0, 0, 1])


def test_contiguous_transpose_is_not_contiguous() raises:
    var device = Device()
    var x = Tensor.ones(device, (3, 4))
    var t = x.transpose()
    assert_true(not t.is_contiguous())


def test_contiguous_transpose_after_contiguous_is_contiguous() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6], (2, 3))
    var t = x.transpose()
    assert_true(not t.is_contiguous())
    var c = t.contiguous()
    assert_true(c.is_contiguous())
    assert_allclose(c, [Float32(1), 4, 2, 5, 3, 6])


def test_contiguous_transpose_after_contiguous_is_contiguous_3d() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (2, 2, 3))
    var t = x.transpose()
    assert_true(not t.is_contiguous())
    var c = t.contiguous()
    assert_true(c.is_contiguous())
    assert_allclose(c, [Float32(1), 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12])


def test_contiguous_transpose_explicit_dims_after_contiguous_is_contiguous() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (2, 3, 2))
    var t = x.transpose(0, 1)  # swaps the first two axes, leaves the last untouched
    assert_true(not t.is_contiguous())
    var c = t.contiguous()
    assert_true(c.is_contiguous())
    assert_allclose(c, [Float32(1), 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12])


def test_contiguous_strided_slice_after_contiguous_is_contiguous() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6], (6,))
    var s = x[::2]
    assert_true(not s.is_contiguous())
    var c = s.contiguous()
    assert_true(c.is_contiguous())
    assert_allclose(c, [Float32(1), 3, 5])


def test_contiguous_already_contiguous_is_noop() raises:
    var device = Device()
    var x = Tensor.ones(device, (4, 4))
    assert_true(x.is_contiguous())
    var c = x.contiguous()
    assert_true(c.is_contiguous())


def test_gather_rows() raises:
    var device = Device()
    var table = Tensor(device, [Float32(i) for i in range(12)], (4, 3))
    var indices = Tensor(device, [Int64(1), 3, 1, 0], (4,))
    assert_allclose(table.gather(indices), [Float32(3), 4, 5, 9, 10, 11, 3, 4, 5, 0, 1, 2])


def test_gather_shape() raises:
    var device = Device()
    var table = Tensor.ones(device, (5, 8))
    var indices = Tensor(device, [Int64(0), 2, 4], (3,))
    var out = table.gather(indices)
    assert_true(out.shape(0) == 3 and out.shape(1) == 8)


def test_gather_2d_indices() raises:
    var device = Device()
    var table = Tensor(device, [Float32(i) for i in range(8)], (4, 2))
    var indices = Tensor(device, [Int64(0), 1, 2, 3], (2, 2))
    assert_allclose(table.gather(indices), [Float32(0), 1, 2, 3, 4, 5, 6, 7])


def test_gather_transposed_table_is_stride_aware() raises:
    var device = Device()
    var table = Tensor(device, [Float32(i) for i in range(12)], (3, 4)).transpose()
    var indices = Tensor(device, [Int64(1), 3, 1, 0], (4,))
    assert_allclose(table.gather(indices), [Float32(1), 5, 9, 3, 7, 11, 1, 5, 9, 0, 4, 8])


def test_gather_sliced_indices_is_stride_aware() raises:
    var device = Device()
    var table = Tensor(device, [Float32(i) for i in range(12)], (4, 3))
    var indices = Tensor(device, [Int64(0), 1, 3, 1], (4,))[1:3]
    assert_allclose(table.gather(indices), [Float32(3), 4, 5, 9, 10, 11])


def test_getitem_tensor_is_gather_sugar() raises:
    var device = Device()
    var table = Tensor(device, [Float32(i) for i in range(12)], (4, 3))
    var indices = Tensor(device, [Int64(1), 3, 1, 0], (4,))
    assert_allclose(table[indices], table.gather(indices))


def test_getitem_tensor_2d_indices() raises:
    var device = Device()
    var table = Tensor(device, [Float32(i) for i in range(8)], (4, 2))
    var indices = Tensor(device, [Int64(0), 1, 2, 3], (2, 2))
    assert_allclose(table[indices], [Float32(0), 1, 2, 3, 4, 5, 6, 7])


def test_randint_shape() raises:
    var device = Device()
    var idx = Tensor.randint(device, (5, 3), 0, 100)
    assert_true(idx.shape(0) == 5 and idx.shape(1) == 3)


def test_randint_in_range() raises:
    var device = Device()
    var idx = Tensor.randint(device, (200,), 10, 20)
    for v in idx.to_list[DType.int64]():
        assert_true(v >= 10 and v < 20)


def test_randint_default_dtype_is_int64() raises:
    var device = Device()
    var idx = Tensor.randint(device, (4,), 0, 10)
    assert_true(idx.dtype == DType.int64)


def test_randint_usable_as_gather_indices() raises:
    var device = Device()
    var table = Tensor(device, [Float32(i) for i in range(10)], (10, 1))
    var idx = Tensor.randint(device, (6,), 0, 10)
    var gathered = table[idx]
    assert_true(gathered.shape(0) == 6 and gathered.shape(1) == 1)


def test_scatter_add_rows() raises:
    var device = Device()
    var values = Tensor(device, [Float32(1) for _ in range(12)], (4, 3))
    var indices = Tensor(device, [Int64(1), 3, 1, 0], (4,))
    var out = values.scatter_add(indices, 5)
    assert_allclose(
        out,
        [Float32(1), 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    )


def test_scatter_add_shape() raises:
    var device = Device()
    var values = Tensor.ones(device, (3, 8))
    var indices = Tensor(device, [Int64(0), 2, 4], (3,))
    var out = values.scatter_add(indices, 5)
    assert_true(out.shape(0) == 5 and out.shape(1) == 8)


def test_scatter_add_float16() raises:
    var device = Device()
    var values = Tensor(device, [Float16(1) for _ in range(12)], (4, 3))
    var indices = Tensor(device, [Int64(1), 3, 1, 0], (4,))

    comptime if has_apple_gpu_accelerator():
        with assert_raises():
            _ = values.scatter_add(indices, 5)
    else:
        var out = values.scatter_add(indices, 5)
        assert_allclose(out, [Float16(1), 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0])


def test_scatter_add_is_gather_inverse() raises:
    var device = Device()
    var table = Tensor(device, [Float32(i) for i in range(12)], (4, 3))
    var indices = Tensor(device, [Int64(2), 0, 3], (3,))
    var gathered = table.gather(indices)
    assert_allclose(gathered.scatter_add(indices, 4), [Float32(0), 1, 2, 0, 0, 0, 6, 7, 8, 9, 10, 11])


def test_unsqueeze_shape() raises:
    var device = Device()
    var x = Tensor.ones(device, (3, 4))
    var y = x.unsqueeze(0)
    assert_true(y.shape(0) == 1 and y.shape(1) == 3 and y.shape(2) == 4)


def test_unsqueeze_values() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4], (2, 2))
    var y = x.unsqueeze(0)
    assert_allclose(y, [Float32(1), 2, 3, 4])


def test_unsqueeze_middle() raises:
    var device = Device()
    var x = Tensor.ones(device, (2, 3))
    var y = x.unsqueeze(1)
    assert_true(y.shape(0) == 2 and y.shape(1) == 1 and y.shape(2) == 3)


def test_squeeze_shape() raises:
    var device = Device()
    var x = Tensor.ones(device, (1, 3, 4))
    var y = x.squeeze(0)
    assert_true(y.shape(0) == 3 and y.shape(1) == 4)


def test_squeeze_values() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6], (1, 2, 3))
    var y = x.squeeze(0)
    assert_allclose(y, [Float32(1), 2, 3, 4, 5, 6])


def test_squeeze_middle() raises:
    var device = Device()
    var x = Tensor.ones(device, (2, 1, 3))
    var y = x.squeeze(1)
    assert_true(y.shape(0) == 2 and y.shape(1) == 3)


def test_squeeze_unsqueeze_roundtrip() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4], (2, 2))
    var y = x.unsqueeze(1).squeeze(1)
    assert_allclose(y, x)


def test_unsqueeze_negative_dim() raises:
    var device = Device()
    var x = Tensor.ones(device, (3, 4))
    var y = x.unsqueeze(-1)
    assert_true(y.shape(0) == 3 and y.shape(1) == 4 and y.shape(2) == 1)


def test_expand_shape_and_values() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3], (3, 1))
    var y = x.expand(3, 4)
    assert_true(y.shape(0) == 3 and y.shape(1) == 4)
    assert_allclose(y, [Float32(1), 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])


def test_expand_pads_rank() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3], (3,))
    var y = x.expand(4, -1)
    assert_true(y.shape(0) == 4 and y.shape(1) == 3)
    assert_allclose(y, [Float32(1), 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])


def test_triu_2d() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8, 9], (3, 3))
    var t = x.triu(0)
    assert_allclose(t, [Float32(1), 2, 3, 0, 5, 6, 0, 0, 9])


def test_triu_diagonal_1() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8, 9], (3, 3))
    var t = x.triu(1)
    assert_allclose(t, [Float32(0), 2, 3, 0, 0, 6, 0, 0, 0])


def test_triu_3d() raises:
    var device = Device()
    var x = Tensor.ones(device, (2, 3, 3))
    var t = x.triu(0)
    assert_true(t.shape() == x.shape())


def test_triu_with_inf() raises:
    var device = Device()
    var mask = Tensor.full(device, (2, 2), -1e9).triu(1)
    assert_allclose(mask, [Float32(0), -1e9, 0, 0])


def test_triu_with_inf_causal_mask() raises:
    var device = Device()
    var mask = Tensor.full(device, (1, 1, 4, 4), -1e9).triu(1)
    assert_true(mask.shape(0) == 1 and mask.shape(1) == 1 and mask.shape(2) == 4 and mask.shape(3) == 4)
    var flat = mask.flatten()
    var data = flat.to_list()
    for i in range(len(data)):
        var row = i // 4
        var col = i % 4
        if col > row:
            assert_true(data[i] == Float32(-1e9))
        else:
            assert_true(data[i] == Float32(0))


def test_full_with_float32_literal() raises:
    var device = Device()
    var x = Tensor.full(device, (2, 3), Float32(3.5))
    assert_allclose(x, Tensor.full(device, (2, 3), 3.5))


def test_full_with_large_negative() raises:
    var device = Device()
    var x = Tensor.full(device, (2, 2), Float32(-1e9))
    var mask = x.triu(1)
    assert_allclose(mask, [Float32(0), -1e9, 0, 0])


def test_full_like() raises:
    var device = Device()
    var x = Tensor.full(device, (2, 3, 4), Float32(1), dtype=DType.float16)
    var y = Tensor.full_like(x, Float32(7))
    assert_true(y.shape() == x.shape())
    assert_true(y.dtype == x.dtype)


def test_zeros_like_static() raises:
    var device = Device()
    var x = Tensor.full(device, (2, 3), Float32(5), dtype=DType.float16)
    var z = Tensor.zeros_like(x)
    assert_true(z.shape() == x.shape())
    assert_true(z.dtype == x.dtype)
    assert_allclose(z, Tensor.full(device, (2, 3), Float32(0), dtype=DType.float16))


def test_zeros_like_instance() raises:
    var device = Device()
    var x = Tensor.full(device, (3, 4), Float32(7))
    var z = x.zeros_like()
    assert_true(z.shape() == x.shape())
    assert_true(z.dtype == x.dtype)
    assert_allclose(z, Tensor.full(device, (3, 4), Float32(0)))


def test_flatten_all() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6], (2, 3))
    var flat = x.flatten()
    assert_true(flat.shape(0) == 6)
    assert_allclose(flat, [Float32(1), 2, 3, 4, 5, 6])


def test_flatten_partial() raises:
    var device = Device()
    var x = Tensor.ones(device, (2, 3, 4))
    var flat = x.flatten(1, 2)
    assert_true(flat.shape(0) == 2 and flat.shape(1) == 12)


def _causal_mask(device: Device, scores: Tensor) raises -> Tensor:
    return Tensor.full_like(scores, neg_inf[DType.float32]()).triu(1)


def test_flash_attn_output_shape() raises:
    var device = Device()
    var B = 2
    var H = 4
    var T = 8
    var Dh = 16
    var Q = Tensor.randn(device, (B, H, T, Dh), seed=0)
    var K = Tensor.randn(device, (B, H, T, Dh), seed=1)
    var V = Tensor.randn(device, (B, H, T, Dh), seed=2)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var scores = (Q @ K.transpose(-2, -1)) * scale
    var out = (scores + _causal_mask(device, scores)).softmax() @ V
    assert_true(out.shape(0) == B and out.shape(1) == H and out.shape(2) == T and out.shape(3) == Dh)


def test_flash_attn_causal_first_token_equals_v0() raises:
    # With a causal mask, position 0 can only attend to itself.
    # B=H=1 so out is (1,1,T,Dh). Flatten to (T,Dh) to slice position 0.
    var device = Device()
    var B = 1
    var H = 1
    var T = 4
    var Dh = 8
    var Q = Tensor.randn(device, (B, H, T, Dh), seed=3)
    var K = Tensor.randn(device, (B, H, T, Dh), seed=4)
    var V = Tensor.randn(device, (B, H, T, Dh), seed=5)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var scores = (Q @ K.transpose(-2, -1)) * scale
    var out = (scores + _causal_mask(device, scores)).softmax() @ V
    var out_seq = out.reshape((T, Dh))
    var v_seq = V.reshape((T, Dh))
    assert_allclose(out_seq[0:1], v_seq[0:1], tol=1e-3)


def test_flash_attn_known_values_t2_h1_dh1() raises:
    var device = Device()
    var Q = Tensor.ones(device, (1, 1, 2, 1))
    var K = Tensor.ones(device, (1, 1, 2, 1))
    var V = Tensor(device, [Float32(1), -1], (1, 1, 2, 1))
    var scores = (Q @ K.transpose(-2, -1)) * Float32(1.0)
    var out = (scores + _causal_mask(device, scores)).softmax() @ V
    assert_allclose(out, [Float32(1), 0], tol=1e-3)


def test_flash_attn_no_mask_uniform_qk_averages_v() raises:
    # Q=K=zeros → scores=0 → uniform softmax = 1/T.
    # V is constant (all 3), so output = 3 everywhere.
    var device = Device()
    var B = 1
    var H = 1
    var T = 4
    var Dh = 4
    var Q = Tensor.full(device, (B, H, T, Dh), Float32(0))
    var K = Tensor.full(device, (B, H, T, Dh), Float32(0))
    var V = Tensor.full(device, (B, H, T, Dh), Float32(3))
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var no_mask = Tensor.full(device, (B, H, T, T), Float32(0))
    var out = (Q @ K.transpose(-2, -1) * scale + no_mask).softmax() @ V
    assert_allclose(out, Tensor.full(device, (B, H, T, Dh), Float32(3)), tol=1e-3)


def test_flash_attn_multi_batch_multi_head_shape() raises:
    # Mirrors the reshape/transpose pattern from tinyshakespear
    var device = Device()
    var B = 2
    var T = 6
    var D = 32
    var H = 4
    var Dh = D // H
    var x = Tensor.randn(device, (B, T, D), seed=10)
    var Q = x.reshape((B, T, H, Dh)).transpose(1, 2)  # (B, H, T, Dh)
    var K = x.reshape((B, T, H, Dh)).transpose(1, 2)
    var V = x.reshape((B, T, H, Dh)).transpose(1, 2)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var scores = (Q @ K.transpose(-2, -1)) * scale
    var out = (scores + _causal_mask(device, scores)).softmax() @ V
    var ctx = out.transpose(1, 2).reshape((B, T, D))
    assert_true(ctx.shape(0) == B and ctx.shape(1) == T and ctx.shape(2) == D)


def test_sdpa_causal_matches_unfused() raises:
    # Compare the fused FLASH_ATTN kernel against the unfused scalar path.
    # simplifier=False bypasses GPU rewrites so the reference runs the slow
    # but definitely-correct element-wise softmax+matmul implementation.
    var device = Device()
    var B = 2
    var H = 4
    var T = 8
    var Dh = 16
    var Q = Tensor.randn(device, (B, H, T, Dh), seed=40)
    var K = Tensor.randn(device, (B, H, T, Dh), seed=41)
    var V = Tensor.randn(device, (B, H, T, Dh), seed=42)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var scores = (Q @ K.transpose(-2, -1)) * scale
    var unfused = (scores + _causal_mask(device, scores)).softmax() @ V
    var fused = Q.scaled_dot_product_attention(K, V, is_causal=True)
    # fused.value() runs through the FLASH_ATTN kernel. unfused.value(simplifier=False)
    # bypasses GPU rewrites and runs the scalar softmax+matmul fallback.
    assert_allclose(unfused.value(simplifier=False), unfused.op.layout(), fused, tol=1e-3)


def test_sdpa_causal_transposed_bshd_large_matches_unfused() raises:
    # Mirrors the fused training path: contiguous BSHD leaves are transposed to
    # BHSD immediately before SDPA, then GPU_REWRITES lowers to FLASH_ATTN.
    var device = Device()
    var B = 1
    var T = 16
    var H = 1
    var Dh = 64
    var Q_bshd = Tensor.randn(device, (B, T, H, Dh), seed=140)
    var K_bshd = Tensor.randn(device, (B, T, H, Dh), seed=141)
    var V_bshd = Tensor.randn(device, (B, T, H, Dh), seed=142)
    var Q = Q_bshd.transpose(1, 2)
    var K = K_bshd.transpose(1, 2)
    var V = V_bshd.transpose(1, 2)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var scores = (Q @ K.transpose(-2, -1)) * scale
    var unfused = (scores + _causal_mask(device, scores)).softmax() @ V
    var fused = Q.scaled_dot_product_attention(K, V, is_causal=True)
    assert_allclose(unfused.value(simplifier=False), unfused.op.layout(), fused, tol=5e-3)

    # Position 0 is a stronger causal invariant: it can only attend to V[0].
    assert_allclose(fused.reshape((T, Dh))[0:1], V.reshape((T, Dh))[0:1], tol=5e-3)


def test_sdpa_causal_f16_transposed_bshd_d128_first_token_equals_v0() raises:
    var device = Device()
    var B = 1
    var T = 16
    var H = 1
    var Dh = 128
    var Q_bshd = Tensor.randn(device, (B, T, H, Dh), seed=240).cast(DType.float16)
    var K_bshd = Tensor.randn(device, (B, T, H, Dh), seed=241).cast(DType.float16)
    var V_bshd = Tensor.randn(device, (B, T, H, Dh), seed=242).cast(DType.float16)
    var Q = Q_bshd.transpose(1, 2)
    var K = K_bshd.transpose(1, 2)
    var V = V_bshd.transpose(1, 2)
    var fused = Q.scaled_dot_product_attention(K, V, is_causal=True)
    assert_allclose(fused.reshape((T, Dh))[0:1], V.reshape((T, Dh))[0:1], tol=7e-2)


def test_sdpa_explicit_mask_f16_d128_matches_unfused() raises:
    var device = Device()
    var B = 1
    var H = 1
    var T = 16
    var Dh = 128
    var Q = Tensor.randn(device, (B, H, T, Dh), seed=250).cast(DType.float16)
    var K = Tensor.randn(device, (B, H, T, Dh), seed=251).cast(DType.float16)
    var V = Tensor.randn(device, (B, H, T, Dh), seed=252).cast(DType.float16)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var mask = Tensor.full(device, (B, H, T, T), Float32(0)).cast(DType.float16)
    var unfused = (Q @ K.transpose(-2, -1) * scale + mask).softmax() @ V
    var fused = Q.scaled_dot_product_attention(K, V, attn_mask=mask)
    assert_allclose(unfused.value(simplifier=False), unfused.op.layout(), fused, tol=7e-2)


def test_sdpa_explicit_mask_matches_unfused() raises:
    var device = Device()
    var B = 1
    var H = 2
    var T = 6
    var Dh = 8
    var Q = Tensor.randn(device, (B, H, T, Dh), seed=43)
    var K = Tensor.randn(device, (B, H, T, Dh), seed=44)
    var V = Tensor.randn(device, (B, H, T, Dh), seed=45)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var mask = Tensor.full(device, (B, H, T, T), Float32(0))
    var unfused = (Q @ K.transpose(-2, -1) * scale + mask).softmax() @ V
    var fused = Q.scaled_dot_product_attention(K, V, attn_mask=mask)
    assert_allclose(unfused.value(simplifier=False), unfused.op.layout(), fused, tol=1e-3)


def test_sdpa_3d_input_output_shape() raises:
    # 3D (B, T, D) → unsqueeze H=1 inside, result is (B, T, D)
    var device = Device()
    var B = 2
    var T = 6
    var D = 16
    var Q = Tensor.randn(device, (B, T, D), seed=46)
    var K = Tensor.randn(device, (B, T, D), seed=47)
    var V = Tensor.randn(device, (B, T, D), seed=48)
    var out = Q.scaled_dot_product_attention(K, V, is_causal=True)
    assert_true(out.shape(0) == B and out.shape(1) == T and out.shape(2) == D)


def test_sdpa_3d_matches_4d_unsqueeze() raises:
    # 3D path and manually unsqueezed 4D path must agree.
    var device = Device()
    var B = 1
    var T = 4
    var D = 8
    var Q = Tensor.randn(device, (B, T, D), seed=49)
    var K = Tensor.randn(device, (B, T, D), seed=50)
    var V = Tensor.randn(device, (B, T, D), seed=51)
    var out_3d = Q.scaled_dot_product_attention(K, V, is_causal=True)
    var Q4 = Q.reshape((B, 1, T, D))
    var K4 = K.reshape((B, 1, T, D))
    var V4 = V.reshape((B, 1, T, D))
    var out_4d = Q4.scaled_dot_product_attention(K4, V4, is_causal=True).reshape((B, T, D))
    assert_allclose(out_3d, out_4d, tol=1e-3)


def test_sdpa_causal_long_seq_prefix_mean() raises:
    # S=128 spans two 64-row MMA tiles per block, exercising the multi-tile
    # online-softmax path. With Q = K = 0 causal attention is a prefix mean:
    # O[i] = mean(V[0..i]).
    var device = Device()
    var S = 128
    var Dh = 8
    var Q = Tensor.zeros(device, (1, S, 1, Dh))
    var K = Tensor.zeros(device, (1, S, 1, Dh))
    var V = Tensor.randn(device, (1, S, 1, Dh), seed=150)
    var out = Q.transpose(1, 2).scaled_dot_product_attention(K.transpose(1, 2), V.transpose(1, 2), is_causal=True)
    var v_vals = V.to_list()  # (1, S, 1, Dh) row-major == (S, Dh)
    var expected = List[Float32]()
    var prefix = List[Float32]()
    for _ in range(Dh):
        prefix.append(Float32(0))
    for i in range(S):
        for d in range(Dh):
            prefix[d] += v_vals[i * Dh + d]
            expected.append(prefix[d] / Float32(i + 1))
    assert_allclose(out.reshape((S, Dh)), expected, tol=Float32(2e-2))


def test_sdpa_causal_f16_d128_multi_tile_matches_unfused() raises:
    # f16 causal at D == D_BUCKET(128) exercises the half-MMA forward kernel's
    # causal path (previously asserted off, falling back to TF32+convert).
    # S=80 spans three 32-row KV tiles with a ragged tail. H=2 exercises the
    # BSHD row stride.
    var device = Device()
    var B = 1
    var T = 80
    var H = 2
    var Dh = 128
    var Q_bshd = Tensor.randn(device, (B, T, H, Dh), seed=260).cast(DType.float16)
    var K_bshd = Tensor.randn(device, (B, T, H, Dh), seed=261).cast(DType.float16)
    var V_bshd = Tensor.randn(device, (B, T, H, Dh), seed=262).cast(DType.float16)
    var Q = Q_bshd.transpose(1, 2)
    var K = K_bshd.transpose(1, 2)
    var V = V_bshd.transpose(1, 2)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var scores = (Q @ K.transpose(-2, -1)) * scale
    var unfused = (scores + _causal_mask(device, scores)).softmax() @ V
    var fused = Q.scaled_dot_product_attention(K, V, is_causal=True)
    assert_allclose(unfused.value(simplifier=False), unfused.op.layout(), fused, tol=7e-2)


def test_sdpa_custom_scale_matches_unfused() raises:
    # A user-provided scale must reach the fused kernel. The rewrite once
    # recomputed 1/sqrt(d_head) regardless, which this test would catch.
    var device = Device()
    var B = 1
    var T = 16
    var H = 2
    var Dh = 16
    var Q = Tensor.randn(device, (B, T, H, Dh), seed=280).transpose(1, 2)
    var K = Tensor.randn(device, (B, T, H, Dh), seed=281).transpose(1, 2)
    var V = Tensor.randn(device, (B, T, H, Dh), seed=282).transpose(1, 2)
    var custom = Float32(0.5)
    var unfused = ((Q @ K.transpose(-2, -1)) * custom).softmax() @ V
    var fused = Q.scaled_dot_product_attention(K, V, scale=custom)
    # 5e-3: TF32 attention kernel vs full-f32 unfused chain, with the larger
    # scale sharpening the softmax. A dropped scale would miss by ~0.1-1.0.
    assert_allclose(unfused.value(simplifier=False), unfused.op.layout(), fused, tol=5e-3)


def test_sdpa_causal_multi_head_matches_unfused() raises:
    # H > 1 through the fused path: Q/K/V reach the kernel as BSHD buffers
    # (row stride H*D), while O is produced BHSD. Catches head/sequence
    # stride mix-ups that are invisible at H == 1.
    var device = Device()
    var B = 2
    var T = 16
    var H = 4
    var Dh = 16
    var Q_bshd = Tensor.randn(device, (B, T, H, Dh), seed=170)
    var K_bshd = Tensor.randn(device, (B, T, H, Dh), seed=171)
    var V_bshd = Tensor.randn(device, (B, T, H, Dh), seed=172)
    var Q = Q_bshd.transpose(1, 2)
    var K = K_bshd.transpose(1, 2)
    var V = V_bshd.transpose(1, 2)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var scores = (Q @ K.transpose(-2, -1)) * scale
    var unfused = (scores + _causal_mask(device, scores)).softmax() @ V
    var fused = Q.scaled_dot_product_attention(K, V, is_causal=True)
    assert_allclose(unfused.value(simplifier=False), unfused.op.layout(), fused, tol=5e-3)


def test_sdpa_f16_multi_head_mask_matches_unfused() raises:
    # f16 half-MMA path with H > 1: rows are strided by H*D, so the
    # synchronous smem load path (not the H==1 async copy) must be used.
    var device = Device()
    var B = 1
    var T = 16
    var H = 2
    var Dh = 64
    var Q_bshd = Tensor.randn(device, (B, T, H, Dh), seed=180).cast(DType.float16)
    var K_bshd = Tensor.randn(device, (B, T, H, Dh), seed=181).cast(DType.float16)
    var V_bshd = Tensor.randn(device, (B, T, H, Dh), seed=182).cast(DType.float16)
    var Q = Q_bshd.transpose(1, 2)
    var K = K_bshd.transpose(1, 2)
    var V = V_bshd.transpose(1, 2)
    var mask = Tensor.randn(device, (B, H, T, T), seed=183).cast(DType.float16)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var unfused = (Q @ K.transpose(-2, -1) * scale + mask).softmax() @ V
    var fused = Q.scaled_dot_product_attention(K, V, attn_mask=mask)
    assert_allclose(unfused.value(simplifier=False), unfused.op.layout(), fused, tol=7e-2)


def test_sdpa_nonzero_mask_matches_unfused() raises:
    # Non-zero additive mask (bias): fused forward must add the bias inside
    # the online softmax, not just tolerate a zero-filled buffer.
    var device = Device()
    var B = 1
    var T = 8
    var H = 2
    var Dh = 16
    var Q_bshd = Tensor.randn(device, (B, T, H, Dh), seed=160)
    var K_bshd = Tensor.randn(device, (B, T, H, Dh), seed=161)
    var V_bshd = Tensor.randn(device, (B, T, H, Dh), seed=162)
    var Q = Q_bshd.transpose(1, 2)
    var K = K_bshd.transpose(1, 2)
    var V = V_bshd.transpose(1, 2)
    var mask = Tensor.randn(device, (B, H, T, T), seed=163)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var unfused = (Q @ K.transpose(-2, -1) * scale + mask).softmax() @ V
    var fused = Q.scaled_dot_product_attention(K, V, attn_mask=mask)
    assert_allclose(unfused.value(simplifier=False), unfused.op.layout(), fused, tol=5e-3)


def test_sdpa_causal_first_token_equals_v0() raises:
    # Position 0 can only attend to itself under a causal mask → out[0] == V[0]
    var device = Device()
    var B = 1
    var H = 1
    var T = 4
    var Dh = 8
    var Q = Tensor.randn(device, (B, H, T, Dh), seed=52)
    var K = Tensor.randn(device, (B, H, T, Dh), seed=53)
    var V = Tensor.randn(device, (B, H, T, Dh), seed=54)
    var out = Q.scaled_dot_product_attention(K, V, is_causal=True).reshape((T, Dh))
    var v_seq = V.reshape((T, Dh))
    assert_allclose(out[0:1], v_seq[0:1], tol=1e-3)


def test_manual_seed_replays() raises:
    var device = Device()
    device.manual_seed(7)
    var a = Tensor.randn(device, (64,))
    device.manual_seed(7)
    var b = Tensor.randn(device, (64,))
    assert_allclose(a, b)


def test_randn_is_random() raises:
    var device = Device()
    var a = Tensor.randn(device, (1024, 1024))
    var b = Tensor.randn(device, (1024, 1024))
    with assert_raises():
        assert_allclose(a, b)


# ===-------------------------------------------------------------------===#
# Tensor literal dtype inference & promotion
# ===-------------------------------------------------------------------===#


def test_literal_bare_ints_store_int64() raises:
    var device = Device()
    var t = Tensor(device, [1, 2, 3], (1, 3))
    assert_true(t.dtype == DType.int64)
    var vals = t.to_list[DType.int64]()
    for i in range(3):
        assert_equal(vals[i], Int64(i + 1))


def test_literal_bare_floats_store_float32() raises:
    var device = Device()
    var t = Tensor(device, [1.5, 2.5], (1, 2))
    assert_true(t.dtype == DType.float32)
    assert_allclose(t, [Float32(1.5), 2.5])


def test_literal_pinned_dtypes_preserved() raises:
    var device = Device()
    var f16 = Tensor(device, [Float16(1), 2], (1, 2))
    var i64 = Tensor(device, [Int64(1), 2], (1, 2))
    assert_true(f16.dtype == DType.float16)
    assert_true(i64.dtype == DType.int64)


def test_mixed_dtype_add_promotes_to_float() raises:
    var device = Device()
    var a = Tensor.ones(device, (1, 3))
    var b = Tensor(device, [1, 2, 3], (1, 3))
    var c = a + b
    assert_true(c.dtype == DType.float32)
    assert_allclose(c, [Float32(2), 3, 4])


def test_f16_f32_add_promotes_to_float32() raises:
    var device = Device()
    var a = Tensor(device, [Float16(1), 2], (1, 2))
    var b = Tensor(device, [Float32(0.5), 0.5], (1, 2))
    var c = a + b
    assert_true(c.dtype == DType.float32)
    assert_allclose(c, [Float32(1.5), 2.5])


def main() raises:
    comptime assert has_accelerator(), "GPU required to run tensor op tests"
    TestSuite.discover_tests[__functions_in_module()]().run()
