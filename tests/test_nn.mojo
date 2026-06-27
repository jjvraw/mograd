from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal
from std.math import abs

from mograd import Tensor, Device
from mograd.testing import assert_allclose
import mograd.nn as nn


def test_linear_lazy_init() raises:
    var device = Device()
    var l = nn.Linear(4, 2)
    var is_none = False
    if not l._weight[]:
        is_none = True
    if not is_none:
        raise Error("weight should be None before first call")
    var x = Tensor(device, [Float32(1), 0, 0, 0], (1, 4))
    _ = l(x)
    var is_set = False
    if l._weight[]:
        is_set = True
    if not is_set:
        raise Error("weight should be initialized after first call")
    var layout = l._weight[].value()
    if layout.shape(0) != 2 or layout.shape(1) != 4:
        raise Error("weight shape should be [2, 4], got " + String(layout.shape(0)) + "x" + String(layout.shape(1)))


def test_sgd_updates_weights() raises:
    var device = Device()
    var l = nn.Linear(4, 2)
    var opt = nn.SGD(l.parameters(), lr=Float32(0.1))
    var x = Tensor(device, [Float32(1), 2, 3, 4], (1, 4))
    var logits = l(x)
    var params = opt.params()
    var before = params[0].to_list()
    var loss = logits.sum()
    var grads = loss.gradient(params)
    opt.step(grads)
    var after = opt.params()[0].to_list()
    var changed = False
    for i in range(len(before)):
        if abs(before[i] - after[i]) > Float32(1e-6):
            changed = True
    if not changed:
        raise Error("SGD should update at least one weight")


def test_embedding_lazy_init() raises:
    var device = Device()
    var emb = nn.Embedding(10, 4)
    var is_none = False
    if not emb._weight[]:
        is_none = True
    if not is_none:
        raise Error("weight should be None before first call")
    var indices = Tensor(device, [Int64(1), 3], (2,))
    var out = emb(indices)
    var is_set = False
    if emb._weight[]:
        is_set = True
    if not is_set:
        raise Error("weight should be initialized after first call")
    if out.shape(0) != 2 or out.shape(1) != 4:
        raise Error("output shape should be [2, 4], got " + String(out.shape(0)) + "x" + String(out.shape(1)))


def test_embedding_lookup_matches_weight_rows() raises:
    var device = Device()
    var emb = nn.Embedding(5, 3)
    var indices = Tensor(device, [Int64(0)], (1,))
    _ = emb(indices)
    var weight = emb._weight[].value()
    var indices2 = Tensor(device, [Int64(2), 4], (2,))
    var out = emb(indices2)
    assert_allclose(out, weight.gather(indices2))


def test_embedding_grad_accumulates_repeated_indices() raises:
    var device = Device()
    var emb = nn.Embedding(5, 3)
    var indices = Tensor(device, [Int64(1), 1], (2,))
    var out = emb(indices)
    var loss = out.sum()
    var grads = loss.gradient([emb._weight[].value()])
    var grad_vals = grads[0].to_list()
    # Index 1 is used twice, so its gradient row should accumulate to 2, not 1.
    for col in range(3):
        assert_almost_equal(grad_vals[3 + col], Float32(2))


def test_sgd_arc_sharing() raises:
    var device = Device()
    var l = nn.Linear(4, 2)
    var opt = nn.SGD(l.parameters(), lr=Float32(0.1))
    var x = Tensor(device, [Float32(1), 2, 3, 4], (1, 4))
    _ = l(x)
    var params = opt.params()
    var before = l._weight[].value().to_list()
    var loss = l(x).sum()
    var grads = loss.gradient(params)
    opt.step(grads)
    var after = l._weight[].value().to_list()
    var changed = False
    for i in range(len(before)):
        if abs(before[i] - after[i]) > Float32(1e-6):
            changed = True
    if not changed:
        raise Error("opt.step should update model weights via ArcPointer sharing")


def test_linear_3d_output_shape() raises:
    var device = Device()
    var l = nn.Linear(4, 8, bias=False)
    var x = Tensor(device, [Float32(0), 1, 2, 3, 4, 5, 6, 7], (2, 2, 4))
    var out = l(x)
    if out.shape(0) != 2 or out.shape(1) != 2 or out.shape(2) != 8:
        raise Error(
            "expected (2,2,8), got " + String(out.shape(0)) + "," + String(out.shape(1)) + "," + String(out.shape(2))
        )


def test_linear_4d_output_shape() raises:
    var device = Device()
    var l = nn.Linear(4, 6, bias=False)
    var x = Tensor(device, [Float32(0), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], (2, 2, 2, 4))
    var out = l(x)
    if out.shape(0) != 2 or out.shape(1) != 2 or out.shape(2) != 2 or out.shape(3) != 6:
        raise Error(
            "expected (2,2,2,6), got "
            + String(out.shape(0))
            + ","
            + String(out.shape(1))
            + ","
            + String(out.shape(2))
            + ","
            + String(out.shape(3))
        )


def test_linear_3d_grad_flows() raises:
    var device = Device()
    var l = nn.Linear(4, 8, bias=False)
    var x = Tensor(device, [Float32(0), 1, 2, 3, 4, 5, 6, 7], (2, 1, 4), requires_grad=True)
    var out = l(x)
    var loss = out.sum()
    var grads = loss.gradient([l._weight[].value(), x])
    var w_grad = grads[0]
    var x_grad = grads[1]
    if w_grad.shape(0) != 8 or w_grad.shape(1) != 4:
        raise Error("weight grad shape should be (8, 4)")
    if x_grad.shape(0) != 2 or x_grad.shape(1) != 1 or x_grad.shape(2) != 4:
        raise Error("input grad shape should be (2, 1, 4)")


def test_linear_4d_grad_flows() raises:
    var device = Device()
    var l = nn.Linear(4, 6, bias=False)
    var x = Tensor(
        device, [Float32(0), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], (2, 2, 2, 4), requires_grad=True
    )
    var out = l(x)
    var loss = out.sum()
    var grads = loss.gradient([l._weight[].value(), x])
    var w_grad = grads[0]
    var x_grad = grads[1]
    if w_grad.shape(0) != 6 or w_grad.shape(1) != 4:
        raise Error("weight grad shape should be (6, 4)")
    if x_grad.shape(0) != 2 or x_grad.shape(1) != 2 or x_grad.shape(2) != 2 or x_grad.shape(3) != 4:
        raise Error("input grad shape should be (2, 2, 2, 4)")


def test_adam_updates_weights() raises:
    var device = Device()
    var l = nn.Linear(4, 2, bias=False)
    var opt = nn.Adam(l.parameters(), lr=Float32(0.1))
    var x = Tensor(device, [Float32(1), 2, 3, 4], (1, 4))
    var loss = l(x).sum()
    var grads = loss.gradient(opt.params())
    var before = l._weight[].value().to_list()
    opt.step(grads)
    var after = l._weight[].value().to_list()
    var changed = False
    for i in range(len(before)):
        if abs(before[i] - after[i]) > Float32(1e-6):
            changed = True
    if not changed:
        raise Error("Adam should update at least one weight")


def test_adam_moments_initialised_on_first_step() raises:
    var device = Device()
    var l = nn.Linear(2, 2, bias=False)
    var opt = nn.Adam(l.parameters(), lr=Float32(0.01))
    var x = Tensor(device, [Float32(1), 1], (1, 2))
    var loss = l(x).sum()
    var grads = loss.gradient(opt.params())
    opt.step(grads)
    var m_set = False
    if opt._m[0]:
        m_set = True
    if not m_set:
        raise Error("first moment should be initialised after first step")
    var v_set = False
    if opt._v[0]:
        v_set = True
    if not v_set:
        raise Error("second moment should be initialised after first step")


def test_adam_step_count_increments() raises:
    var device = Device()
    var l = nn.Linear(2, 2, bias=False)
    var opt = nn.Adam(l.parameters(), lr=Float32(0.01))
    var x = Tensor(device, [Float32(1), 1], (1, 2))
    for _ in range(3):
        var loss = l(x).sum()
        var grads = loss.gradient(opt.params())
        opt.step(grads)
    if opt._step != 3:
        raise Error("step count should be 3, got " + String(opt._step))


def test_adam_weight_decay_changes_moments() raises:
    var device = Device()
    var l_no_wd = nn.Linear(2, 2, bias=False)
    var l_wd = nn.Linear(2, 2, bias=False)
    var x = Tensor(device, [Float32(1), 1], (1, 2))
    _ = l_no_wd(x)
    _ = l_wd(x)

    var opt_no_wd = nn.Adam(l_no_wd.parameters(), lr=Float32(0.01), weight_decay=Float32(0.0))
    var opt_wd = nn.Adam(l_wd.parameters(), lr=Float32(0.01), weight_decay=Float32(0.1))

    var loss1 = l_no_wd(x).sum()
    opt_no_wd.step(loss1.gradient(opt_no_wd.params()))

    var loss2 = l_wd(x).sum()
    opt_wd.step(loss2.gradient(opt_wd.params()))

    var m_no_wd = opt_no_wd._m[0].value().to_list()
    var m_wd = opt_wd._m[0].value().to_list()
    var different = False
    for i in range(len(m_no_wd)):
        if abs(m_no_wd[i] - m_wd[i]) > Float32(1e-6):
            different = True
    if not different:
        raise Error("Adam L2 weight decay should change first moment accumulation")


def test_adamw_shrinks_weights() raises:
    var device = Device()
    var l_adam = nn.Linear(2, 2, bias=False)
    var l_adamw = nn.Linear(2, 2, bias=False)
    var x = Tensor(device, [Float32(1), 1], (1, 2))
    _ = l_adam(x)
    _ = l_adamw(x)

    var opt_adam = nn.Adam(l_adam.parameters(), lr=Float32(0.01))
    var opt_adamw = nn.AdamW(l_adamw.parameters(), lr=Float32(0.01), weight_decay=Float32(0.1))

    for _ in range(5):
        var loss1 = l_adam(x).sum()
        opt_adam.step(loss1.gradient(opt_adam.params()))
        var loss2 = l_adamw(x).sum()
        opt_adamw.step(loss2.gradient(opt_adamw.params()))

    var w_adam = l_adam._weight[].value().to_list()
    var w_adamw = l_adamw._weight[].value().to_list()
    var smaller_norm = False
    var norm_adam = Float32(0)
    var norm_adamw = Float32(0)
    for i in range(len(w_adam)):
        norm_adam += w_adam[i] * w_adam[i]
        norm_adamw += w_adamw[i] * w_adamw[i]
    if norm_adamw < norm_adam:
        smaller_norm = True
    if not smaller_norm:
        raise Error("AdamW weight decay should produce smaller weight norms")


def test_layer_norm_fwd_single_row() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4], (1, 4))
    var ln = nn.LayerNorm(4)
    ln._weight[] = Tensor(device, [Float32(1), 2, 1, 2], (4,), requires_grad=True)
    ln._bias[] = Tensor(device, [Float32(0.1), 0.2, 0.1, 0.2], (4,), requires_grad=True)
    assert_allclose(
        ln(x),
        [Float32(-1.241641), Float32(-0.694427), Float32(0.547214), Float32(2.883282)],
        tol=Float32(1e-4),
    )


def test_layer_norm_fwd_two_rows() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, -1, -2, -3, -4], (2, 4))
    var ln = nn.LayerNorm(4)
    ln._weight[] = Tensor(device, [Float32(1), 1, 1, 1], (4,), requires_grad=True)
    ln._bias[] = Tensor(device, [Float32(0), 0, 0, 0], (4,), requires_grad=True)
    assert_allclose(
        ln(x),
        [
            Float32(-1.341641),
            Float32(-0.447214),
            Float32(0.447214),
            Float32(1.341641),
            Float32(1.341641),
            Float32(0.447214),
            Float32(-0.447214),
            Float32(-1.341641),
        ],
        tol=Float32(1e-4),
    )


def test_layer_norm_fwd_constant_row_gives_beta() raises:
    var device = Device()
    var x = Tensor(device, [Float32(3), 3, 3, 3], (1, 4))
    var ln = nn.LayerNorm(4)
    ln._weight[] = Tensor(device, [Float32(1), 1, 1, 1], (4,), requires_grad=True)
    ln._bias[] = Tensor(device, [Float32(0.5), 0.5, 0.5, 0.5], (4,), requires_grad=True)
    assert_allclose(ln(x), [Float32(0.5), Float32(0.5), Float32(0.5), Float32(0.5)], tol=Float32(1e-4))


def main() raises:
    comptime assert has_accelerator(), "GPU required to run nn tests"
    TestSuite.discover_tests[__functions_in_module()]().run()
