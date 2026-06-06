from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import abs

from mograd import Tensor, Device
from mograd.testing import assert_allclose
import mograd.nn as nn


def test_linear_lazy_init() raises:
    var ctx = Device()
    var l = nn.Linear(4, 2)
    var is_none = False
    if not l._weight[]:
        is_none = True
    if not is_none:
        raise Error("weight should be None before first call")
    var x = Tensor(ctx, [Float32(1), 0, 0, 0], (1, 4))
    _ = l(x)
    var is_set = False
    if l._weight[]:
        is_set = True
    if not is_set:
        raise Error("weight should be initialized after first call")
    var shape = l._weight[].value().shape()
    if shape[0] != 2 or shape[1] != 4:
        raise Error("weight shape should be [2, 4], got " + String(shape[0]) + "x" + String(shape[1]))


def test_sgd_updates_weights() raises:
    var ctx = Device()
    var l = nn.Linear(4, 2)
    var opt = nn.SGD([l], lr=Float32(0.1))
    var x = Tensor(ctx, [Float32(1), 2, 3, 4], (1, 4))
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


def test_sgd_arc_sharing() raises:
    var ctx = Device()
    var l = nn.Linear(4, 2)
    var opt = nn.SGD([l], lr=Float32(0.1))
    var x = Tensor(ctx, [Float32(1), 2, 3, 4], (1, 4))
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


def main() raises:
    comptime assert has_accelerator(), "GPU required to run nn tests"
    TestSuite.discover_tests[__functions_in_module()]().run()
