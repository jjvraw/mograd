from std.testing import TestSuite, assert_equal, assert_true

from mograd import Shape


def test_construction_variadic() raises:
    var s = Shape(2, 3, 4)
    assert_equal(len(s), 3)
    assert_equal(s[0], 2)
    assert_equal(s[1], 3)
    assert_equal(s[2], 4)


def test_construction_list() raises:
    var dims: List[Int] = [5, 6]
    var s = Shape(dims)
    assert_equal(len(s), 2)
    assert_equal(s[0], 5)
    assert_equal(s[1], 6)


def test_construction_tuple() raises:
    var s: Shape = (8, 16)
    assert_equal(len(s), 2)
    assert_equal(s[0], 8)
    assert_equal(s[1], 16)


def test_negative_index() raises:
    var s = Shape(2, 3, 4)
    assert_equal(s[-1], 4)
    assert_equal(s[-2], 3)
    assert_equal(s[-3], 2)


def test_numel() raises:
    assert_equal(Shape(2, 3, 4).numel(), 24)
    assert_equal(Shape(10).numel(), 10)
    assert_equal(Shape(1, 1, 1).numel(), 1)


def test_strides_row_major() raises:
    var s = Shape(2, 3, 4).strides()
    assert_equal(s[0], 12)
    assert_equal(s[1], 4)
    assert_equal(s[2], 1)


def test_strides_2d() raises:
    var s = Shape(4, 5).strides()
    assert_equal(s[0], 5)
    assert_equal(s[1], 1)


def test_to_list() raises:
    var lst = Shape(7, 8, 9).to_list()
    assert_equal(len(lst), 3)
    assert_equal(lst[0], 7)
    assert_equal(lst[1], 8)
    assert_equal(lst[2], 9)


def test_write_to() raises:
    assert_equal(String(Shape(2, 3)), "[2, 3]")
    assert_equal(String(Shape(5)), "[5]")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
