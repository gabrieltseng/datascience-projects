from minitorch import operators as op
from hypothesis import given
from hypothesis.strategies import lists
from .strategies import small_floats, assert_close
import pytest


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_add_and_mul(x, y):
    assert_close(op.mul(x, y), x * y)
    assert_close(op.add(x, y), x + y)
    assert_close(op.neg(x), -x)


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a):
    if a > 0:
        assert op.relu(a) == a
    else:
        assert op.relu(a) == 0.0


## Task 0.2
## Property Testing


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_symmetric(x, y):
    assert op.mul(x, y) == op.mul(y, x)
    assert op.add(x, y) == op.add(y, x)
    assert op.max(x, y) == op.max(y, x)
    assert op.eq(x, y) == op.eq(y, x)


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_distribute(x, y, z):
    assert_close(op.mul(z, op.add(x, y)), op.add(op.mul(z, x), op.mul(z, y)))


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_relu_back(x, y):
    """
    Write a test that ensures some other property holds for your functions.
    """
    if x > 0:
        assert op.relu_back(x, y) == y
    else:
        assert op.relu_back(x, y) == 0.0


# HIGHER ORDER


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a, b, c, d):
    assert_close(op.addLists([a, b], [c, d]), [a + c, b + d])


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_property(ls1, ls2):
    assert_close(
        op.add(op.sum(ls1), op.sum(ls2)),
        op.sum([op.add(x, y) for x, y in zip(ls1, ls2)])
    )


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls):
    assert_close(op.sum(ls), sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x, y, z):
    assert_close(op.prod([x, y, z]), x * y * z)
