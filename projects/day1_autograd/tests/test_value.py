import math
import random

import numpy as np
import pytest

from core.autograd import Value

random.seed(0)
np.random.seed(0)


def numerical_grad(f, a0: float, b0: float, eps: float = 1e-6):
    # central difference
    fa = (f(a0 + eps, b0) - f(a0 - eps, b0)) / (2 * eps)
    fb = (f(a0, b0 + eps) - f(a0, b0 - eps)) / (2 * eps)
    return fa, fb


def forward(a0: float, b0: float):
    a, b = Value(a0), Value(b0)
    out = (a * b + a) * b.tanh()
    out.backward()
    return out.data, a.grad, b.grad


def f_scalar(a0: float, b0: float) -> float:
    return (a0 * b0 + a0) * math.tanh(b0)


@pytest.mark.parametrize("a0,b0", [(1.3, -0.7), (0.5, 0.5), (-1.2, 2.0)])
def test_autograd_matches_numerical(a0, b0):
    y, ga, gb = forward(a0, b0)
    fa, fb = numerical_grad(f_scalar, a0, b0)
    # value sanity
    assert not math.isnan(y)

    # relative error check
    def relerr(x, y):
        denom = max(1.0, abs(x), abs(y))
        return abs(x - y) / denom

    TOLERANCE = 1e-4
    assert relerr(ga, fa) < TOLERANCE
    assert relerr(gb, fb) < TOLERANCE
