from __future__ import annotations

import math
from typing import Callable


class Value:
    def __init__(self, data: float, _prev: set[Value] = None, _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev = _prev or set()
        self._op = _op

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    # ---- core ops ----
    def __add__(self, other: float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, {self, other}, "+")

        def _backward() -> None:
            # 덧셈의 미분: d(out)/d(self) = 1, d(out)/d(other) = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float | Value) -> Value:
        return self.__add__(other)

    def __mul__(self, other: float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, {self, other}, "*")

        def _backward() -> None:
            # 곱셈의 미분 (product rule): d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float | Value) -> Value:
        return self.__mul__(other)

    
    # 연습: 빼기와 나누기 구현
    def __sub__(self, other):
        # TODO: 빼기 구현
        # 힌트: a - b = a + (-b)
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, {self, other})

        def _backward() -> None:
            self.grad += out.grad * 1.0
            other.grad += out.grad * -1.0

        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        return self.__sub__(other)

    def __truediv__(self, other):
        # TODO: 나누기 구현
        # 힌트: a / b의 미분
        # d(a/b)/da = 1/b
        # d(a/b)/db = -a/b^2
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, {self, other})

        def _backward() -> None:
            self.grad += out.grad * 1.0 / other.data
            other.grad += out.grad * -self.data / other.data**2

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __pow__(self, other: float | int) -> Value:
        """거듭제곱 연산: self ** other
        
        거듭제곱의 미분 규칙:
        - f(x) = x^n 일 때, f'(x) = n * x^(n-1)
        - 따라서 d(self^other)/d(self) = other * self^(other-1)
        """
        assert isinstance(other, (int, float)), "지수는 숫자여야 합니다"
        out = Value(self.data ** other, {self}, f"**{other}")

        def _backward() -> None:
            # 거듭제곱의 미분: d(x^n)/dx = n * x^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, {self}, "tanh")

        def _backward() -> None:
            # tanh의 미분: dtanh/dx = 1 - tanh(x)^2
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    # optional: relu
    def relu(self) -> Value:
        out = Value(self.data if self.data > 0 else 0.0, {self}, "relu")

        def _backward() -> None:
            # ReLU의 미분: dReLU/dx = 1 if x > 0 else 0
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    # ---- autodiff ----
    def backward(self) -> None:
        # 위상 정렬로 그래프 순회
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0  # 출력 노드의 gradient는 1로 시작
        for v in reversed(topo):
            v._backward()  # 역순으로 backpropagation

    # convenience
    def __repr__(self) -> str:  # pragma: no cover
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"
