# Day 1 Package: Tiny Autograd (Scalar) — PR Workflow Ready

> 목표(40분): 스칼라 연산 그래프와 역전파를 직접 구현해 보며, GitHub PR 워크플로우까지 한 번에 체득합니다. 오늘은 **정답 없이 스켈레톤+테스트**만 제공합니다. 테스트를 통과하도록 구현해 보세요.

---

## 파일 트리

```
my-llm-lab/
  00_common/
  10_core/
    autograd_tiny/
      __init__.py
      value.py                # 구현 대상 (스켈레톤)
  50_eval/
    smoke.py                  # 스모크 실행 예시(옵션)
  tests/
    test_value.py             # 정답 검증(수치미분 대조)
  .github/
    workflows/ci.yml          # CI: ruff/black/mypy/pytest
    PULL_REQUEST_TEMPLATE.md
  Makefile
  pyproject.toml
  requirements.txt
  README.md
```

---

## README.md

````md
# Day 1 — Tiny Autograd (Scalar)

**목표:**
- 스칼라용 자동미분 엔진의 최소구성 구현
- 연산자 오버로딩(`+`, `*`, `tanh`, 선택: `relu`)과 `backward()`
- 수치미분과의 상대오차 비교(테스트 제공)

## 진행 순서(권장)
1. 테스트 먼저 실행해 실패를 확인합니다.
   ```bash
   make setup  # 의존성 설치
   make test   # pytest -q
````

2. `10_core/autograd_tiny/value.py`의 TODO를 채웁니다.
3. 다시 `make test`로 통과 여부를 확인합니다.
4. PR을 올리고, PR 템플릿에 **의도/결과/리스크**를 적습니다.

## 구현 요구사항

- `Value` 클래스
  - 속성: `data: float`, `grad: float=0.0`, `_prev: set[Value]`, `_backward: Callable[[], None]`
  - 연산: `__add__`, `__mul__`, `tanh()` (선택: `relu()`), `__radd__`, `__rmul__`
  - `backward()`는 그래프를 위상정렬로 순회하며 chain rule 적용
- 테스트:
  - 함수: `f(a,b) = (a*b + a) * tanh(b)`
  - 수치미분과의 상대오차가 `1e-4` 이하

## 힌트

- 각 연산 시 **로컬 미분 규칙**을 `_backward` 클로저로 저장
- `backward()`에서 `topo` 리스트는 DFS로 쌓아 **역순**으로 실행
- `tanh' (x) = 1 - tanh(x)^2`

## 명령어 모음

```bash
make setup   # 의존성 설치
make fmt     # ruff/black/mypy 체크
make test    # pytest -q
python 50_eval/smoke.py
```

## PR 체크리스트(요약)

-

````

---

## requirements.txt
```txt
numpy>=1.26
pytest>=8.2
ruff>=0.5
black>=24.4
mypy>=1.10
rich>=13.7
````

---

## pyproject.toml

```toml
[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
line-length = 100
select = ["E","F","I","B","UP","SIM","PL"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
warn_unused_ignores = true
warn_redundant_casts = true
warn_unused_configs = true
strict_optional = true

[tool.pytest.ini_options]
addopts = "-q"
```

---

## Makefile

```makefile
setup: ; pip install -U pip && pip install -r requirements.txt
fmt:   ; ruff check . && black --check . && mypy .
test:  ; pytest -q
smoke: ; python 50_eval/smoke.py
```

---

## .github/workflows/ci.yml

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -U pip && pip install -r requirements.txt
      - run: ruff check .
      - run: black --check .
      - run: mypy .
      - run: pytest -q
```

---

## .github/PULL\_REQUEST\_TEMPLATE.md

```md
## 목적/문제정의

## 변경 사항

## 재현 방법
```

make test

```

## 결과/로그 스냅샷

## 리스크/추가 TODO

- [ ] CI(ruff/black/mypy/pytest) 통과
```

---

## 10\_core/autograd\_tiny/**init**.py

```python
"""Tiny scalar autograd package."""
```

---

## 10\_core/autograd\_tiny/value.py (구현 대상 스켈레톤)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable, Set
import math

@dataclass
class Value:
    data: float
    _prev: Set["Value"] = field(default_factory=set, repr=False)
    _backward: Callable[[], None] = field(default=lambda: None, repr=False)
    grad: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.data, (int, float)):
            raise TypeError("Value.data must be a number")
        self.data = float(self.data)

    # ---- core ops ----
    def __add__(self, other: float | "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, {self, other})

        def _backward() -> None:
            # TODO: d(out)/d(self) = 1, d(out)/d(other) = 1
            raise NotImplementedError

        out._backward = _backward
        return out

    def __radd__(self, other: float | "Value") -> "Value":
        return self.__add__(other)

    def __mul__(self, other: float | "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, {self, other})

        def _backward() -> None:
            # TODO: product rule
            raise NotImplementedError

        out._backward = _backward
        return out

    def __rmul__(self, other: float | "Value") -> "Value":
        return self.__mul__(other)

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, {self})

        def _backward() -> None:
            # TODO: dtanh/dx = 1 - tanh(x)^2
            raise NotImplementedError

        out._backward = _backward
        return out

    # optional: relu
    def relu(self) -> "Value":
        out = Value(self.data if self.data > 0 else 0.0, {self})

        def _backward() -> None:
            # TODO: dReLU/dx = 1(x>0)
            raise NotImplementedError

        out._backward = _backward
        return out

    # ---- autodiff ----
    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # convenience
    def __repr__(self) -> str:  # pragma: no cover
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"
```

---

## tests/test\_value.py

```python
import math
import random
import numpy as np
import pytest
from 10_core.autograd_tiny.value import Value

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

    assert relerr(ga, fa) < 1e-4
    assert relerr(gb, fb) < 1e-4
```

> 주의: `from 10_core.autograd_tiny.value import Value` 임포트 경로가 동작하려면 프로젝트 루트에서 `pytest`를 실행하세요.

---

## 50\_eval/smoke.py

```python
from 10_core.autograd_tiny.value import Value


def main() -> None:
    a, b = Value(1.0), Value(2.0)
    y = (a * b + a).tanh()
    try:
        y.backward()
    except NotImplementedError:
        print("Not implemented yet — fill in value.py TODOs first.")
        return
    print("y=", y)
    print("a.grad=", a.grad, "b.grad=", b.grad)


if __name__ == "__main__":
    main()
```

---

## 사용법(요약)

```bash
# 1) 의존성 설치
make setup

# 2) 테스트(실패를 먼저 확인)
make test

# 3) 구현: 10_core/autograd_tiny/value.py 의 TODO 채우기
#    - __add__, __mul__, tanh(), relu(), backward() 내부 로직

# 4) 다시 테스트
make test

# 5) 스모크 실행
python 50_eval/smoke.py
```

---

## 다음 할 일

- `feature/day1-autograd` 브랜치로 작업 → PR 생성
- PR 템플릿에 의도/결과 기록, CI 통과 확인 후 머지
- Day 2에서 `softmax(log-sum-exp)`와 `cross-entropy` 벡터화 연산으로 확장 예정

