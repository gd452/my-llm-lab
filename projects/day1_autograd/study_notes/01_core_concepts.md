# 📖 자동미분(Autograd) 핵심 개념 정리

## 1. 미분(Differentiation) 기초

### 1.1 도함수(Derivative)란?
- **정의**: 함수의 순간 변화율
- **기하학적 의미**: 접선의 기울기
- **수식**: f'(x) = lim(h→0) [f(x+h) - f(x)] / h

### 1.2 편미분(Partial Derivative)
- **정의**: 다변수 함수에서 한 변수에 대한 미분
- **표기**: ∂f/∂x (f를 x에 대해 편미분)
- **예시**: 
  ```
  f(x,y) = x²y + xy²
  ∂f/∂x = 2xy + y²
  ∂f/∂y = x² + 2xy
  ```

## 2. Chain Rule (연쇄 법칙)

### 2.1 개념
- **합성함수의 미분법칙**
- z = f(g(x)) 일 때: dz/dx = dz/dg × dg/dx

### 2.2 신경망에서의 적용
```python
# 예시: z = (x * y) + x
# x = 2, y = 3 일 때

# Forward pass:
a = x * y  # a = 6
z = a + x  # z = 8

# Backward pass (Chain Rule):
dz/dz = 1           # 출력의 gradient는 1
dz/da = 1           # 덧셈의 로컬 gradient
dz/dx = dz/da * da/dx + 1  # Chain rule + 직접 연결
      = 1 * y + 1 = 4
dz/dy = dz/da * da/dy = 1 * x = 2
```

## 3. 계산 그래프(Computation Graph)

### 3.1 구성 요소
- **노드(Node)**: 값 또는 연산
- **엣지(Edge)**: 데이터 흐름
- **방향**: Forward(순방향) → Backward(역방향)

### 3.2 그래프 예시
```
     x (2) ──┐
              ├─[×]─→ a (6) ──┐
     y (3) ──┘                 ├─[+]─→ z (8)
                      x (2) ──┘
```

### 3.3 위상정렬(Topological Sort)
- **목적**: 의존성 순서대로 노드 처리
- **역전파 시**: 역순으로 gradient 계산

## 4. 자동미분 구현 원리

### 4.1 Forward Pass
```python
# 각 연산마다:
1. 결과값 계산
2. 연산 그래프에 노드 추가
3. 로컬 gradient 저장 (_backward 함수)
```

### 4.2 Backward Pass
```python
# 역전파 알고리즘:
1. 출력 노드의 grad = 1.0
2. 위상정렬 역순으로:
   - 각 노드의 _backward() 실행
   - gradient 누적 (+=)
```

## 5. 주요 연산의 미분

### 5.1 기본 연산
| 연산 | Forward | 로컬 Gradient |
|------|---------|---------------|
| 덧셈 | z = x + y | ∂z/∂x = 1, ∂z/∂y = 1 |
| 곱셈 | z = x × y | ∂z/∂x = y, ∂z/∂y = x |
| 거듭제곱 | z = x^n | ∂z/∂x = n×x^(n-1) |

### 5.2 활성화 함수
| 함수 | Forward | Gradient |
|------|---------|----------|
| tanh | y = tanh(x) | dy/dx = 1 - tanh²(x) |
| ReLU | y = max(0,x) | dy/dx = 1 if x>0 else 0 |
| Sigmoid | y = 1/(1+e^-x) | dy/dx = y(1-y) |

## 6. 구현 체크리스트

### Value 클래스 필수 요소
- [ ] `data`: 실제 값 저장
- [ ] `grad`: gradient 저장
- [ ] `_prev`: 부모 노드들 (set)
- [ ] `_backward`: 역전파 함수
- [ ] `__hash__`, `__eq__`: set에 저장 가능하게

### 연산 구현 패턴
```python
def operation(self, other):
    # 1. Forward 계산
    out = Value(계산_결과, {self, other})
    
    # 2. Backward 함수 정의
    def _backward():
        self.grad += 로컬_gradient * out.grad
        other.grad += 로컬_gradient * out.grad
    
    # 3. 함수 연결
    out._backward = _backward
    return out
```

## 7. 디버깅 팁

### 7.1 Gradient 검증
```python
# 수치 미분과 비교
def check_gradient(f, x, h=1e-5):
    # 수치 미분
    grad_numerical = (f(x+h) - f(x-h)) / (2*h)
    
    # 자동 미분
    y = f(x)
    y.backward()
    grad_auto = x.grad
    
    # 상대 오차
    error = abs(grad_numerical - grad_auto) / max(abs(grad_numerical), 1e-8)
    assert error < 1e-4
```

### 7.2 일반적인 문제들
1. **Gradient 폭발**: 값이 너무 커짐 → 학습률 조정
2. **Gradient 소실**: 값이 0에 가까워짐 → 활성화 함수 변경
3. **NaN 발생**: 0으로 나누기, log(0) 등 → 안정화 기법 적용

## 8. 연습 문제

### Level 1: 기초
1. `__sub__` 구현하기 (힌트: a-b = a+(-b))
2. `__neg__` 구현하기 (부호 반전)
3. `__pow__` 구현하기 (거듭제곱)

### Level 2: 중급
1. `sigmoid` 활성화 함수 추가
2. `log`, `exp` 연산 추가
3. MSE 손실 함수 구현

### Level 3: 고급
1. 행렬 연산으로 확장
2. Batch 처리 지원
3. GPU 연산 시뮬레이션

## 9. 참고 자료

### 필독 자료
- [Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/)
- [Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)
- [Automatic Differentiation in Machine Learning](https://arxiv.org/abs/1502.05767)

### 영상 자료
- [3Blue1Brown - Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
- [Andrej Karpathy - Building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)