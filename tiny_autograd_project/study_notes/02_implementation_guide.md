# 🛠️ Tiny Autograd 구현 가이드

## 📝 학습 일지 템플릿

### Day 1: 전체 구조 파악
**날짜**: ___________

**학습 목표**:
- [ ] 노트북 전체 실행
- [ ] Value 클래스 구조 이해
- [ ] 연산 그래프 개념 파악

**핵심 개념**:
```python
# 오늘 배운 핵심 코드
```

**질문/의문점**:
1. 
2. 

**내일 할 일**:

---

## 🔬 단계별 구현 실습

### Step 1: 최소 구현 (Minimal Value)
```python
class SimpleValue:
    """가장 간단한 Value 구현"""
    def __init__(self, data):
        self.data = data
        self.grad = 0
    
    def __repr__(self):
        return f"Value({self.data})"

# 테스트
v = SimpleValue(3.0)
print(v)  # Value(3.0)
```

### Step 2: 덧셈 추가
```python
class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set()
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data)
        out._prev = {self, other}
        
        def _backward():
            self.grad += out.grad  # ∂out/∂self = 1
            other.grad += out.grad  # ∂out/∂other = 1
        
        out._backward = _backward
        return out

# 테스트
a = Value(2)
b = Value(3)
c = a + b
print(c.data)  # 5
```

### Step 3: 역전파 구현
```python
def backward(self):
    # 위상정렬
    topo = []
    visited = set()
    
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    
    build_topo(self)
    
    # 역전파
    self.grad = 1.0
    for v in reversed(topo):
        v._backward()

Value.backward = backward

# 테스트
a = Value(2)
b = Value(3)
c = a + b
c.backward()
print(f"a.grad={a.grad}, b.grad={b.grad}")  # 1, 1
```

## 📊 실습 프로젝트

### Project 1: 간단한 함수 미분
```python
# 목표: f(x) = x² + 2x + 1 의 x=3에서 미분값 구하기
# 답: f'(x) = 2x + 2, f'(3) = 8

x = Value(3.0)
# TODO: 함수 구현
# f = ...
# f.backward()
# assert abs(x.grad - 8.0) < 1e-6
```

### Project 2: 2변수 함수
```python
# 목표: f(x,y) = x*y + x 의 편미분
# ∂f/∂x = y + 1, ∂f/∂y = x

def test_two_vars():
    x = Value(2.0)
    y = Value(3.0)
    # TODO: 구현
    pass
```

### Project 3: 복잡한 함수
```python
# 목표: f(x,y) = tanh(x*y) * (x + y)

def complex_function():
    x = Value(1.0)
    y = Value(2.0)
    # TODO: 구현
    pass
```

## 🧪 테스트 작성하기

### 단위 테스트 템플릿
```python
import math

def test_addition():
    """덧셈 테스트"""
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()
    
    assert c.data == 5.0
    assert a.grad == 1.0
    assert b.grad == 1.0
    print("✅ Addition test passed")

def test_multiplication():
    """곱셈 테스트"""
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    
    assert c.data == 6.0
    assert a.grad == 3.0  # ∂(a*b)/∂a = b
    assert b.grad == 2.0  # ∂(a*b)/∂b = a
    print("✅ Multiplication test passed")

def test_tanh():
    """tanh 테스트"""
    x = Value(0.5)
    y = x.tanh()
    y.backward()
    
    expected_grad = 1 - math.tanh(0.5)**2
    assert abs(x.grad - expected_grad) < 1e-6
    print("✅ Tanh test passed")

# 모든 테스트 실행
if __name__ == "__main__":
    test_addition()
    test_multiplication()
    test_tanh()
    print("🎉 All tests passed!")
```

## 📈 성능 분석

### 시간 복잡도 분석
```python
import time

def benchmark_operations(n):
    """n개의 연산 벤치마크"""
    
    # 선형 그래프
    start = time.time()
    x = Value(1.0)
    for _ in range(n):
        x = x + 1
    x.backward()
    linear_time = time.time() - start
    
    # 이진 트리 그래프
    start = time.time()
    values = [Value(1.0) for _ in range(n)]
    while len(values) > 1:
        new_values = []
        for i in range(0, len(values)-1, 2):
            new_values.append(values[i] + values[i+1])
        if len(values) % 2 == 1:
            new_values.append(values[-1])
        values = new_values
    values[0].backward()
    tree_time = time.time() - start
    
    print(f"Linear graph ({n} ops): {linear_time:.4f}s")
    print(f"Tree graph ({n} ops): {tree_time:.4f}s")

# 테스트
benchmark_operations(100)
benchmark_operations(1000)
```

## 🎯 체크포인트

### Week 1 목표
- [ ] Value 클래스 완전 이해
- [ ] 4가지 기본 연산 구현 (+, -, *, /)
- [ ] backward() 메서드 이해
- [ ] 수치 미분과 비교 검증

### Week 2 목표
- [ ] 활성화 함수 3개 추가 (sigmoid, relu, leaky_relu)
- [ ] 복잡한 함수 5개 테스트
- [ ] 그래프 시각화 구현
- [ ] 메모리 사용량 분석

### Week 3 목표
- [ ] Neuron 클래스 구현
- [ ] Layer 클래스 구현
- [ ] 간단한 MLP 구현
- [ ] XOR 문제 해결

## 💡 트러블슈팅

### 자주 발생하는 문제

#### 1. TypeError: unhashable type
```python
# 문제: Value를 set에 넣을 수 없음
# 해결: __hash__와 __eq__ 구현
def __hash__(self):
    return id(self)
```

#### 2. Gradient가 0으로 유지
```python
# 문제: backward() 후에도 grad가 0
# 체크사항:
# 1. out.grad = 1.0 설정했는지
# 2. _backward 함수가 제대로 연결되었는지
# 3. += 로 gradient 누적하는지
```

#### 3. 잘못된 gradient 값
```python
# 문제: 수치 미분과 다른 값
# 디버깅:
def debug_gradient():
    x = Value(2.0)
    y = x * x  # x²
    y.backward()
    
    print(f"Autograd: {x.grad}")  # 4.0이어야 함
    
    # 수치 미분으로 확인
    h = 1e-5
    numerical = ((2+h)**2 - (2-h)**2) / (2*h)
    print(f"Numerical: {numerical}")
```

## 📚 추가 학습 자료

### 읽기 자료
1. **CS231n Backprop Notes**: [링크](http://cs231n.github.io/optimization-2/)
2. **PyTorch Autograd Tutorial**: [링크](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
3. **Micrograd 원본**: [GitHub](https://github.com/karpathy/micrograd)

### 비디오 강의
1. **Andrej Karpathy - micrograd**: 2시간 30분
2. **3Blue1Brown - Neural Networks**: 시리즈 4편
3. **Fast.ai - Lesson 8**: From scratch implementation

### 연습 문제 사이트
1. **Brilliant.org**: 미적분 인터랙티브 코스
2. **Khan Academy**: 다변수 미적분
3. **MIT OCW 18.06**: 선형대수

## 🏆 마일스톤

### Bronze 🥉 (1주차)
- Value 클래스 이해
- 기본 연산 구현
- 간단한 함수 미분

### Silver 🥈 (2주차)
- 모든 연산 구현
- 복잡한 함수 처리
- 테스트 작성

### Gold 🥇 (3주차)
- 신경망 구현
- XOR 해결
- 최적화 구현

### Platinum 💎 (4주차+)
- 벡터 연산 확장
- CNN/RNN 구현
- 자체 프레임워크 제작