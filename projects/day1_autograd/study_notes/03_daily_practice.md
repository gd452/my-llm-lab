# 📅 일일 실습 계획

## Week 1: 기초 다지기

### Day 1 (월): 노트북 첫 실행
**목표**: 전체 흐름 파악

**오전 (30분)**
```python
# 1. Jupyter 노트북 열기
jupyter notebook tiny_autograd_tutorial.ipynb

# 2. 모든 셀 실행 (Run All)
# 3. 결과 관찰
```

**오후 (30분)**
- 이해 안 되는 부분 메모
- `study_notes/my_questions.md` 파일 생성
- 3가지 주요 질문 작성

### Day 2 (화): Value 클래스 분석
**목표**: 데이터 구조 이해

**실습**
```python
# 새 노트북에서 실험
from _10_core.autograd_tiny.value import Value

# 1. 간단한 연산
a = Value(2.0)
b = Value(3.0)
c = a + b

# 2. 속성 확인
print(f"c.data = {c.data}")
print(f"c._prev = {c._prev}")
print(f"c._op = {c._op}")

# 3. 그래프 구조 이해
def print_graph(v, level=0):
    print("  " * level + f"Value({v.data}, op={v._op})")
    for child in v._prev:
        print_graph(child, level + 1)

print_graph(c)
```

### Day 3 (수): Forward Pass 이해
**목표**: 연산 그래프 구축 과정

**실습 과제**
```python
# 복잡한 함수 만들기
x = Value(2.0)
y = Value(3.0)

# TODO: 다음 함수 구현
# f(x,y) = x² + 2xy + y²
# 힌트: (x+y)² = x² + 2xy + y²

z = # 여기 구현

print(f"Result: {z.data}")
# Expected: 25.0 (5²)
```

### Day 4 (목): Backward Pass 이해
**목표**: 역전파 과정 추적

**디버깅 실습**
```python
# backward() 과정 추적
class DebugValue(Value):
    def backward(self):
        print("=== Starting Backward Pass ===")
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                print(f"Visiting: Value({v.data})")
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        print(f"\nTopological order: {[v.data for v in topo]}")
        
        self.grad = 1.0
        for v in reversed(topo):
            print(f"\nProcessing: Value({v.data})")
            print(f"  Before: grad={v.grad}")
            v._backward()
            print(f"  After: grad={v.grad}")

# 테스트
x = DebugValue(2.0)
y = DebugValue(3.0)
z = x * y
z.backward()
```

### Day 5 (금): 수치 미분과 비교
**목표**: 구현 검증

**검증 코드**
```python
def numerical_diff(f, x, h=1e-5):
    """수치 미분 계산"""
    return (f(x + h) - f(x - h)) / (2 * h)

def verify_gradient(func_name, func, x_val):
    """자동미분과 수치미분 비교"""
    # 자동미분
    x = Value(x_val)
    y = func(x)
    y.backward()
    auto_grad = x.grad
    
    # 수치미분
    num_grad = numerical_diff(lambda v: func(Value(v)).data, x_val)
    
    # 비교
    error = abs(auto_grad - num_grad)
    print(f"{func_name}:")
    print(f"  Auto: {auto_grad:.6f}")
    print(f"  Numerical: {num_grad:.6f}")
    print(f"  Error: {error:.2e}")
    print(f"  ✅ Pass" if error < 1e-4 else f"  ❌ Fail")

# 테스트 함수들
verify_gradient("x²", lambda x: x * x, 3.0)
verify_gradient("x³", lambda x: x * x * x, 2.0)
verify_gradient("tanh(x)", lambda x: x.tanh(), 0.5)
```

### Weekend Project: 미니 신경망
```python
# 주말 과제: 1개 뉴런 구현
class SimpleNeuron:
    def __init__(self):
        self.w = Value(0.5)  # weight
        self.b = Value(0.1)  # bias
    
    def forward(self, x):
        # TODO: wx + b 구현
        pass
    
    def train_step(self, x_data, y_target, lr=0.01):
        # Forward
        y_pred = self.forward(Value(x_data))
        
        # Loss (MSE)
        loss = (y_pred - Value(y_target)) ** 2
        
        # Backward
        loss.backward()
        
        # Update (gradient descent)
        self.w.data -= lr * self.w.grad
        self.b.data -= lr * self.b.grad
        
        # Reset gradients
        self.w.grad = 0
        self.b.grad = 0
        
        return loss.data

# 선형 함수 학습: y = 2x + 1
neuron = SimpleNeuron()
for epoch in range(100):
    loss = neuron.train_step(1.0, 3.0)  # x=1, y=3
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss={loss:.4f}, w={neuron.w.data:.2f}, b={neuron.b.data:.2f}")
```

## Week 2: 심화 학습

### Day 6-7: 새로운 연산 추가
```python
# 구현할 연산들
1. __sub__ (빼기)
2. __truediv__ (나누기)
3. __pow__ (거듭제곱)
4. sigmoid 활성화 함수
5. log, exp 함수
```

### Day 8-9: 복잡한 함수
```python
# 구현할 함수들
1. Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
2. Himmelblau: f(x,y) = (x²+y-11)² + (x+y²-7)²
3. Beale: f(x,y) = (1.5-x+xy)² + (2.25-x+xy²)² + (2.625-x+xy³)²
```

### Day 10: 최적화 알고리즘
```python
class SGD:
    """Stochastic Gradient Descent"""
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
    
    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad
    
    def zero_grad(self):
        for p in self.params:
            p.grad = 0

# 사용 예시
params = [w1, w2, b]
optimizer = SGD(params, lr=0.001)

for epoch in range(100):
    # Forward
    loss = compute_loss()
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Update
    optimizer.step()
```

## Week 3: 프로젝트

### Mini Project 1: XOR 문제
```python
# 2층 신경망으로 XOR 해결
# Input: (0,0), (0,1), (1,0), (1,1)
# Output: 0, 1, 1, 0
```

### Mini Project 2: 회귀 문제
```python
# sin 함수 근사
# 데이터 생성
import numpy as np
X = np.linspace(-np.pi, np.pi, 100)
Y = np.sin(X)

# 3층 신경망으로 학습
```

### Mini Project 3: 분류 문제
```python
# 2D 나선형 데이터 분류
# 2개 클래스, 나선형으로 분포
```

## 📊 진도 체크리스트

### 필수 이해 개념
- [ ] Forward pass와 연산 그래프
- [ ] Chain rule의 원리
- [ ] Backward pass와 gradient 전파
- [ ] 위상정렬의 필요성
- [ ] Gradient 누적 (+=) 이유

### 구현 능력
- [ ] Value 클래스 처음부터 구현
- [ ] 기본 연산 4개 구현
- [ ] 활성화 함수 2개 구현
- [ ] backward() 메서드 구현
- [ ] 간단한 최적화 구현

### 응용 능력
- [ ] 복잡한 함수 미분
- [ ] 뉴런 클래스 구현
- [ ] 손실 함수 구현
- [ ] 학습 루프 작성
- [ ] 수렴 확인

## 🎯 최종 목표

**3주 후 도달 목표:**
1. ✅ Autograd 완전 이해
2. ✅ 간단한 신경망 구현 가능
3. ✅ XOR 문제 해결
4. ✅ PyTorch autograd와 비교 가능
5. ✅ 나만의 미니 프레임워크 제작

**포트폴리오 프로젝트:**
- GitHub에 정리된 코드
- 학습 과정 블로그 포스트
- 구현한 신경망 데모
- 성능 벤치마크 결과