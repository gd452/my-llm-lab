# 🧠 Neural Network 핵심 개념

## 1. 신경망의 구성 요소

### 🔸 Neuron (뉴런)
```python
output = activation(Σ(wi * xi) + b)
```

**구성 요소:**
- **입력 (x)**: 이전 층에서 오는 신호
- **가중치 (w)**: 각 입력의 중요도
- **편향 (b)**: 활성화 임계값 조정
- **활성화 함수**: 비선형 변환

**생물학적 비유:**
- 수상돌기 → 입력
- 시냅스 강도 → 가중치
- 세포체 → 가중합 계산
- 축삭 → 출력

### 🔲 Layer (층)
- 같은 입력을 공유하는 뉴런들의 집합
- 각 뉴런은 독립적인 가중치
- 벡터 → 벡터 변환

### 🏗️ Network (네트워크)
- 여러 층의 순차적 연결
- 입력층 → 은닉층(들) → 출력층

## 2. Forward Propagation

### 수학적 표현
```
층 l에서:
z[l] = W[l] × a[l-1] + b[l]  # 선형 변환
a[l] = f(z[l])                # 활성화
```

### 코드 구현 패턴
```python
def forward(self, x):
    for layer in self.layers:
        x = layer(x)
    return x
```

## 3. 활성화 함수

### Tanh
```python
f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
f'(x) = 1 - tanh²(x)
```
- 범위: [-1, 1]
- 중심: 0
- 용도: 은닉층

### ReLU
```python
f(x) = max(0, x)
f'(x) = 1 if x > 0 else 0
```
- 범위: [0, ∞)
- 장점: 계산 간단, gradient vanishing 완화
- 단점: dying ReLU 문제

### Sigmoid
```python
f(x) = 1 / (1 + e^(-x))
f'(x) = f(x) × (1 - f(x))
```
- 범위: [0, 1]
- 용도: 이진 분류 출력층

## 4. 손실 함수

### MSE (Mean Squared Error)
```python
L = (1/n) × Σ(y_pred - y_true)²
∂L/∂y_pred = (2/n) × (y_pred - y_true)
```
- 용도: 회귀 문제
- 특징: 큰 오차에 민감

### Cross-Entropy
```python
L = -Σ(y_true × log(y_pred))
∂L/∂y_pred = -y_true/y_pred
```
- 용도: 분류 문제
- 특징: 확률 분포 비교

## 5. Backpropagation

### Chain Rule
```
∂L/∂w[l] = ∂L/∂a[l] × ∂a[l]/∂z[l] × ∂z[l]/∂w[l]
```

### 구현 패턴
```python
def backward(self):
    # 1. 출력층 gradient
    self.grad = 1.0
    
    # 2. 역순으로 전파
    for layer in reversed(self.layers):
        layer.backward()
```

## 6. 최적화 (Optimization)

### SGD (Stochastic Gradient Descent)
```python
w = w - learning_rate × ∂L/∂w
```

### Momentum
```python
v = β × v - learning_rate × ∂L/∂w
w = w + v
```

### Adam
```python
m = β1 × m + (1-β1) × grad      # 1차 모멘트
v = β2 × v + (1-β2) × grad²     # 2차 모멘트
w = w - lr × m / (√v + ε)
```

## 7. XOR 문제의 의미

### 왜 중요한가?
1. **선형 분리 불가능**: 단층으로 해결 불가
2. **은닉층 필요성**: 비선형 변환 필요
3. **신경망의 표현력**: Universal Approximation

### XOR 진리표
```
X1  X2  |  Y
--------|----
0   0   |  0
0   1   |  1
1   0   |  1
1   1   |  0
```

### 해결 방법
```python
# 최소 구조: 2-2-1
# 안정적: 2-4-1
model = MLP(2, [4, 1])
```

## 8. 학습 과정

### 1. 초기화
- Xavier: `w ~ N(0, 2/(nin + nout))`
- He: `w ~ N(0, 2/nin)`

### 2. 학습 루프
```python
for epoch in range(epochs):
    # Forward
    pred = model(x)
    loss = loss_fn(pred, y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Update
    optimizer.step()
```

### 3. 수렴 확인
- Loss 감소
- Gradient norm 감소
- Validation 성능

## 9. 일반적인 문제와 해결

### Gradient Vanishing
- 원인: 깊은 네트워크, sigmoid/tanh
- 해결: ReLU, BatchNorm, ResNet

### Gradient Exploding
- 원인: 큰 가중치, 깊은 네트워크
- 해결: Gradient clipping, 작은 초기화

### Overfitting
- 원인: 모델 복잡도 > 데이터
- 해결: Dropout, L2 정규화, 데이터 증강

## 10. 실습 체크리스트

### 구현 순서
1. ✅ Neuron: 가중합 + 활성화
2. ✅ Layer: 뉴런 집합
3. ✅ MLP: 층 연결
4. ✅ Loss: MSE
5. ✅ Optimizer: SGD
6. ✅ Training Loop

### 테스트 순서
1. 단일 뉴런 → AND 게이트
2. 2층 네트워크 → XOR
3. 3층 네트워크 → 원형 데이터

## 11. 코드 스니펫

### 뉴런 구현
```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) 
                  for _ in range(nin)]
        self.b = Value(0)
    
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.tanh()
```

### 학습 루프
```python
for epoch in range(1000):
    # Batch 처리
    for x_batch, y_batch in dataloader:
        pred = model(x_batch)
        loss = mse_loss(pred, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 12. 다음 단계

### Day 2 준비
- NumPy 배열 연산
- 행렬곱 이해
- Broadcasting

### 확장 주제
- Batch 처리
- 정규화 (BatchNorm, LayerNorm)
- Dropout
- Skip Connection

---

**"뉴런 하나하나가 모여 지능을 만듭니다"**