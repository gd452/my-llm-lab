# 🎯 Broadcasting: NumPy의 마법

## 🎯 학습 목표
- Broadcasting 규칙 완벽 이해
- 효율적인 연산 패턴 습득
- 실전 활용 능력 배양

## 1. Broadcasting이란?

Broadcasting은 shape이 다른 배열 간의 연산을 가능하게 하는 NumPy의 강력한 기능입니다.

### 기본 예제
```python
import numpy as np

# 스칼라와 벡터
a = np.array([1, 2, 3])
b = 10
c = a + b  # [11, 12, 13]

# 벡터와 행렬
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
vector = np.array([10, 20, 30])
result = matrix + vector  # 각 행에 vector 더하기
# [[11, 22, 33],
#  [14, 25, 36]]
```

## 2. Broadcasting 규칙

### 규칙 1: 차원 맞추기
뒤에서부터 차원을 비교합니다.

```python
A: (2, 3, 4)
B:    (3, 4)  # 앞에 1 추가 → (1, 3, 4)
# 연산 가능!
```

### 규칙 2: 크기 1인 차원 확장
크기가 1인 차원은 다른 배열의 크기로 확장됩니다.

```python
A: (2, 3, 1)
B: (1, 3, 5)
# 결과: (2, 3, 5)
```

### 규칙 3: 호환 불가능한 경우
```python
A: (2, 3)
B: (4,)  # Error! 3 ≠ 4
```

## 3. Broadcasting 시각화

```python
# 예제 1: 벡터 + 스칼라
a = [1, 2, 3]         # (3,)
b = 5                 # ()
# b가 [5, 5, 5]로 확장
# 결과: [6, 7, 8]

# 예제 2: 행렬 + 벡터 (행 방향)
A = [[1, 2, 3],       # (2, 3)
     [4, 5, 6]]
b = [10, 20, 30]      # (3,)
# b가 각 행에 적용
# 결과: [[11, 22, 33],
#        [14, 25, 36]]

# 예제 3: 행렬 + 벡터 (열 방향)
A = [[1, 2, 3],       # (2, 3)
     [4, 5, 6]]
b = [[10],            # (2, 1)
     [20]]
# b가 각 열에 적용
# 결과: [[11, 12, 13],
#        [24, 25, 26]]
```

## 4. 실전 Broadcasting 패턴

### 패턴 1: 평균 빼기 (Centering)
```python
# 각 특성의 평균을 빼서 중심화
X = np.random.randn(100, 5)  # 100개 샘플, 5개 특성
mean = X.mean(axis=0)         # (5,) - 각 특성의 평균
X_centered = X - mean          # Broadcasting!
```

### 패턴 2: 정규화 (Normalization)
```python
# Min-Max 정규화
X_min = X.min(axis=0)  # (5,)
X_max = X.max(axis=0)  # (5,)
X_normalized = (X - X_min) / (X_max - X_min)  # 0~1 범위로
```

### 패턴 3: Softmax 구현
```python
def softmax(X):
    """
    안정적인 Softmax 구현
    X: (batch_size, num_classes)
    """
    # 수치 안정성을 위해 최댓값 빼기
    X_max = X.max(axis=1, keepdims=True)  # (batch_size, 1)
    X_exp = np.exp(X - X_max)             # Broadcasting!
    X_sum = X_exp.sum(axis=1, keepdims=True)  # (batch_size, 1)
    return X_exp / X_sum                   # Broadcasting!
```

### 패턴 4: Batch Normalization
```python
def batch_norm(X, gamma, beta, eps=1e-8):
    """
    Batch Normalization
    X: (batch_size, features)
    gamma, beta: (features,) - 학습 가능한 파라미터
    """
    mean = X.mean(axis=0)  # (features,)
    var = X.var(axis=0)    # (features,)
    
    # 정규화
    X_norm = (X - mean) / np.sqrt(var + eps)  # Broadcasting!
    
    # 스케일과 시프트
    out = gamma * X_norm + beta  # Broadcasting!
    return out
```

## 5. Broadcasting과 메모리

### 메모리 효율적인 코드
```python
# 나쁜 예: 명시적 복제
A = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])
b_repeated = np.tile(b, (2, 1))  # 메모리 낭비!
result = A + b_repeated

# 좋은 예: Broadcasting 활용
result = A + b  # 메모리 효율적!
```

### Broadcasting의 내부 동작
```python
# Broadcasting은 실제로 메모리를 복사하지 않음
# stride를 조정하여 가상으로 확장

a = np.array([1, 2, 3])
b = a[:, np.newaxis]  # (3, 1)로 reshape
print(b.strides)  # (8, 0) - 두 번째 차원은 stride가 0!
```

## 6. 고급 Broadcasting 기법

### 외적 (Outer Product)
```python
a = np.array([1, 2, 3])      # (3,)
b = np.array([4, 5, 6, 7])   # (4,)

# 외적 계산
outer = a[:, np.newaxis] * b  # (3, 1) * (4,) → (3, 4)
# [[4, 5, 6, 7],
#  [8, 10, 12, 14],
#  [12, 15, 18, 21]]
```

### 거리 행렬 계산
```python
def pairwise_distances(X, Y):
    """
    두 점 집합 간의 거리 행렬
    X: (n, d), Y: (m, d)
    Returns: (n, m) 거리 행렬
    """
    X2 = (X**2).sum(axis=1)[:, np.newaxis]  # (n, 1)
    Y2 = (Y**2).sum(axis=1)[np.newaxis, :]  # (1, m)
    XY = X @ Y.T                             # (n, m)
    
    # Broadcasting으로 거리 계산
    distances = np.sqrt(X2 + Y2 - 2*XY)
    return distances
```

## 7. Broadcasting 함정과 해결책

### 함정 1: 예상치 못한 Broadcasting
```python
# 의도: 각 행의 합
A = np.array([[1, 2], [3, 4]])
row_sums = A.sum(axis=1)  # [3, 7]

# 실수: shape 불일치
# A / row_sums  # Error!

# 해결: keepdims 사용
row_sums = A.sum(axis=1, keepdims=True)  # [[3], [7]]
A_normalized = A / row_sums  # OK!
```

### 함정 2: 성능 저하
```python
# 나쁜 예: 큰 배열을 작은 배열에 맞춤
large = np.random.randn(1000000, 10)
small = np.random.randn(10, 1)

# 실수: transpose로 인한 메모리 재배치
result = large.T + small  # 느림!

# 좋은 예: Broadcasting 방향 고려
result = large + small.T  # 빠름!
```

## 8. 실전 예제: 신경망 레이어

```python
class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros(output_dim)
    
    def forward(self, X):
        """
        X: (batch_size, input_dim)
        Returns: (batch_size, output_dim)
        """
        # Broadcasting으로 bias 추가
        return X @ self.W + self.b  # self.b가 각 샘플에 broadcast
    
    def backward(self, X, dZ):
        """
        역전파 계산
        dZ: (batch_size, output_dim)
        """
        batch_size = X.shape[0]
        
        # 그래디언트 계산
        self.dW = X.T @ dZ / batch_size
        self.db = dZ.sum(axis=0) / batch_size  # Broadcasting 준비
        
        # 입력에 대한 그래디언트
        dX = dZ @ self.W.T
        return dX
```

## 💡 Broadcasting 마스터 팁

1. **Shape 먼저 생각**: 연산 전 shape 확인
2. **keepdims=True 활용**: 차원 유지로 Broadcasting 용이
3. **newaxis 활용**: 차원 추가로 Broadcasting 제어
4. **성능 고려**: Broadcasting 방향이 성능에 영향

## 🔍 디버깅 도구

```python
def broadcast_shapes(a_shape, b_shape):
    """두 shape의 broadcast 결과 예측"""
    # 차원 맞추기
    ndim = max(len(a_shape), len(b_shape))
    a_shape = (1,) * (ndim - len(a_shape)) + a_shape
    b_shape = (1,) * (ndim - len(b_shape)) + b_shape
    
    # Broadcasting 규칙 적용
    result_shape = []
    for a, b in zip(a_shape, b_shape):
        if a == 1:
            result_shape.append(b)
        elif b == 1:
            result_shape.append(a)
        elif a == b:
            result_shape.append(a)
        else:
            raise ValueError(f"Cannot broadcast {a_shape} with {b_shape}")
    
    return tuple(result_shape)

# 테스트
print(broadcast_shapes((2, 3), (3,)))     # (2, 3)
print(broadcast_shapes((2, 1), (1, 3)))   # (2, 3)
```

## 📝 연습 문제

1. 이미지 배치 (32, 224, 224, 3)에서 각 채널의 평균을 빼는 코드를 작성하세요.
2. Attention Score 계산: Q(n, d) @ K(m, d).T를 Broadcasting으로 구현하세요.
3. Dropout 마스크를 Broadcasting으로 적용하는 함수를 작성하세요.

## 다음 단계

Batch 처리의 힘을 알아봅시다! → [03_batch_processing.md](03_batch_processing.md)