# 📚 NumPy 기초: 효율적인 배열 연산

## 🎯 학습 목표
- NumPy ndarray 이해하기
- 기본 연산과 인덱싱 마스터하기
- 메모리 레이아웃 이해하기

## 1. NumPy란?

NumPy는 Python의 과학 계산 라이브러리입니다.
C로 구현되어 있어 순수 Python보다 훨씬 빠릅니다.

### 왜 NumPy인가?
```python
# Python 리스트 (느림)
a = [1, 2, 3, 4, 5]
b = [6, 7, 8, 9, 10]
c = []
for i in range(len(a)):
    c.append(a[i] + b[i])

# NumPy 배열 (빠름)
import numpy as np
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10])
c = a + b  # 벡터화된 연산!
```

## 2. ndarray 생성

### 기본 생성
```python
# 리스트에서 생성
arr = np.array([1, 2, 3, 4])

# 2차원 배열
matrix = np.array([[1, 2], 
                    [3, 4]])

# 특수 배열
zeros = np.zeros((3, 4))      # 0으로 채워진 3x4 행렬
ones = np.ones((2, 3))        # 1로 채워진 2x3 행렬
eye = np.eye(3)               # 3x3 단위 행렬
random = np.random.randn(2, 3) # 랜덤 정규분포
```

## 3. Shape과 Dimension

### Shape 이해하기
```python
# Shape: 각 차원의 크기
arr = np.array([[1, 2, 3],
                 [4, 5, 6]])
print(arr.shape)  # (2, 3) - 2행 3열

# Reshape
reshaped = arr.reshape(3, 2)  # 3행 2열로 변경
flattened = arr.flatten()     # 1차원으로 평탄화

# 차원 추가
expanded = arr[:, :, np.newaxis]  # (2, 3, 1)
# 또는
expanded = np.expand_dims(arr, axis=2)
```

## 4. 인덱싱과 슬라이싱

### 기본 인덱싱
```python
arr = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# 단일 요소
print(arr[0, 1])  # 2

# 행 선택
print(arr[1, :])  # [4, 5, 6]

# 열 선택
print(arr[:, 0])  # [1, 4, 7]

# 부분 선택
print(arr[:2, 1:])  # [[2, 3], [5, 6]]
```

### 고급 인덱싱
```python
# Boolean 인덱싱
mask = arr > 5
print(arr[mask])  # [6, 7, 8, 9]

# Fancy 인덱싱
indices = [0, 2]
print(arr[indices])  # [[1, 2, 3], [7, 8, 9]]
```

## 5. 기본 연산

### 요소별 연산
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 산술 연산
print(a + b)  # [5, 7, 9]
print(a * b)  # [4, 10, 18]
print(a ** 2) # [1, 4, 9]

# 비교 연산
print(a > 2)  # [False, False, True]
```

### 집계 연산
```python
arr = np.array([[1, 2, 3],
                 [4, 5, 6]])

print(np.sum(arr))      # 21 (전체 합)
print(np.sum(arr, axis=0))  # [5, 7, 9] (열 합)
print(np.sum(arr, axis=1))  # [6, 15] (행 합)

print(np.mean(arr))     # 3.5
print(np.std(arr))      # 표준편차
print(np.max(arr))      # 6
print(np.argmax(arr))   # 5 (최대값 인덱스)
```

## 6. 행렬 연산

### 행렬곱
```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# 행렬곱 (dot product)
C = np.dot(A, B)
# 또는
C = A @ B  # Python 3.5+

print(C)  # [[19, 22], [43, 50]]

# 요소별 곱셈과 구분!
D = A * B  # [[5, 12], [21, 32]]
```

### 전치 (Transpose)
```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

A_T = A.T
print(A_T.shape)  # (3, 2)
```

## 7. 메모리와 성능

### View vs Copy
```python
# View (메모리 공유)
arr = np.array([1, 2, 3, 4])
view = arr[1:3]
view[0] = 100
print(arr)  # [1, 100, 3, 4] - 원본도 변경됨!

# Copy (별도 메모리)
copy = arr[1:3].copy()
copy[0] = 200
print(arr)  # [1, 100, 3, 4] - 원본 유지
```

### Contiguous 메모리
```python
# C-contiguous (행 우선)
arr_c = np.array([[1, 2, 3],
                  [4, 5, 6]], order='C')

# Fortran-contiguous (열 우선)
arr_f = np.array([[1, 2, 3],
                  [4, 5, 6]], order='F')

# 성능에 영향을 줌
print(arr_c.flags['C_CONTIGUOUS'])  # True
```

## 8. 실전 예제: 간단한 신경망 레이어

```python
def linear_forward(X, W, b):
    """
    선형 레이어의 순전파
    X: (batch_size, input_dim)
    W: (input_dim, output_dim)
    b: (output_dim,)
    """
    Z = X @ W + b  # Broadcasting 자동 적용
    return Z

def relu(Z):
    """ReLU 활성화 함수"""
    return np.maximum(0, Z)

# 사용 예
batch_size = 32
input_dim = 784
output_dim = 128

X = np.random.randn(batch_size, input_dim)
W = np.random.randn(input_dim, output_dim) * 0.01
b = np.zeros(output_dim)

Z = linear_forward(X, W, b)
A = relu(Z)
print(f"출력 shape: {A.shape}")  # (32, 128)
```

## 💡 핵심 포인트

1. **벡터화**: 루프 대신 벡터 연산 사용
2. **Broadcasting**: 차원이 다른 배열 간 연산
3. **메모리 효율**: View 활용, 불필요한 복사 피하기
4. **축(axis) 이해**: 0=행, 1=열 방향 연산

## 🔍 디버깅 팁

```python
# Shape 확인은 필수!
def debug_shapes(*arrays):
    for i, arr in enumerate(arrays):
        print(f"Array {i}: shape={arr.shape}, dtype={arr.dtype}")

# 연산 전 shape 확인
debug_shapes(X, W, b)
```

## 📝 연습 문제

1. 100x100 랜덤 행렬을 생성하고 평균과 표준편차를 계산하세요.
2. 두 벡터의 코사인 유사도를 계산하는 함수를 작성하세요.
3. Batch Normalization을 NumPy로 구현해보세요.

## 다음 단계

Broadcasting의 마법을 알아봅시다! → [02_broadcasting.md](02_broadcasting.md)