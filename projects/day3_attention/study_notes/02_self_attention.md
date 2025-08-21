# 🔧 Self-Attention의 수학적 구현

## 🎯 학습 목표
- Self-Attention의 수식 이해
- Query, Key, Value 행렬 계산
- Scaled Dot-Product Attention 구현
- 실제 코드로 작성하기

## 1. Self-Attention 수식

### 핵심 공식

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

각 구성 요소:
- **Q** (Query): 내가 찾고 있는 것
- **K** (Key): 각 위치가 제공할 수 있는 정보
- **V** (Value): 실제 전달될 정보
- **d_k**: Key 벡터의 차원 (스케일링 팩터)

## 2. 단계별 계산

### Step 0: 입력 임베딩

```python
# 입력: "The cat sat"
# 각 단어를 벡터로 변환 (예: d_model=4)

X = [[0.1, 0.2, 0.3, 0.4],  # "The"
     [0.5, 0.6, 0.7, 0.8],  # "cat"  
     [0.9, 1.0, 1.1, 1.2]]  # "sat"

# Shape: (seq_len=3, d_model=4)
```

### Step 1: Q, K, V 생성

```python
# 학습 가능한 가중치 행렬
W_Q = [[...], [...], ...]  # (d_model, d_k)
W_K = [[...], [...], ...]  # (d_model, d_k)
W_V = [[...], [...], ...]  # (d_model, d_v)

# 선형 변환
Q = X @ W_Q  # (seq_len, d_k)
K = X @ W_K  # (seq_len, d_k)
V = X @ W_V  # (seq_len, d_v)
```

### Step 2: Attention Score 계산

```python
# Query와 Key의 내적
scores = Q @ K.T  # (seq_len, seq_len)

# 예시 결과:
#        The  cat  sat
# The  [[2.0, 1.5, 0.8],
# cat   [1.5, 3.0, 2.1],
# sat   [0.8, 2.1, 2.5]]
```

### Step 3: Scaling

```python
# √d_k로 나누기 (안정성을 위해)
d_k = Q.shape[-1]  # Key 차원
scores = scores / sqrt(d_k)

# 왜 스케일링?
# d_k가 크면 내적 값이 커져서 softmax가 saturate됨
# 예: [100, 1, 1] → softmax → [0.99999, 0.00001, 0.00001]
#     [10, 1, 1] → softmax → [0.99, 0.005, 0.005]
```

### Step 4: Softmax

```python
# 각 행별로 softmax 적용
attention_weights = softmax(scores, axis=-1)

# 결과:
#        The   cat   sat
# The  [[0.50, 0.30, 0.20],  # The는 자신에 50% 주목
# cat   [0.20, 0.60, 0.20],  # cat은 자신에 60% 주목
# sat   [0.15, 0.35, 0.50]]  # sat은 자신에 50% 주목

# 각 행의 합 = 1.0
```

### Step 5: Value 가중합

```python
# Attention 가중치로 Value 가중합
output = attention_weights @ V  # (seq_len, d_v)

# 각 위치의 출력 = 모든 Value의 가중 평균
# output[0] = 0.50*V[0] + 0.30*V[1] + 0.20*V[2]
```

## 3. 구체적인 예제

### 완전한 Self-Attention 구현

```python
import numpy as np

def self_attention(X, W_Q, W_K, W_V):
    """
    Self-Attention 계산
    
    Args:
        X: 입력 (seq_len, d_model)
        W_Q, W_K, W_V: 가중치 행렬
    
    Returns:
        출력 (seq_len, d_v)
    """
    # 1. Q, K, V 계산
    Q = X @ W_Q
    K = X @ W_K  
    V = X @ W_V
    
    # 2. Attention scores
    scores = Q @ K.T
    
    # 3. Scaling
    d_k = K.shape[-1]
    scores = scores / np.sqrt(d_k)
    
    # 4. Softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 5. Value 가중합
    output = attention_weights @ V
    
    return output, attention_weights
```

### 실행 예제

```python
# 설정
seq_len = 4
d_model = 8
d_k = d_v = 4

# 입력 생성
X = np.random.randn(seq_len, d_model)

# 가중치 초기화
W_Q = np.random.randn(d_model, d_k) * 0.1
W_K = np.random.randn(d_model, d_k) * 0.1
W_V = np.random.randn(d_model, d_v) * 0.1

# Self-Attention 실행
output, attention_weights = self_attention(X, W_Q, W_K, W_V)

print(f"입력 shape: {X.shape}")
print(f"출력 shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
print(f"\nAttention weights:\n{attention_weights}")
```

## 4. Masked Self-Attention (Causal)

### GPT 스타일: 미래 정보 차단

```python
def causal_self_attention(X, W_Q, W_K, W_V):
    """
    Causal Self-Attention (미래를 볼 수 없음)
    """
    seq_len = X.shape[0]
    
    # 기본 attention 계산
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    
    scores = Q @ K.T / np.sqrt(K.shape[-1])
    
    # Causal mask 적용
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    scores = scores - mask * 1e10  # 미래 위치에 큰 음수
    
    # Softmax
    attention_weights = softmax(scores, axis=-1)
    
    # Output
    output = attention_weights @ V
    
    return output, attention_weights

# Causal mask 예시:
# [[0, -∞, -∞, -∞],  # 위치 0은 자신만 볼 수 있음
#  [0,  0, -∞, -∞],  # 위치 1은 0,1을 볼 수 있음
#  [0,  0,  0, -∞],  # 위치 2는 0,1,2를 볼 수 있음
#  [0,  0,  0,  0]]  # 위치 3은 모두 볼 수 있음
```

## 5. 효율적인 구현 팁

### 배치 처리

```python
def batch_self_attention(X, W_Q, W_K, W_V):
    """
    배치 단위 Self-Attention
    
    Args:
        X: (batch_size, seq_len, d_model)
    """
    # Q, K, V 계산 (broadcasting 활용)
    Q = X @ W_Q  # (batch, seq_len, d_k)
    K = X @ W_K
    V = X @ W_V
    
    # 배치 행렬곱
    scores = np.matmul(Q, K.transpose(0, 2, 1))
    scores = scores / np.sqrt(K.shape[-1])
    
    # Softmax (마지막 차원)
    attention_weights = softmax(scores, axis=-1)
    
    # Output
    output = np.matmul(attention_weights, V)
    
    return output
```

### 메모리 효율적 구현

```python
def memory_efficient_attention(Q, K, V, chunk_size=32):
    """
    청크 단위로 계산하여 메모리 절약
    """
    seq_len = Q.shape[0]
    outputs = []
    
    for i in range(0, seq_len, chunk_size):
        q_chunk = Q[i:i+chunk_size]
        
        # 청크별 attention
        scores = q_chunk @ K.T / np.sqrt(K.shape[-1])
        weights = softmax(scores, axis=-1)
        output_chunk = weights @ V
        
        outputs.append(output_chunk)
    
    return np.concatenate(outputs, axis=0)
```

## 6. Attention Pattern 분석

### 다양한 Attention 패턴

```python
# 1. 자기 자신에 집중
attention_diagonal = np.eye(seq_len)

# 2. 균등 분포
attention_uniform = np.ones((seq_len, seq_len)) / seq_len

# 3. 첫 토큰에 집중 (CLS token style)
attention_first = np.zeros((seq_len, seq_len))
attention_first[:, 0] = 1.0

# 4. 인접 토큰 집중
attention_local = np.zeros((seq_len, seq_len))
for i in range(seq_len):
    for j in range(max(0, i-1), min(seq_len, i+2)):
        attention_local[i, j] = 1/3
```

## 7. 실전 디버깅

### Attention 가중치 검증

```python
def validate_attention(attention_weights):
    """Attention 가중치 검증"""
    
    # 1. 각 행의 합이 1인지
    row_sums = attention_weights.sum(axis=-1)
    assert np.allclose(row_sums, 1.0), "행 합이 1이 아님"
    
    # 2. 모든 값이 0~1 사이인지
    assert (attention_weights >= 0).all(), "음수 가중치"
    assert (attention_weights <= 1).all(), "1보다 큰 가중치"
    
    # 3. NaN이나 Inf가 없는지
    assert not np.isnan(attention_weights).any(), "NaN 발견"
    assert not np.isinf(attention_weights).any(), "Inf 발견"
    
    print("✅ Attention 가중치 정상")
```

## 💡 핵심 포인트

1. **Q, K, V는 학습되는 변환**
   - 같은 입력에서 다른 역할로 변환

2. **Scaling은 필수**
   - 큰 차원에서 gradient vanishing 방지

3. **Softmax로 확률 분포화**
   - 가중치의 해석 가능성

4. **행렬 연산으로 효율성**
   - 모든 위치를 동시에 계산

## 🔍 이해도 체크

1. Q @ K.T의 결과 shape은?
2. Scaling factor √d_k가 필요한 이유는?
3. Causal mask는 언제 필요한가?
4. Attention weights의 특성 3가지는?

## 📝 연습 문제

1. d_model=512, d_k=64일 때 메모리 사용량 계산
2. seq_len=1000일 때 attention 행렬 크기는?
3. Relative position encoding 구현해보기

## 다음 단계

Single head를 Multi-head로 확장해봅시다!
→ [03_multi_head.md](03_multi_head.md)