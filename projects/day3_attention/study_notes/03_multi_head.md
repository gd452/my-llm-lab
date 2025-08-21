# 🎭 Multi-Head Attention: 다양한 관점에서 보기

## 🎯 학습 목표
- Multi-Head Attention의 필요성 이해
- 병렬 Attention head 구현
- Concatenation과 Linear projection
- 실제 Transformer에서의 활용

## 1. 왜 Multi-Head인가?

### Single Head의 한계

```
문장: "The bank is by the river bank"

Single Head는 하나의 관점만 학습:
- 문법적 관계 OR
- 의미적 관계 OR  
- 위치적 관계

하지만 모두 동시에 필요!
```

### Multi-Head의 해결책

```
Head 1: 문법 (주어-동사)
Head 2: 의미 (bank의 중의성)
Head 3: 위치 (공간 관계)
Head 4: 문맥 (전체 의미)

→ 각 head가 다른 패턴을 학습
```

## 2. Multi-Head Attention 수식

### 전체 공식

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 파라미터

- **h**: head 개수 (보통 8, 16)
- **d_model**: 모델 차원 (예: 512)
- **d_k = d_v = d_model / h**: 각 head의 차원 (예: 64)

## 3. 단계별 구현

### Step 1: 입력 준비

```python
import numpy as np

# 설정
d_model = 512  # 전체 모델 차원
num_heads = 8  # head 개수
d_k = d_v = d_model // num_heads  # 64

# 입력
seq_len = 10
X = np.random.randn(seq_len, d_model)
```

### Step 2: Multi-Head 투영

```python
# 각 head별 가중치 행렬
W_Q = np.random.randn(num_heads, d_model, d_k) * 0.1
W_K = np.random.randn(num_heads, d_model, d_k) * 0.1
W_V = np.random.randn(num_heads, d_model, d_v) * 0.1

# 모든 head의 Q, K, V 계산
Q_heads = []
K_heads = []
V_heads = []

for i in range(num_heads):
    Q_heads.append(X @ W_Q[i])  # (seq_len, d_k)
    K_heads.append(X @ W_K[i])  
    V_heads.append(X @ W_V[i])
```

### Step 3: 병렬 Attention

```python
def scaled_dot_product_attention(Q, K, V):
    """단일 attention head 계산"""
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights

# 각 head별 attention 계산
head_outputs = []
attention_weights_all = []

for i in range(num_heads):
    output, weights = scaled_dot_product_attention(
        Q_heads[i], K_heads[i], V_heads[i]
    )
    head_outputs.append(output)
    attention_weights_all.append(weights)
```

### Step 4: Concatenation

```python
# 모든 head 출력을 연결
# 각 head: (seq_len, d_v)
# 연결 후: (seq_len, num_heads * d_v)

multi_head_output = np.concatenate(head_outputs, axis=-1)
# Shape: (seq_len, d_model)
```

### Step 5: 최종 Linear Projection

```python
# 출력 투영 행렬
W_O = np.random.randn(d_model, d_model) * 0.1

# 최종 출력
output = multi_head_output @ W_O
# Shape: (seq_len, d_model)
```

## 4. 효율적인 구현

### 재구성을 통한 병렬화

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 통합된 가중치 행렬 (더 효율적)
        self.W_Q = np.random.randn(d_model, d_model) * 0.1
        self.W_K = np.random.randn(d_model, d_model) * 0.1
        self.W_V = np.random.randn(d_model, d_model) * 0.1
        self.W_O = np.random.randn(d_model, d_model) * 0.1
    
    def forward(self, X, mask=None):
        batch_size = 1 if X.ndim == 2 else X.shape[0]
        seq_len = X.shape[-2]
        
        # 1. Linear projections in batch
        Q = X @ self.W_Q  # (seq_len, d_model)
        K = X @ self.W_K
        V = X @ self.W_V
        
        # 2. Reshape for multi-head
        Q = self.split_heads(Q)  # (num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 3. Scaled dot-product attention
        attention_output, weights = self.scaled_attention(Q, K, V, mask)
        
        # 4. Concatenate heads
        concat_output = self.combine_heads(attention_output)
        
        # 5. Final linear projection
        output = concat_output @ self.W_O
        
        return output, weights
    
    def split_heads(self, X):
        """(seq_len, d_model) → (num_heads, seq_len, d_k)"""
        seq_len = X.shape[0]
        X = X.reshape(seq_len, self.num_heads, self.d_k)
        return X.transpose(1, 0, 2)
    
    def combine_heads(self, X):
        """(num_heads, seq_len, d_k) → (seq_len, d_model)"""
        X = X.transpose(1, 0, 2)
        seq_len = X.shape[0]
        return X.reshape(seq_len, self.d_model)
    
    def scaled_attention(self, Q, K, V, mask=None):
        """Multi-head attention 계산"""
        # (num_heads, seq_len, d_k) @ (num_heads, d_k, seq_len)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores += mask * -1e9
        
        weights = self.softmax(scores)
        output = np.matmul(weights, V)
        
        return output, weights
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

## 5. 각 Head가 학습하는 패턴

### Head 특화 예시

```python
def visualize_head_patterns():
    """각 head가 다른 패턴을 학습하는 예시"""
    
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    
    # Head 1: 인접 단어 (local)
    head1 = np.array([
        [0.7, 0.3, 0.0, 0.0, 0.0, 0.0],  # The → The, cat
        [0.3, 0.4, 0.3, 0.0, 0.0, 0.0],  # cat → The, cat, sat
        [0.0, 0.3, 0.4, 0.3, 0.0, 0.0],  # sat → cat, sat, on
        # ...
    ])
    
    # Head 2: 문법 관계 (주어-동사)
    head2 = np.array([
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0],  # The → The, the
        [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],  # cat → cat, sat
        [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],  # sat → cat, sat
        # ...
    ])
    
    # Head 3: 위치 정보 (시작/끝)
    head3 = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 시작 토큰
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.0],  # 중간
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 끝 토큰
        # ...
    ])
    
    return head1, head2, head3
```

## 6. Head 개수 선택

### Trade-offs

```python
# 적은 head (예: 2-4개)
# + 각 head가 더 많은 차원 (d_k가 큼)
# + 메모리 효율적
# - 다양성 부족

# 많은 head (예: 16-32개)
# + 다양한 패턴 학습
# + 더 풍부한 표현력
# - 각 head의 용량 감소 (d_k가 작음)
# - 계산량 증가

# 일반적인 선택
models = {
    "BERT-base": {"d_model": 768, "heads": 12},  # d_k = 64
    "GPT-2": {"d_model": 768, "heads": 12},      # d_k = 64
    "GPT-3": {"d_model": 12288, "heads": 96},    # d_k = 128
}
```

## 7. 실전 활용

### Cross-Attention (Encoder-Decoder)

```python
def cross_attention(query_seq, key_value_seq, num_heads=8):
    """
    다른 시퀀스를 참조하는 attention
    
    예: 번역에서 소스 언어 참조
    Query: 타겟 언어 (디코더)
    Key, Value: 소스 언어 (인코더)
    """
    mha = MultiHeadAttention(d_model=512, num_heads=num_heads)
    
    # Query는 타겟, Key/Value는 소스에서
    Q = query_seq @ mha.W_Q
    K = key_value_seq @ mha.W_K
    V = key_value_seq @ mha.W_V
    
    # 이후 동일한 과정
    # ...
```

### Masked Multi-Head Attention

```python
def create_causal_mask(seq_len):
    """GPT 스타일 causal mask 생성"""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask

def masked_multi_head_attention(X, num_heads=8):
    """미래 정보를 볼 수 없는 attention"""
    seq_len = X.shape[0]
    mask = create_causal_mask(seq_len)
    
    mha = MultiHeadAttention(d_model=512, num_heads=num_heads)
    output, weights = mha.forward(X, mask=mask)
    
    return output, weights
```

## 8. 디버깅과 분석

### Head별 기여도 분석

```python
def analyze_head_importance(model, X):
    """각 head의 중요도 측정"""
    
    # 각 head를 하나씩 제거하면서 성능 측정
    head_importance = []
    
    for head_idx in range(model.num_heads):
        # head_idx를 0으로 만들어 제거
        masked_output = model.forward_with_head_mask(
            X, masked_heads=[head_idx]
        )
        
        # 원본과의 차이 측정
        original_output = model.forward(X)
        difference = np.mean((original_output - masked_output) ** 2)
        
        head_importance.append(difference)
    
    return head_importance
```

### Attention 패턴 시각화

```python
def visualize_multi_head_attention(attention_weights, tokens):
    """Multi-head attention 가중치 시각화"""
    
    num_heads = attention_weights.shape[0]
    seq_len = attention_weights.shape[1]
    
    # 각 head별 heatmap
    for head in range(num_heads):
        print(f"\n=== Head {head + 1} ===")
        weights = attention_weights[head]
        
        # 텍스트 기반 시각화
        print("     ", end="")
        for token in tokens:
            print(f"{token[:3]:>6}", end="")
        print()
        
        for i, token in enumerate(tokens):
            print(f"{token[:3]:>6}", end="")
            for j in range(seq_len):
                intensity = int(weights[i, j] * 10)
                print(f"  {'█' * intensity:>4}", end="")
            print()
```

## 💡 핵심 통찰

1. **다양성이 핵심**
   - 각 head가 다른 관계를 포착
   - 앙상블 효과

2. **계산 효율성**
   - 병렬 처리 가능
   - 재구성으로 배치 연산

3. **표현력 증가**
   - 단일 head보다 풍부한 표현
   - 복잡한 패턴 학습 가능

## 🔍 이해도 체크

1. d_model=512, num_heads=8일 때 d_k는?
2. Multi-head의 장점 3가지는?
3. Concatenation 후 왜 Linear projection이 필요한가?
4. Cross-attention과 Self-attention의 차이는?

## 📝 연습 문제

1. num_heads를 1, 2, 4, 8로 바꿔가며 성능 비교
2. 각 head가 학습한 패턴 시각화
3. Head pruning 구현 (불필요한 head 제거)

## 다음 단계

위치 정보를 어떻게 주입할까요?
→ [04_positional_encoding.md](04_positional_encoding.md)