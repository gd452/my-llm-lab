# 📍 Positional Encoding: 위치 정보 주입하기

## 🎯 학습 목표
- Positional Encoding이 필요한 이유
- Sinusoidal encoding 이해와 구현
- Learned positional embedding
- 최신 기법들 (RoPE, ALiBi)

## 1. 왜 위치 정보가 필요한가?

### Attention의 맹점

```
문장 1: "The cat sat on the mat"
문장 2: "The mat sat on the cat"

Attention만으로는 두 문장이 동일!
- 같은 단어들
- 같은 attention 패턴 가능
- 순서 정보 없음
```

### RNN vs Transformer

```
RNN: 순차적 처리 → 위치 정보 자동 포함
     A → B → C → D

Transformer: 병렬 처리 → 위치 정보 없음
            A, B, C, D (동시에)
            
해결책: 위치 정보를 명시적으로 추가!
```

## 2. Sinusoidal Positional Encoding

### 수식

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

pos: 토큰의 위치 (0, 1, 2, ...)
i: 차원 인덱스 (0, 1, ..., d_model/2)
```

### 직관적 이해

```
각 차원이 다른 주파수의 sin/cos 파동:
- 낮은 차원: 빠른 변화 (세밀한 위치)
- 높은 차원: 느린 변화 (전체적 위치)

시계의 비유:
- 초침: 빠른 변화 (1초마다)
- 분침: 중간 변화 (60초마다)
- 시침: 느린 변화 (3600초마다)
```

### 구현

```python
import numpy as np

def positional_encoding(seq_len, d_model):
    """
    Sinusoidal Positional Encoding 생성
    
    Args:
        seq_len: 시퀀스 길이
        d_model: 모델 차원
    
    Returns:
        PE matrix (seq_len, d_model)
    """
    # 위치 인덱스
    positions = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    
    # 차원 인덱스
    dimensions = np.arange(d_model)[np.newaxis, :]  # (1, d_model)
    
    # 각 차원의 주파수
    angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / d_model)
    
    # 위치 * 주파수
    angle_rads = positions * angle_rates  # (seq_len, d_model)
    
    # sin을 짝수 인덱스에, cos을 홀수 인덱스에
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])  # 짝수
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])  # 홀수
    
    return pe

# 테스트
pe = positional_encoding(100, 512)
print(f"PE shape: {pe.shape}")
print(f"PE 범위: [{pe.min():.2f}, {pe.max():.2f}]")
```

## 3. 시각화와 특성

### Positional Encoding 패턴 시각화

```python
def visualize_positional_encoding(seq_len=100, d_model=128):
    """PE 패턴 시각화"""
    pe = positional_encoding(seq_len, d_model)
    
    # 히트맵 스타일 출력 (간단한 ASCII)
    print("Positional Encoding Heatmap")
    print("Position →")
    print("Dimension ↓")
    
    # 샘플링하여 표시
    for dim in range(0, min(10, d_model), 2):
        print(f"Dim {dim:3d}: ", end="")
        for pos in range(0, min(50, seq_len), 5):
            val = pe[pos, dim]
            if val > 0.5:
                print("█", end="")
            elif val > 0:
                print("▓", end="")
            elif val > -0.5:
                print("░", end="")
            else:
                print(" ", end="")
        print()
    
    # 주파수 분석
    print(f"\n주파수 분석:")
    for dim in [0, d_model//4, d_model//2, d_model-1]:
        wavelength = 2 * np.pi / (1 / 10000**(2*dim/d_model))
        print(f"  Dim {dim}: 파장 ≈ {wavelength:.1f} positions")

visualize_positional_encoding()
```

### 특성 분석

```python
def analyze_pe_properties(pe):
    """Positional Encoding의 특성 분석"""
    
    # 1. 직교성: 다른 위치는 구별 가능
    dot_products = pe @ pe.T
    print("위치 간 유사도 (대각선 제외):")
    np.fill_diagonal(dot_products, 0)
    print(f"  평균: {np.mean(np.abs(dot_products)):.4f}")
    print(f"  최대: {np.max(np.abs(dot_products)):.4f}")
    
    # 2. 상대 위치 표현
    def relative_position_score(pe, pos1, pos2):
        """두 위치 간의 관계가 일정한지 확인"""
        offset = pos2 - pos1
        scores = []
        for i in range(len(pe) - offset):
            sim = np.dot(pe[i], pe[i + offset])
            scores.append(sim)
        return np.mean(scores), np.std(scores)
    
    print("\n상대 위치 일관성:")
    for offset in [1, 5, 10]:
        mean, std = relative_position_score(pe, 0, offset)
        print(f"  Offset {offset}: {mean:.3f} ± {std:.3f}")
```

## 4. 입력에 추가하기

### 임베딩과 결합

```python
def add_positional_encoding(embeddings, max_len=5000):
    """
    입력 임베딩에 위치 인코딩 추가
    
    Args:
        embeddings: (batch_size, seq_len, d_model) or (seq_len, d_model)
        max_len: 최대 시퀀스 길이
    
    Returns:
        위치 인코딩이 추가된 임베딩
    """
    if embeddings.ndim == 2:
        seq_len, d_model = embeddings.shape
        pe = positional_encoding(seq_len, d_model)
        return embeddings + pe
    
    elif embeddings.ndim == 3:
        batch_size, seq_len, d_model = embeddings.shape
        pe = positional_encoding(seq_len, d_model)
        # Broadcasting으로 배치 차원에 적용
        return embeddings + pe[np.newaxis, :, :]

# 예제
embeddings = np.random.randn(10, 512)  # 10 tokens, 512 dims
embeddings_with_pe = add_positional_encoding(embeddings)

print("원본 임베딩 평균:", np.mean(np.abs(embeddings)))
print("PE 추가 후 평균:", np.mean(np.abs(embeddings_with_pe)))
```

## 5. Learned Positional Embeddings

### 학습 가능한 위치 임베딩

```python
class LearnedPositionalEmbedding:
    """학습 가능한 위치 임베딩"""
    
    def __init__(self, max_len, d_model):
        # 각 위치마다 학습 가능한 벡터
        self.pe = np.random.randn(max_len, d_model) * 0.1
    
    def forward(self, seq_len):
        return self.pe[:seq_len]
    
    def backward(self, grad):
        """그래디언트로 업데이트"""
        self.pe -= 0.01 * grad  # SGD update

# 비교: Sinusoidal vs Learned
print("Sinusoidal PE:")
print("  - 장점: 학습 불필요, 임의 길이 처리")
print("  - 단점: 고정된 패턴")
print("\nLearned PE:")
print("  - 장점: 태스크에 최적화")
print("  - 단점: 최대 길이 제한, 학습 필요")
```

## 6. 최신 기법들

### Relative Positional Encoding

```python
def relative_positional_encoding(seq_len, max_relative_dist=128):
    """
    상대 위치 인코딩
    절대 위치가 아닌 토큰 간 거리를 인코딩
    """
    # 상대 거리 행렬
    positions = np.arange(seq_len)
    relative_dist = positions[:, np.newaxis] - positions[np.newaxis, :]
    
    # 최대 거리로 클리핑
    relative_dist = np.clip(relative_dist, 
                           -max_relative_dist, 
                           max_relative_dist)
    
    # 상대 거리를 임베딩 인덱스로 변환
    relative_idx = relative_dist + max_relative_dist
    
    return relative_idx

# 예제
rel_pos = relative_positional_encoding(10, max_relative_dist=5)
print("상대 위치 행렬 (10x10):")
print(rel_pos)
```

### RoPE (Rotary Positional Encoding)

```python
def rotary_positional_encoding(q, k, seq_len, d_model):
    """
    RoPE: 회전 행렬을 사용한 위치 인코딩
    LLaMA, GPT-NeoX에서 사용
    """
    # 위치별 회전 각도
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * 
                     -(np.log(10000.0) / d_model))
    
    # 회전 행렬 생성
    angles = position * div_term
    
    # Query와 Key에 회전 적용
    q_rot = apply_rotation(q, angles)
    k_rot = apply_rotation(k, angles)
    
    return q_rot, k_rot

def apply_rotation(x, angles):
    """회전 변환 적용"""
    cos = np.cos(angles)
    sin = np.sin(angles)
    
    # 2D 회전
    x_rot = np.zeros_like(x)
    x_rot[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
    x_rot[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
    
    return x_rot
```

### ALiBi (Attention with Linear Biases)

```python
def alibi_bias(seq_len, num_heads):
    """
    ALiBi: Attention 점수에 선형 bias 추가
    위치 임베딩 대신 사용
    """
    # 각 head마다 다른 slope
    slopes = 2 ** (-8 * np.arange(num_heads) / num_heads)
    
    # 거리 행렬
    positions = np.arange(seq_len)
    distance = positions[:, np.newaxis] - positions[np.newaxis, :]
    
    # 각 head의 bias
    biases = []
    for slope in slopes:
        bias = slope * distance
        biases.append(bias)
    
    return np.array(biases)

# 사용 예
alibi = alibi_bias(100, 8)
print(f"ALiBi shape: {alibi.shape}")  # (8, 100, 100)
print(f"Head 0 bias 샘플:\n{alibi[0, :5, :5]}")
```

## 7. 실전 적용

### Transformer with Positional Encoding

```python
class TransformerWithPE:
    """위치 인코딩이 포함된 Transformer 블록"""
    
    def __init__(self, d_model, num_heads, max_len=5000):
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Positional encoding 미리 계산
        self.pe = positional_encoding(max_len, d_model)
        
        # Multi-head attention
        self.mha = MultiHeadAttention(d_model, num_heads)
    
    def forward(self, x):
        """
        Args:
            x: 입력 임베딩 (seq_len, d_model)
        """
        seq_len = x.shape[0]
        
        # 1. 위치 인코딩 추가
        x_with_pe = x + self.pe[:seq_len]
        
        # 2. Multi-head attention
        attn_output, weights = self.mha.forward(x_with_pe)
        
        return attn_output, weights

# 테스트
model = TransformerWithPE(d_model=512, num_heads=8)
input_embeddings = np.random.randn(20, 512)
output, attention = model.forward(input_embeddings)
print(f"출력 shape: {output.shape}")
```

## 8. 위치 인코딩 비교

### 성능 비교 실험

```python
def compare_positional_methods():
    """다양한 위치 인코딩 방법 비교"""
    
    seq_len = 100
    d_model = 128
    
    methods = {
        "Sinusoidal": positional_encoding(seq_len, d_model),
        "Learned": np.random.randn(seq_len, d_model) * 0.1,
        "None": np.zeros((seq_len, d_model))
    }
    
    # 각 방법의 특성
    for name, pe in methods.items():
        print(f"\n{name} Positional Encoding:")
        
        # 인접 위치 구별 능력
        diffs = []
        for i in range(seq_len - 1):
            diff = np.linalg.norm(pe[i] - pe[i+1])
            diffs.append(diff)
        
        print(f"  인접 거리 평균: {np.mean(diffs):.4f}")
        print(f"  인접 거리 표준편차: {np.std(diffs):.4f}")
        
        # 장거리 구별 능력
        if seq_len > 50:
            long_diff = np.linalg.norm(pe[0] - pe[50])
            print(f"  장거리 (0-50) 거리: {long_diff:.4f}")

compare_positional_methods()
```

## 💡 핵심 통찰

1. **위치 정보는 필수**
   - Attention은 순서를 모름
   - 명시적 위치 신호 필요

2. **Sinusoidal의 장점**
   - 학습 불필요
   - 임의 길이 처리 가능
   - 상대 위치 표현

3. **최신 트렌드**
   - RoPE: 회전 기반 (LLaMA)
   - ALiBi: Bias 기반 (효율적)
   - 상대 위치 선호

## 🔍 이해도 체크

1. Transformer에 위치 인코딩이 필요한 이유는?
2. Sinusoidal PE에서 낮은/높은 차원의 차이는?
3. Learned PE의 장단점은?
4. RoPE와 ALiBi의 핵심 아이디어는?

## 📝 연습 문제

1. 최대 길이 1000, d_model=256인 PE 생성 및 시각화
2. 상대 위치 인코딩 구현
3. PE 있을 때와 없을 때 성능 비교

## 🎉 축하합니다!

Day 3의 모든 내용을 완료했습니다!
이제 Attention의 핵심을 이해했습니다.

다음: Day 4에서 완전한 Transformer를 구현합니다!