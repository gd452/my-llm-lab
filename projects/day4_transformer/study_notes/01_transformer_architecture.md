# Transformer Architecture Overview

## Introduction
Transformer는 2017년 "Attention is All You Need" 논문에서 소개된 혁신적인 아키텍처입니다.
RNN/LSTM 없이 오직 Attention 메커니즘만으로 구성되어 병렬 처리가 가능합니다.

## 전체 구조

```
[Input] → [Encoder Stack] → [Encoder Output]
                                    ↓
[Output] → [Decoder Stack] → [Output Probabilities]
```

## Encoder-Decoder Architecture

### Encoder Stack
- N개의 identical layer로 구성 (보통 N=6)
- 각 layer는 2개의 sub-layer를 가짐:
  1. Multi-Head Self-Attention
  2. Position-wise Feed-Forward Network
- 각 sub-layer 주변에 residual connection과 layer normalization 적용

### Decoder Stack  
- N개의 identical layer로 구성 (보통 N=6)
- 각 layer는 3개의 sub-layer를 가짐:
  1. Masked Multi-Head Self-Attention
  2. Multi-Head Cross-Attention (Encoder-Decoder Attention)
  3. Position-wise Feed-Forward Network
- 마찬가지로 residual connection과 layer normalization 적용

## Key Components

### 1. Embedding Layer
```python
# Token을 dense vector로 변환
embedding = token_embedding + positional_encoding
```

### 2. Positional Encoding
- Transformer는 순서 정보가 없음
- Sinusoidal positional encoding 사용
- 각 위치와 차원에 고유한 값 부여

### 3. Multi-Head Attention (Day 3에서 학습)
- Query, Key, Value 메커니즘
- 여러 representation subspace에서 동시에 attention 수행

### 4. Layer Normalization
- 각 layer의 출력을 정규화
- Training 안정성 향상
```python
LayerNorm(x) = γ * (x - μ) / σ + β
```

### 5. Residual Connection
- Gradient vanishing 문제 해결
- 깊은 네트워크 학습 가능
```python
output = LayerNorm(x + Sublayer(x))
```

### 6. Position-wise Feed-Forward
- 각 position에 독립적으로 적용
- 2개의 linear transformation + ReLU
```python
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

## Information Flow

### Encoder Flow
1. Input tokens → Embedding
2. Add positional encoding
3. Pass through N encoder layers:
   - Self-attention (모든 position이 서로를 볼 수 있음)
   - Feed-forward network
4. Output: Context representations

### Decoder Flow
1. Output tokens → Embedding  
2. Add positional encoding
3. Pass through N decoder layers:
   - Masked self-attention (미래 tokens 못 봄)
   - Cross-attention with encoder output
   - Feed-forward network
4. Linear projection → Softmax → Output probabilities

## Why Transformer?

### Advantages
1. **병렬 처리**: RNN과 달리 모든 position 동시 처리
2. **Long-range dependency**: Attention으로 직접 연결
3. **계산 효율성**: 행렬 연산으로 GPU 최적화
4. **Interpretability**: Attention weights 시각화 가능

### Key Innovations
1. **Self-Attention**: 입력 시퀀스 내 관계 학습
2. **Multi-Head**: 다양한 representation 학습
3. **Positional Encoding**: 순서 정보 주입
4. **Layer Norm + Residual**: 깊은 네트워크 학습

## Hyperparameters

| Parameter | Typical Value | Description |
|-----------|--------------|-------------|
| d_model | 512 | Model dimension |
| n_heads | 8 | Number of attention heads |
| n_layers | 6 | Number of encoder/decoder layers |
| d_ff | 2048 | Feed-forward dimension |
| dropout | 0.1 | Dropout rate |
| max_len | 5000 | Maximum sequence length |

## 실습 포인트
1. 각 component가 어떻게 연결되는지 이해
2. Residual connection의 중요성 체감
3. Layer normalization의 효과 확인
4. Encoder-Decoder interaction 구현