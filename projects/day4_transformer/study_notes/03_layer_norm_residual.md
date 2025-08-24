# Layer Normalization과 Residual Connection

## Introduction
Layer Normalization과 Residual Connection은 Deep Transformer를 학습 가능하게 만드는 핵심 기술입니다.
이 두 기법이 없다면 6층 이상의 깊은 네트워크 학습이 거의 불가능합니다.

## Layer Normalization

### 개념
- Batch Normalization과 달리 각 샘플 내에서 정규화
- 시퀀스 길이나 배치 크기에 독립적
- RNN, Transformer 등 시퀀스 모델에 적합

### 수학적 정의
```
LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β

여기서:
- x: 입력 벡터 [d_model]
- μ: x의 평균
- σ²: x의 분산
- γ: scale parameter (learnable)
- β: shift parameter (learnable)
- ε: numerical stability를 위한 작은 값 (1e-5)
```

### PyTorch 구현
```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # Normalize
        x_norm = (x - mean) / (std + self.eps)
        
        # Scale and shift
        output = self.gamma * x_norm + self.beta
        return output
```

### Layer Norm vs Batch Norm

| Aspect | Layer Norm | Batch Norm |
|--------|-----------|------------|
| 정규화 축 | Feature dimension | Batch dimension |
| 시퀀스 모델 | 적합 | 부적합 |
| 추론 시 | 학습과 동일 | 이동 평균 필요 |
| 배치 크기 의존성 | 없음 | 있음 |

## Residual Connection

### 개념
- 입력을 출력에 직접 더하는 shortcut connection
- Gradient vanishing/exploding 문제 해결
- Deep network 학습 가능하게 함

### 수학적 정의
```
output = x + F(x)

여기서:
- x: 입력
- F(x): sub-layer 함수 (attention, FFN 등)
```

### PyTorch 구현
```python
class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        # Pre-norm: Norm → Sublayer → Dropout → Add
        return x + self.dropout(sublayer(self.norm(x)))
        
        # Post-norm 대안: Sublayer → Dropout → Add → Norm
        # return self.norm(x + self.dropout(sublayer(x)))
```

## Pre-Norm vs Post-Norm

### Pre-Norm (현대 Transformer 표준)
```python
# LayerNorm을 sublayer 전에 적용
x_norm = LayerNorm(x)
output = x + Sublayer(x_norm)
```

**장점:**
- 더 안정적인 학습
- Gradient flow 개선
- Warmup 필요성 감소

### Post-Norm (Original Transformer)
```python
# LayerNorm을 residual 후에 적용
output = LayerNorm(x + Sublayer(x))
```

**장점:**
- 이론적으로 더 직관적
- 일부 태스크에서 더 나은 성능

## Transformer Block에서의 적용

### 전체 구조
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Sub-layer 1: Multi-Head Attention
        attn_output = self.attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Sub-layer 2: Feed-Forward
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x
```

### Pre-Norm 버전
```python
class PreNormTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-Norm for attention
        x_norm = self.norm1(x)
        attn_output = self.attention(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout(attn_output)
        
        # Pre-Norm for FFN
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)
        
        return x
```

## Gradient Flow 분석

### Residual Connection의 효과
```
Backward pass:
∂L/∂x = ∂L/∂y * ∂y/∂x
      = ∂L/∂y * (1 + ∂F(x)/∂x)
      = ∂L/∂y + ∂L/∂y * ∂F(x)/∂x

직접 경로(1)가 있어 gradient가 소실되지 않음
```

### Layer Norm의 효과
- Gradient 크기를 안정화
- Internal covariate shift 감소
- 학습률에 덜 민감하게 만듦

## 실제 효과 비교

### Without Residual & Norm
```python
# 6-layer network
loss_without = [10.5, 9.8, 9.5, 9.4, 9.4, 9.4]  # Gradient vanishing
```

### With Residual & Norm
```python
# 6-layer network
loss_with = [10.5, 8.2, 6.1, 4.3, 2.8, 1.5]  # 정상 학습
```

## Advanced Techniques

### 1. ReZero
```python
# Learnable residual weight
output = x + α * F(x)  # α는 0으로 초기화된 학습 가능 파라미터
```

### 2. FixUp Initialization
- Residual branch를 특별하게 초기화
- Layer Norm 없이도 깊은 네트워크 학습 가능

### 3. Admin (Adaptive Model Initialization)
```python
# Adaptive rescaling
output = x + λ * F(x) / ||F(x)||
```

## 구현 팁

### 1. Dropout 위치
```python
# Dropout은 residual connection 전에 적용
x = x + dropout(sublayer(x))
```

### 2. 초기화
```python
# Layer Norm parameters
nn.init.ones_(self.gamma)
nn.init.zeros_(self.beta)
```

### 3. Gradient Clipping과 함께 사용
```python
# Layer Norm과 함께 사용 시 더 큰 gradient clip value 가능
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 실험: Layer Norm과 Residual의 중요성

### 실험 설정
```python
def test_importance():
    # 4가지 설정 비교
    configs = [
        {"residual": False, "norm": False},  # Both off
        {"residual": True, "norm": False},   # Only residual
        {"residual": False, "norm": True},   # Only norm
        {"residual": True, "norm": True},    # Both on
    ]
    
    for config in configs:
        model = build_transformer(**config)
        train_loss = train(model)
        print(f"Config: {config}, Final loss: {train_loss}")
```

### 예상 결과
```
Config: {'residual': False, 'norm': False}, Final loss: 8.5 (수렴 안함)
Config: {'residual': True, 'norm': False}, Final loss: 3.2
Config: {'residual': False, 'norm': True}, Final loss: 4.8
Config: {'residual': True, 'norm': True}, Final loss: 1.5 (최고 성능)
```

## 실습 포인트
1. Layer Norm 있을 때와 없을 때 gradient 크기 비교
2. Residual connection 제거 시 학습 곡선 변화 관찰
3. Pre-norm vs Post-norm 성능 비교
4. 다양한 depth에서 수렴 속도 측정