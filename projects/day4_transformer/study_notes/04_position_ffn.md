# Position-wise Feed-Forward Network

## Introduction
Position-wise FFN은 Transformer의 핵심 구성 요소 중 하나입니다.
각 position에 독립적으로 적용되는 간단하지만 강력한 네트워크입니다.

## 기본 개념

### 정의
- "Position-wise": 각 시퀀스 position에 독립적으로 적용
- 2개의 Linear transformation + 활성화 함수
- Attention이 관계를 학습한다면, FFN은 특징을 변환

### 수식
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

또는 더 일반적으로:
FFN(x) = σ(xW₁ + b₁)W₂ + b₂

여기서:
- W₁ ∈ R^(d_model × d_ff)
- W₂ ∈ R^(d_ff × d_model)
- d_ff는 보통 4 * d_model (예: 2048 if d_model=512)
- σ는 활성화 함수 (ReLU, GELU 등)
```

## PyTorch 구현

### 기본 구현 (ReLU)
```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = self.w_1(x)           # [batch_size, seq_len, d_ff]
        x = self.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)           # [batch_size, seq_len, d_model]
        return x
```

### GELU 활성화 함수 버전
```python
class FeedForwardGELU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        return self.w_2(self.dropout(self.gelu(self.w_1(x))))
```

### GLU Variants (Gated Linear Units)
```python
class GatedFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # Gate와 value를 위한 projection
        self.w_gate = nn.Linear(d_model, d_ff)
        self.w_value = nn.Linear(d_model, d_ff)
        self.w_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # Gating mechanism
        gate = self.activation(self.w_gate(x))
        value = self.w_value(x)
        gated = gate * value  # Element-wise multiplication
        output = self.w_out(self.dropout(gated))
        return output
```

## 왜 Position-wise인가?

### 1. 계산 효율성
```python
# Position-wise: 각 position 독립 처리
for pos in range(seq_len):
    output[pos] = ffn(input[pos])

# 실제로는 병렬 처리
output = ffn(input)  # Broadcast over seq_len dimension
```

### 2. Parameter Sharing
- 모든 position이 같은 FFN parameters 공유
- 시퀀스 길이에 관계없이 동일한 변환 적용
- CNN의 weight sharing과 유사한 개념

### 3. Local Processing
- Attention: Global interaction
- FFN: Local feature transformation
- 두 메커니즘의 균형

## Hidden Dimension의 중요성

### 왜 d_ff = 4 * d_model?
```python
# Information bottleneck 분석
d_model = 512
d_ff = 2048  # 4x expansion

# Forward pass
x -> [512] -> expand -> [2048] -> compress -> [512]
```

**이유:**
1. **표현력 증가**: 더 큰 hidden space에서 복잡한 변환 학습
2. **Non-linearity**: 활성화 함수와 함께 비선형 변환 강화
3. **Empirical finding**: 실험적으로 4배가 최적

### 다양한 Hidden Dimension 실험
```python
def test_hidden_dims():
    d_model = 512
    ratios = [1, 2, 4, 8]  # d_ff / d_model 비율
    
    for ratio in ratios:
        d_ff = d_model * ratio
        model = build_model(d_model, d_ff)
        perplexity = evaluate(model)
        print(f"Ratio {ratio}: PPL = {perplexity:.2f}")

# 예상 결과:
# Ratio 1: PPL = 45.3
# Ratio 2: PPL = 38.7
# Ratio 4: PPL = 32.1 (최적)
# Ratio 8: PPL = 33.5 (과적합 시작)
```

## 활성화 함수 비교

### ReLU vs GELU vs SwiGLU
```python
import torch
import torch.nn.functional as F

def compare_activations(x):
    # ReLU: max(0, x)
    relu_out = F.relu(x)
    
    # GELU: x * Φ(x) where Φ is CDF of standard normal
    gelu_out = F.gelu(x)
    
    # Swish/SiLU: x * sigmoid(x)
    swish_out = x * torch.sigmoid(x)
    
    return relu_out, gelu_out, swish_out
```

### 성능 비교
| Activation | 장점 | 단점 | 사용 예 |
|------------|------|------|---------|
| ReLU | 간단, 빠름 | Dead neurons | Original Transformer |
| GELU | Smooth, 미분 가능 | 계산 비용 | BERT, GPT |
| SwiGLU | 최고 성능 | 파라미터 증가 | LLaMA, PaLM |

## Advanced FFN Architectures

### 1. Mixture of Experts (MoE)
```python
class MoEFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, n_experts=8):
        super().__init__()
        self.experts = nn.ModuleList([
            PositionwiseFeedForward(d_model, d_ff)
            for _ in range(n_experts)
        ])
        self.gate = nn.Linear(d_model, n_experts)
    
    def forward(self, x):
        # Compute gating scores
        gate_scores = F.softmax(self.gate(x), dim=-1)
        
        # Weighted sum of expert outputs
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            output += gate_scores[..., i:i+1] * expert(x)
        
        return output
```

### 2. Adaptive FFN
```python
class AdaptiveFeedForward(nn.Module):
    def __init__(self, d_model, max_d_ff=4096):
        super().__init__()
        self.d_model = d_model
        self.max_d_ff = max_d_ff
        
        # Learnable dimension selector
        self.dim_controller = nn.Linear(d_model, 1)
        self.w_1 = nn.Linear(d_model, max_d_ff)
        self.w_2 = nn.Linear(max_d_ff, d_model)
    
    def forward(self, x):
        # Dynamically determine active dimensions
        active_dims = torch.sigmoid(self.dim_controller(x.mean(dim=1)))
        active_dims = (active_dims * self.max_d_ff).int()
        
        # Apply FFN with adaptive dimensions
        hidden = F.gelu(self.w_1(x))
        # Mask inactive dimensions
        mask = torch.arange(self.max_d_ff) < active_dims.unsqueeze(-1)
        hidden = hidden * mask.float()
        output = self.w_2(hidden)
        
        return output
```

## FFN in Transformer Block

### 통합 구현
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        # Attention sub-layer
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # FFN sub-layer
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Attention with residual
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        
        return x
```

## 최적화 기법

### 1. Parameter Initialization
```python
def init_ffn_weights(module):
    if isinstance(module, nn.Linear):
        # Xavier/Glorot initialization
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

### 2. Gradient Checkpointing
```python
class CheckpointedFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
    
    def forward(self, x):
        # Memory 절약을 위한 gradient checkpointing
        return torch.utils.checkpoint.checkpoint(self.ffn, x)
```

### 3. Mixed Precision Training
```python
from torch.cuda.amp import autocast

class MixedPrecisionFFN(nn.Module):
    def forward(self, x):
        with autocast():
            # FP16으로 계산 (메모리 절약, 속도 향상)
            return self.ffn(x.half()).float()
```

## 실험: FFN의 역할 분석

### Ablation Study
```python
def ffn_ablation_study():
    configs = [
        {"use_ffn": False, "name": "No FFN"},
        {"d_ff": 512, "name": "FFN 1x"},
        {"d_ff": 1024, "name": "FFN 2x"},
        {"d_ff": 2048, "name": "FFN 4x"},
        {"d_ff": 4096, "name": "FFN 8x"},
    ]
    
    results = {}
    for config in configs:
        model = build_model(**config)
        loss, accuracy = train_and_evaluate(model)
        results[config["name"]] = {
            "loss": loss,
            "accuracy": accuracy,
            "params": count_parameters(model)
        }
    
    return results
```

### Attention vs FFN 파라미터 비율
```python
def parameter_analysis(d_model=512, n_heads=8, d_ff=2048):
    # Attention parameters
    attn_params = 4 * d_model * d_model  # Q, K, V, O projections
    
    # FFN parameters
    ffn_params = 2 * d_model * d_ff  # Two linear layers
    
    total = attn_params + ffn_params
    print(f"Attention: {attn_params:,} ({attn_params/total*100:.1f}%)")
    print(f"FFN: {ffn_params:,} ({ffn_params/total*100:.1f}%)")
    
# 출력:
# Attention: 1,048,576 (33.3%)
# FFN: 2,097,152 (66.7%)
```

## 실습 포인트
1. 다양한 activation function 성능 비교
2. Hidden dimension ratio 실험
3. FFN 제거 시 모델 성능 변화 측정
4. Position-wise vs. sequence-wise processing 비교
5. MoE FFN 구현 및 실험