# Day 4: Transformer Architecture

## 프로젝트 개요
Complete Transformer 아키텍처를 구현합니다. Encoder-Decoder 구조와 함께 Layer Normalization, Residual Connection, Position-wise FFN 등 핵심 컴포넌트를 만듭니다.

## 학습 목표
- Transformer 전체 아키텍처 이해
- Encoder-Decoder 구조 구현
- Layer Normalization과 Residual Connection의 역할 이해
- Position-wise Feed-Forward Network 구현
- 전체 모델 조립 및 테스트

## 구현 요구사항

### 1. Layer Normalization
```python
class LayerNorm:
    def __init__(self, d_model, eps=1e-6)
    def forward(self, x) -> normalized_tensor
```
- 각 position별로 정규화
- Learnable scale (γ)과 shift (β) 파라미터

### 2. Position-wise Feed-Forward Network
```python
class PositionwiseFeedForward:
    def __init__(self, d_model, d_ff, dropout=0.1)
    def forward(self, x) -> output
```
- 2개의 linear transformation + ReLU
- 각 position에 독립적으로 적용

### 3. Transformer Block
```python
class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, is_decoder=False)
    def forward(self, x, encoder_output=None, self_mask=None, cross_mask=None)
```
- Multi-Head Attention (Day 3에서 구현)
- Residual Connection + Layer Norm
- Position-wise FFN
- Decoder block은 추가로 Cross-Attention 포함

### 4. Encoder/Decoder Stack
```python
class TransformerEncoder:
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout)
    def forward(self, x, mask=None)

class TransformerDecoder:
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout)
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None)
```
- N개의 identical layer stack
- 각 layer는 TransformerBlock 사용

### 5. Complete Transformer
```python
class Transformer:
    def __init__(self, n_encoder_layers=6, n_decoder_layers=6, 
                 d_model=512, n_heads=8, d_ff=2048, 
                 vocab_size=10000, max_seq_len=100, dropout=0.1)
    def forward(self, src, tgt, src_mask=None, tgt_mask=None)
```
- Embedding layer + Positional Encoding
- Encoder-Decoder stacks
- Output projection layer

## 테스트 요구사항

### 필수 테스트
1. **Layer Normalization 테스트**
   - Output mean ≈ 0, variance ≈ 1
   - Shape preservation

2. **Feed-Forward Network 테스트**
   - Dimension transformation 확인
   - ReLU activation 동작

3. **Transformer Block 테스트**
   - Encoder block (self-attention only)
   - Decoder block (self + cross attention)

4. **Full Model 테스트**
   - Forward pass with dummy data
   - Output shape 검증
   - Causal mask generation

## 실습 예제

### Sequence-to-Sequence Task
```python
# 숫자 역순 변환 (toy example)
src = [1, 2, 3, 4, 5]  # Input
tgt = [5, 4, 3, 2, 1]  # Target

model = Transformer(
    n_encoder_layers=2,
    n_decoder_layers=2,
    d_model=32,
    n_heads=4,
    d_ff=128,
    vocab_size=20
)

output = model.forward(src, tgt)
```

## 주요 개념

### Residual Connection
```python
output = LayerNorm(x + Sublayer(x))
```
- Gradient vanishing 문제 해결
- 깊은 네트워크 학습 가능

### Layer Normalization vs Batch Normalization
- LayerNorm: 각 sample의 features를 정규화
- BatchNorm: batch 차원에서 정규화
- Transformer는 LayerNorm 사용 (sequence length 가변적)

### Encoder vs Decoder
**Encoder**:
- 모든 position을 동시에 볼 수 있음
- Bidirectional self-attention

**Decoder**:
- 미래 position 못 봄 (causal mask)
- Self-attention + Cross-attention

## 평가 기준
- [ ] 모든 컴포넌트 구현 완료
- [ ] 테스트 통과
- [ ] Forward pass 정상 동작
- [ ] Gradient flow 확인
- [ ] Demo 실행 성공

## 다음 단계 (Day 5)
- GPT 스타일 Decoder-only 모델
- Text generation with sampling strategies
- Tokenization과 vocabulary 관리