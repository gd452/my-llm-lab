# Encoder-Decoder 구조 상세 설명

## Overview
Transformer의 Encoder-Decoder 구조는 sequence-to-sequence 태스크를 위한 강력한 아키텍처입니다.
입력 시퀀스를 인코딩하고, 그 정보를 바탕으로 출력 시퀀스를 생성합니다.

## Encoder: 입력 처리의 핵심

### 구조
```python
class TransformerEncoder:
    def __init__(self, n_layers=6):
        self.layers = [EncoderLayer() for _ in range(n_layers)]
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
```

### Encoder Layer 구성
각 Encoder Layer는 2개의 sub-layer로 구성:

1. **Multi-Head Self-Attention**
   - 입력 시퀀스 내의 모든 position 간 관계 학습
   - Bidirectional: 모든 position이 서로를 볼 수 있음
   ```python
   attention_output = MultiHeadAttention(Q=x, K=x, V=x)
   ```

2. **Position-wise Feed-Forward Network**
   - 각 position에 독립적으로 적용되는 FFN
   - Hidden dimension은 보통 d_model의 4배
   ```python
   ffn_output = FFN(attention_output)
   ```

### Sub-layer Connection
```python
# 각 sub-layer 주변에 적용
output = LayerNorm(x + Sublayer(x))  # Residual + Norm
```

## Decoder: 출력 생성의 핵심

### 구조
```python
class TransformerDecoder:
    def __init__(self, n_layers=6):
        self.layers = [DecoderLayer() for _ in range(n_layers)]
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x
```

### Decoder Layer 구성
각 Decoder Layer는 3개의 sub-layer로 구성:

1. **Masked Multi-Head Self-Attention**
   - Auto-regressive: 미래 position 정보 차단
   - Training 시 병렬 처리를 위한 masking
   ```python
   # Causal mask 적용
   masked_attention = MultiHeadAttention(Q=x, K=x, V=x, mask=causal_mask)
   ```

2. **Multi-Head Cross-Attention**
   - Encoder output을 Key, Value로 사용
   - Decoder의 현재 상태를 Query로 사용
   ```python
   cross_attention = MultiHeadAttention(
       Q=decoder_state, 
       K=encoder_output, 
       V=encoder_output
   )
   ```

3. **Position-wise Feed-Forward Network**
   - Encoder와 동일한 구조
   ```python
   ffn_output = FFN(cross_attention_output)
   ```

## Cross-Attention: Encoder-Decoder 연결

### 역할
- Decoder가 입력 정보에 접근하는 유일한 통로
- 각 출력 position이 입력의 어느 부분에 주목할지 결정

### 메커니즘
```python
def cross_attention(decoder_state, encoder_output):
    # decoder_state: [batch, tgt_len, d_model]
    # encoder_output: [batch, src_len, d_model]
    
    Q = linear_q(decoder_state)  # Decoder가 무엇을 찾고 있는가?
    K = linear_k(encoder_output)  # Encoder의 각 position 특성
    V = linear_v(encoder_output)  # Encoder의 실제 정보
    
    attention_weights = softmax(Q @ K.T / sqrt(d_k))
    output = attention_weights @ V
    return output
```

## Information Flow

### Training Phase
```
1. Encoder Processing:
   Input: "Hello world" → [Token IDs] → [Embeddings]
   ↓
   Encoder Layer 1-6 처리
   ↓
   Encoder Output: Context representations

2. Decoder Processing (Teacher Forcing):
   Input: "<start> Bonjour" → [Token IDs] → [Embeddings]
   ↓
   Masked Self-Attention (미래 단어 못 봄)
   ↓
   Cross-Attention with Encoder Output
   ↓
   Feed-Forward
   ↓
   Output: "Bonjour monde"
```

### Inference Phase
```
1. Encoder: 전체 입력 한 번에 처리
2. Decoder: Auto-regressive 생성
   - Step 1: <start> → "Bonjour"
   - Step 2: <start> Bonjour → "monde"
   - Step 3: <start> Bonjour monde → <end>
```

## Masking Strategies

### Padding Mask (Encoder & Decoder)
- 가변 길이 시퀀스 처리를 위한 padding token 무시
```python
padding_mask = (tokens != PAD_TOKEN_ID)
```

### Causal Mask (Decoder Only)
- Auto-regressive 특성 보장
- 미래 position 정보 차단
```python
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
```

## Key Design Choices

### 1. Stacking Layers
- 깊은 네트워크로 복잡한 패턴 학습
- 각 layer는 다른 수준의 abstraction 포착
- 보통 6개 layer 사용 (BERT, GPT 등은 더 많이 사용)

### 2. Parameter Sharing
- Encoder layers 간 파라미터 비공유
- Decoder layers 간 파라미터 비공유
- 각 layer가 다른 역할 수행

### 3. Bottleneck Architecture
- Encoder output이 정보 bottleneck 역할
- 입력의 핵심 정보만 압축하여 전달
- Decoder는 이를 바탕으로 출력 생성

## 실제 구현 예제

### Simple Encoder-Decoder
```python
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(n_layers, d_model, n_heads)
        self.decoder = TransformerDecoder(n_layers, d_model, n_heads)
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode
        src_emb = self.pos_encoding(self.embedding(src))
        encoder_output = self.encoder(src_emb, src_mask)
        
        # Decode
        tgt_emb = self.pos_encoding(self.embedding(tgt))
        decoder_output = self.decoder(
            tgt_emb, encoder_output, src_mask, tgt_mask
        )
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        return output
```

## Applications

### 번역 (Translation)
- Encoder: 소스 언어 처리
- Decoder: 타겟 언어 생성
- Cross-attention: 언어 간 alignment

### 요약 (Summarization)
- Encoder: 긴 문서 인코딩
- Decoder: 핵심 내용 추출 및 생성
- Cross-attention: 중요 정보 선택

### Question Answering
- Encoder: Context 문서 처리
- Decoder: 답변 생성
- Cross-attention: 관련 정보 추출

## 실습 포인트
1. Encoder output이 어떻게 Decoder에 전달되는지 추적
2. Cross-attention weights 시각화로 alignment 확인
3. Masking이 학습과 추론에 미치는 영향 이해
4. Teacher forcing vs. auto-regressive 생성 비교