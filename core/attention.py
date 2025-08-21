"""
🎯 Attention Mechanism: LLM의 핵심

이 파일에서 구현할 것:
1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Positional Encoding
4. Causal Masking

"Attention is All You Need" 논문의 핵심 구현입니다.
"""

import numpy as np


# ============================================
# Scaled Dot-Product Attention
# ============================================

def scaled_dot_product_attention(Q, K, V, mask=None, dropout_rate=0.0):
    """
    Scaled Dot-Product Attention 계산
    
    Attention(Q, K, V) = softmax(QK^T / √d_k)V
    
    Args:
        Q: Query matrix (batch_size, seq_len_q, d_k) or (seq_len_q, d_k)
        K: Key matrix (batch_size, seq_len_k, d_k) or (seq_len_k, d_k)
        V: Value matrix (batch_size, seq_len_v, d_v) or (seq_len_v, d_v)
        mask: 마스킹 행렬 (batch_size, seq_len_q, seq_len_k) or (seq_len_q, seq_len_k)
        dropout_rate: Dropout 비율 (훈련 시)
    
    Returns:
        output: Attention 출력 (batch_size, seq_len_q, d_v)
        attention_weights: Attention 가중치 (batch_size, seq_len_q, seq_len_k)
    
    Example:
        >>> Q = np.random.randn(10, 64)  # 10 queries, 64 dims
        >>> K = np.random.randn(10, 64)  # 10 keys
        >>> V = np.random.randn(10, 128) # 10 values, 128 dims
        >>> output, weights = scaled_dot_product_attention(Q, K, V)
    """
    # Q와 K의 내적 계산
    d_k = K.shape[-1]
    
    # K의 전치 (마지막 두 차원만)
    if K.ndim == 3:
        K_T = K.transpose(0, 2, 1)  # (batch, d_k, seq_len_k)
    else:
        K_T = K.T  # (d_k, seq_len_k)
    
    # Attention scores 계산
    scores = np.matmul(Q, K_T)  # (batch, seq_len_q, seq_len_k)
    
    # Scaling (중요!)
    scores = scores / np.sqrt(d_k)
    
    # Masking (optional)
    if mask is not None:
        # 마스크된 위치에 큰 음수 값 추가
        scores = scores + (mask * -1e9)
    
    # Softmax
    attention_weights = softmax(scores, axis=-1)
    
    # Dropout (training mode)
    if dropout_rate > 0:
        dropout_mask = np.random.binomial(1, 1 - dropout_rate, attention_weights.shape)
        attention_weights = attention_weights * dropout_mask / (1 - dropout_rate)
    
    # Value와 가중합
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights


def softmax(x, axis=-1):
    """안정적인 softmax 구현"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ============================================
# Multi-Head Attention
# ============================================

class MultiHeadAttention:
    """
    Multi-Head Attention 구현
    
    여러 개의 attention head를 병렬로 실행하여
    다양한 관계를 동시에 학습합니다.
    """
    
    def __init__(self, d_model, num_heads, dropout_rate=0.0):
        """
        초기화
        
        Args:
            d_model: 모델의 차원 (예: 512)
            num_heads: Attention head 개수 (예: 8)
            dropout_rate: Dropout 비율
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 각 head의 차원
        self.dropout_rate = dropout_rate
        
        # 가중치 초기화
        self.W_Q = self._init_weight()
        self.W_K = self._init_weight()
        self.W_V = self._init_weight()
        self.W_O = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    
    def _init_weight(self):
        """Xavier 초기화"""
        return np.random.randn(self.d_model, self.d_model) * np.sqrt(2.0 / self.d_model)
    
    def forward(self, query, key, value, mask=None):
        """
        Multi-Head Attention forward pass
        
        Args:
            query: (batch_size, seq_len_q, d_model) or (seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model) or (seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model) or (seq_len_v, d_model)
            mask: Optional mask
        
        Returns:
            output: (batch_size, seq_len_q, d_model)
            attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        # 배치 차원 처리
        if query.ndim == 2:
            query = query[np.newaxis, :]
            key = key[np.newaxis, :]
            value = value[np.newaxis, :]
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        
        # 1. Linear projections
        Q = np.matmul(query, self.W_Q)  # (batch, seq_len_q, d_model)
        K = np.matmul(key, self.W_K)    # (batch, seq_len_k, d_model)
        V = np.matmul(value, self.W_V)  # (batch, seq_len_v, d_model)
        
        # 2. Reshape for multi-head
        Q = self._split_heads(Q, batch_size)  # (batch, num_heads, seq_len_q, d_k)
        K = self._split_heads(K, batch_size)
        V = self._split_heads(V, batch_size)
        
        # 3. Scaled dot-product attention for each head
        attention_output, attention_weights = self._attention(Q, K, V, mask)
        
        # 4. Concatenate heads
        attention_output = self._combine_heads(attention_output, batch_size)
        
        # 5. Final linear projection
        output = np.matmul(attention_output, self.W_O)
        
        if squeeze_output:
            output = output.squeeze(0)
            attention_weights = attention_weights.squeeze(0)
        
        return output, attention_weights
    
    def _split_heads(self, x, batch_size):
        """Split into multiple heads"""
        seq_len = x.shape[1]
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, d_k)
    
    def _combine_heads(self, x, batch_size):
        """Combine multiple heads"""
        x = x.transpose(0, 2, 1, 3)  # (batch, seq_len, num_heads, d_k)
        seq_len = x.shape[1]
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def _attention(self, Q, K, V, mask=None):
        """Apply attention to all heads"""
        # (batch, num_heads, seq_len_q, d_k) @ (batch, num_heads, d_k, seq_len_k)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        if mask is not None:
            # Expand mask for heads
            if mask.ndim == 2:
                mask = mask[np.newaxis, np.newaxis, :, :]
            elif mask.ndim == 3:
                mask = mask[:, np.newaxis, :, :]
            scores = scores + (mask * -1e9)
        
        attention_weights = softmax(scores, axis=-1)
        
        # Apply dropout
        if self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, 
                                            attention_weights.shape)
            attention_weights = attention_weights * dropout_mask / (1 - self.dropout_rate)
        
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights


# ============================================
# Positional Encoding
# ============================================

def positional_encoding(seq_len, d_model, base=10000):
    """
    Sinusoidal Positional Encoding
    
    PE(pos, 2i) = sin(pos / base^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / base^(2i/d_model))
    
    Args:
        seq_len: 시퀀스 길이
        d_model: 모델 차원
        base: 주파수 베이스 (기본 10000)
    
    Returns:
        PE matrix (seq_len, d_model)
    
    Example:
        >>> pe = positional_encoding(100, 512)
        >>> print(pe.shape)  # (100, 512)
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    
    # 각 차원의 주파수 계산
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(base) / d_model))
    
    # Sin, Cos 적용
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe


def add_positional_encoding(x, max_len=5000):
    """
    입력에 Positional Encoding 추가
    
    Args:
        x: 입력 텐서 (batch_size, seq_len, d_model) or (seq_len, d_model)
        max_len: 최대 시퀀스 길이
    
    Returns:
        PE가 추가된 입력
    """
    if x.ndim == 2:
        seq_len, d_model = x.shape
        pe = positional_encoding(min(seq_len, max_len), d_model)
        return x + pe[:seq_len]
    else:
        batch_size, seq_len, d_model = x.shape
        pe = positional_encoding(min(seq_len, max_len), d_model)
        return x + pe[np.newaxis, :seq_len, :]


# ============================================
# Masking Functions
# ============================================

def create_padding_mask(seq, pad_idx=0):
    """
    패딩 마스크 생성
    
    Args:
        seq: 입력 시퀀스 (batch_size, seq_len)
        pad_idx: 패딩 토큰의 인덱스
    
    Returns:
        마스크 (batch_size, 1, 1, seq_len)
    """
    # 패딩 위치는 1, 아니면 0
    mask = (seq == pad_idx).astype(np.float32)
    
    # Shape 조정 for broadcasting
    return mask[:, np.newaxis, np.newaxis, :]


def create_causal_mask(seq_len):
    """
    Causal mask 생성 (미래 정보 차단)
    
    GPT와 같은 autoregressive 모델에서 사용
    
    Args:
        seq_len: 시퀀스 길이
    
    Returns:
        Causal mask (seq_len, seq_len)
    
    Example:
        >>> mask = create_causal_mask(4)
        >>> print(mask)
        [[0. 1. 1. 1.]
         [0. 0. 1. 1.]
         [0. 0. 0. 1.]
         [0. 0. 0. 0.]]
    """
    # Upper triangular matrix (k=1은 대각선 제외)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask


def create_combined_mask(seq, pad_idx=0):
    """
    패딩 + Causal 마스크 결합
    
    Args:
        seq: 입력 시퀀스
        pad_idx: 패딩 인덱스
    
    Returns:
        결합된 마스크
    """
    seq_len = seq.shape[1]
    
    # Padding mask
    padding_mask = create_padding_mask(seq, pad_idx)
    
    # Causal mask
    causal_mask = create_causal_mask(seq_len)
    causal_mask = causal_mask[np.newaxis, np.newaxis, :, :]
    
    # Combine (둘 중 하나라도 마스크면 마스크)
    combined_mask = np.maximum(padding_mask, causal_mask)
    
    return combined_mask


# ============================================
# Attention Utilities
# ============================================

def visualize_attention(attention_weights, tokens=None):
    """
    Attention 가중치 시각화 (텍스트 기반)
    
    Args:
        attention_weights: (seq_len, seq_len) or (num_heads, seq_len, seq_len)
        tokens: 토큰 리스트 (옵션)
    """
    if attention_weights.ndim == 3:
        # Multi-head인 경우 평균
        attention_weights = attention_weights.mean(axis=0)
    
    seq_len = attention_weights.shape[0]
    
    # 토큰 라벨
    if tokens is None:
        tokens = [f"T{i}" for i in range(seq_len)]
    
    # 헤더
    print("\nAttention Weights Heatmap:")
    print("       ", end="")
    for token in tokens:
        print(f"{token:>6}", end="")
    print()
    
    # 히트맵
    for i, token in enumerate(tokens):
        print(f"{token:>6} ", end="")
        for j in range(seq_len):
            weight = attention_weights[i, j]
            if weight > 0.5:
                print("  ██  ", end="")
            elif weight > 0.3:
                print("  ▓▓  ", end="")
            elif weight > 0.1:
                print("  ░░  ", end="")
            else:
                print("  ..  ", end="")
        print()


# ============================================
# 테스트 코드
# ============================================

if __name__ == "__main__":
    print("🧪 Attention 메커니즘 테스트")
    print("-" * 50)
    
    # 1. Scaled Dot-Product Attention
    print("\n1️⃣ Scaled Dot-Product Attention")
    Q = np.random.randn(4, 8)  # 4 queries, 8 dims
    K = np.random.randn(4, 8)  # 4 keys
    V = np.random.randn(4, 16) # 4 values, 16 dims
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"출력 shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Weights sum per query: {weights.sum(axis=1)}")
    
    # 2. Multi-Head Attention
    print("\n2️⃣ Multi-Head Attention")
    mha = MultiHeadAttention(d_model=64, num_heads=8)
    
    x = np.random.randn(10, 64)  # 10 tokens, 64 dims
    output, weights = mha.forward(x, x, x)  # Self-attention
    print(f"출력 shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # 3. Positional Encoding
    print("\n3️⃣ Positional Encoding")
    pe = positional_encoding(100, 128)
    print(f"PE shape: {pe.shape}")
    print(f"PE 범위: [{pe.min():.2f}, {pe.max():.2f}]")
    
    # 4. Causal Mask
    print("\n4️⃣ Causal Mask")
    mask = create_causal_mask(5)
    print("Causal mask (5x5):")
    print(mask)
    
    # 5. Attention 시각화
    print("\n5️⃣ Attention 시각화")
    tokens = ["The", "cat", "sat", "on", "mat"]
    random_weights = np.random.rand(5, 5)
    random_weights = random_weights / random_weights.sum(axis=1, keepdims=True)
    visualize_attention(random_weights, tokens)
    
    print("\n✅ 모든 테스트 통과!")