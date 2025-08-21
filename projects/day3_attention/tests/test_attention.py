"""
🧪 Attention 메커니즘 테스트
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from core.attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    positional_encoding,
    create_causal_mask,
    create_padding_mask,
    add_positional_encoding
)


class TestScaledDotProductAttention:
    """Scaled Dot-Product Attention 테스트"""
    
    def test_basic_attention(self):
        """기본 attention 동작 테스트"""
        Q = np.random.randn(10, 64)
        K = np.random.randn(10, 64)
        V = np.random.randn(10, 128)
        
        output, weights = scaled_dot_product_attention(Q, K, V)
        
        # Shape 확인
        assert output.shape == (10, 128)
        assert weights.shape == (10, 10)
        
        # Attention weights 합이 1
        np.testing.assert_almost_equal(weights.sum(axis=1), np.ones(10))
        
        # 모든 weights가 0~1 사이
        assert np.all(weights >= 0) and np.all(weights <= 1)
    
    def test_batch_attention(self):
        """배치 처리 테스트"""
        batch_size = 4
        seq_len = 8
        d_k = 32
        d_v = 16
        
        Q = np.random.randn(batch_size, seq_len, d_k)
        K = np.random.randn(batch_size, seq_len, d_k)
        V = np.random.randn(batch_size, seq_len, d_v)
        
        output, weights = scaled_dot_product_attention(Q, K, V)
        
        assert output.shape == (batch_size, seq_len, d_v)
        assert weights.shape == (batch_size, seq_len, seq_len)
    
    def test_with_mask(self):
        """마스킹 테스트"""
        seq_len = 5
        Q = K = V = np.random.randn(seq_len, 16)
        
        # Causal mask
        mask = create_causal_mask(seq_len)
        output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        
        # 미래 위치의 attention이 0인지 확인
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert weights[i, j] < 1e-8
    
    def test_attention_pattern(self):
        """특정 attention 패턴 테스트"""
        # Identity attention (자기 자신에만 주목)
        seq_len = 4
        d = 8
        
        # Q와 K를 직교하게 설정
        Q = np.eye(seq_len, d)
        K = np.eye(seq_len, d)
        V = np.random.randn(seq_len, d)
        
        output, weights = scaled_dot_product_attention(Q, K, V)
        
        # 대각선이 가장 큰 값
        for i in range(seq_len):
            assert np.argmax(weights[i]) == i


class TestMultiHeadAttention:
    """Multi-Head Attention 테스트"""
    
    def test_multi_head_shapes(self):
        """Multi-head 출력 shape 테스트"""
        d_model = 512
        num_heads = 8
        seq_len = 20
        
        mha = MultiHeadAttention(d_model, num_heads)
        x = np.random.randn(seq_len, d_model)
        
        output, weights = mha.forward(x, x, x)
        
        assert output.shape == (seq_len, d_model)
        assert weights.shape == (num_heads, seq_len, seq_len)
    
    def test_head_dimension_split(self):
        """Head 차원 분할 테스트"""
        d_model = 256
        num_heads = 4
        d_k = d_model // num_heads  # 64
        
        mha = MultiHeadAttention(d_model, num_heads)
        assert mha.d_k == d_k
    
    def test_batch_multi_head(self):
        """배치 처리 multi-head 테스트"""
        batch_size = 2
        seq_len = 10
        d_model = 128
        num_heads = 4
        
        mha = MultiHeadAttention(d_model, num_heads)
        x = np.random.randn(batch_size, seq_len, d_model)
        
        output, weights = mha.forward(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    def test_cross_attention(self):
        """Cross-attention 테스트"""
        d_model = 64
        num_heads = 2
        
        mha = MultiHeadAttention(d_model, num_heads)
        
        # 다른 길이의 시퀀스
        query = np.random.randn(5, d_model)  # 5 queries
        key = np.random.randn(8, d_model)    # 8 keys
        value = np.random.randn(8, d_model)  # 8 values
        
        output, weights = mha.forward(query, key, value)
        
        assert output.shape == (5, d_model)
        assert weights.shape == (num_heads, 5, 8)


class TestPositionalEncoding:
    """Positional Encoding 테스트"""
    
    def test_pe_shape(self):
        """PE shape 테스트"""
        seq_len = 100
        d_model = 512
        
        pe = positional_encoding(seq_len, d_model)
        assert pe.shape == (seq_len, d_model)
    
    def test_pe_range(self):
        """PE 값 범위 테스트"""
        pe = positional_encoding(50, 128)
        
        # Sin/Cos 값은 -1~1 사이
        assert pe.min() >= -1.0
        assert pe.max() <= 1.0
    
    def test_pe_pattern(self):
        """PE 패턴 테스트"""
        pe = positional_encoding(100, 4)
        
        # 첫 번째 위치는 특정 패턴
        # PE(0, 0) = sin(0) = 0
        assert abs(pe[0, 0]) < 1e-10
        
        # PE(0, 1) = cos(0) = 1
        assert abs(pe[0, 1] - 1.0) < 1e-10
    
    def test_add_pe(self):
        """PE 추가 테스트"""
        # 2D 입력
        x_2d = np.random.randn(10, 64)
        x_with_pe = add_positional_encoding(x_2d)
        assert x_with_pe.shape == x_2d.shape
        
        # 3D 입력 (배치)
        x_3d = np.random.randn(2, 10, 64)
        x_with_pe = add_positional_encoding(x_3d)
        assert x_with_pe.shape == x_3d.shape


class TestMasking:
    """마스킹 함수 테스트"""
    
    def test_causal_mask(self):
        """Causal mask 테스트"""
        mask = create_causal_mask(4)
        
        expected = np.array([
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        
        np.testing.assert_array_equal(mask, expected)
    
    def test_padding_mask(self):
        """Padding mask 테스트"""
        # 시퀀스 with padding (0이 padding)
        seq = np.array([
            [1, 2, 3, 0, 0],
            [1, 2, 0, 0, 0]
        ])
        
        mask = create_padding_mask(seq, pad_idx=0)
        
        assert mask.shape == (2, 1, 1, 5)
        
        # 첫 번째 시퀀스: 마지막 2개가 padding
        assert mask[0, 0, 0, 3] == 1
        assert mask[0, 0, 0, 4] == 1
        
        # 두 번째 시퀀스: 마지막 3개가 padding
        assert mask[1, 0, 0, 2] == 1
        assert mask[1, 0, 0, 3] == 1
        assert mask[1, 0, 0, 4] == 1


class TestAttentionProperties:
    """Attention의 수학적 특성 테스트"""
    
    def test_attention_sum_to_one(self):
        """Attention 가중치 합이 1인지 테스트"""
        for _ in range(10):
            Q = np.random.randn(5, 32)
            K = np.random.randn(5, 32)
            V = np.random.randn(5, 16)
            
            _, weights = scaled_dot_product_attention(Q, K, V)
            
            # 각 query의 attention 합이 1
            np.testing.assert_almost_equal(
                weights.sum(axis=1), 
                np.ones(5),
                decimal=6
            )
    
    def test_self_attention_permutation(self):
        """Self-attention 순열 불변성 테스트"""
        seq_len = 6
        d_model = 32
        
        mha = MultiHeadAttention(d_model, num_heads=4)
        
        # 원본 입력
        x = np.random.randn(seq_len, d_model)
        output1, _ = mha.forward(x, x, x)
        
        # 순열 적용
        perm = np.random.permutation(seq_len)
        x_perm = x[perm]
        output2, _ = mha.forward(x_perm, x_perm, x_perm)
        
        # 출력도 같은 순열 따름
        output2_restored = output2[np.argsort(perm)]
        
        # 거의 같아야 함 (수치 오차 허용)
        np.testing.assert_almost_equal(output1, output2_restored, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])