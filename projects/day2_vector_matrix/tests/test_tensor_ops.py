"""
🧪 Tensor Operations 테스트
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from core.tensor_ops import (
    matmul, relu, sigmoid, softmax,
    mse_loss, cross_entropy, batch_norm,
    one_hot, accuracy
)


class TestActivations:
    """활성화 함수 테스트"""
    
    def test_relu(self):
        x = np.array([-2, -1, 0, 1, 2])
        result = relu(x)
        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(result, expected)
    
    def test_sigmoid(self):
        x = np.array([0])
        result = sigmoid(x)
        np.testing.assert_almost_equal(result, 0.5)
        
        # 큰 값 테스트 (오버플로우 방지)
        x = np.array([1000, -1000])
        result = sigmoid(x)
        assert result[0] > 0.99
        assert result[1] < 0.01
    
    def test_softmax(self):
        # 2D 배열 테스트
        x = np.array([[1, 2, 3],
                      [1, 2, 3]])
        result = softmax(x)
        
        # 각 행의 합이 1
        np.testing.assert_almost_equal(result.sum(axis=1), [1, 1])
        
        # 수치 안정성 테스트
        x = np.array([[1000, 1001, 1002]])
        result = softmax(x)
        assert not np.any(np.isnan(result))
        np.testing.assert_almost_equal(result.sum(), 1)


class TestLossFunctions:
    """손실 함수 테스트"""
    
    def test_mse_loss(self):
        y_pred = np.array([[1, 2], [3, 4]])
        y_true = np.array([[1, 2], [3, 4]])
        loss = mse_loss(y_pred, y_true)
        assert loss == 0
        
        y_pred = np.array([[1, 2], [3, 4]])
        y_true = np.array([[0, 0], [0, 0]])
        loss = mse_loss(y_pred, y_true)
        expected = (1 + 4 + 9 + 16) / 4
        np.testing.assert_almost_equal(loss, expected)
    
    def test_cross_entropy(self):
        # 완벽한 예측
        y_pred = np.array([[0, 1, 0],
                          [1, 0, 0],
                          [0, 0, 1]])
        y_true = np.array([1, 0, 2])
        loss = cross_entropy(y_pred, y_true)
        np.testing.assert_almost_equal(loss, 0, decimal=5)
        
        # 균등 분포
        y_pred = np.array([[1/3, 1/3, 1/3],
                          [1/3, 1/3, 1/3]])
        y_true = np.array([0, 1])
        loss = cross_entropy(y_pred, y_true)
        expected = -np.log(1/3)
        np.testing.assert_almost_equal(loss, expected, decimal=5)


class TestBatchNorm:
    """배치 정규화 테스트"""
    
    def test_batch_norm_training(self):
        np.random.seed(42)
        x = np.random.randn(32, 10)
        gamma = np.ones(10)
        beta = np.zeros(10)
        
        x_norm, (mean, var) = batch_norm(x, gamma, beta, training=True)
        
        # 정규화 후 평균은 0, 분산은 1에 가까워야 함
        np.testing.assert_almost_equal(x_norm.mean(axis=0), np.zeros(10), decimal=5)
        np.testing.assert_almost_equal(x_norm.var(axis=0), np.ones(10), decimal=4)
    
    def test_batch_norm_inference(self):
        np.random.seed(42)
        x = np.random.randn(32, 10)
        gamma = np.ones(10)
        beta = np.zeros(10)
        
        # Running statistics
        running_mean = np.random.randn(10)
        running_var = np.random.rand(10) + 0.5
        
        x_norm, _ = batch_norm(
            x, gamma, beta,
            running_mean=running_mean,
            running_var=running_var,
            training=False
        )
        
        # 추론 모드에서는 running statistics 사용
        expected = (x - running_mean) / np.sqrt(running_var + 1e-5)
        np.testing.assert_almost_equal(x_norm, expected)


class TestUtilities:
    """유틸리티 함수 테스트"""
    
    def test_one_hot(self):
        indices = np.array([0, 2, 1, 0])
        result = one_hot(indices, num_classes=3)
        expected = np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]])
        np.testing.assert_array_equal(result, expected)
    
    def test_accuracy(self):
        # 확률에서 정확도
        y_pred = np.array([[0.1, 0.9, 0],
                          [0.8, 0.1, 0.1],
                          [0.2, 0.3, 0.5]])
        y_true = np.array([1, 0, 2])
        acc = accuracy(y_pred, y_true)
        assert acc == 1.0
        
        # 클래스에서 정확도
        y_pred = np.array([1, 0, 2])
        y_true = np.array([1, 1, 2])
        acc = accuracy(y_pred, y_true)
        np.testing.assert_almost_equal(acc, 2/3)


class TestMatrixOps:
    """행렬 연산 테스트"""
    
    def test_matmul(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        result = matmul(A, B)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result, expected)
        
        # Batch matmul
        A = np.random.randn(32, 10, 20)
        B = np.random.randn(32, 20, 30)
        result = matmul(A, B)
        assert result.shape == (32, 10, 30)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])