"""
🔢 Tensor Operations: 벡터/행렬 연산의 핵심

이 파일에서 구현할 것:
1. 효율적인 행렬 연산
2. Softmax와 활성화 함수들
3. 손실 함수들
4. 배치 정규화

NumPy를 활용한 벡터화된 연산으로 
스칼라 연산보다 훨씬 빠른 처리가 가능합니다.
"""

import numpy as np


# ============================================
# 기본 연산
# ============================================

def matmul(a, b):
    """
    행렬곱 연산
    
    Args:
        a: (..., m, k) shape의 배열
        b: (..., k, n) shape의 배열
    
    Returns:
        (..., m, n) shape의 결과
    
    Example:
        >>> A = np.random.randn(32, 784)  # batch_size=32, features=784
        >>> W = np.random.randn(784, 128)  # 784 -> 128 변환
        >>> result = matmul(A, W)  # (32, 128)
    """
    return np.matmul(a, b)


# ============================================
# 활성화 함수 (Vectorized)
# ============================================

def relu(x):
    """
    ReLU 활성화 함수
    
    Args:
        x: 입력 배열
    
    Returns:
        max(0, x)
    """
    return np.maximum(0, x)


def sigmoid(x):
    """
    Sigmoid 활성화 함수 (수치적으로 안정)
    
    Args:
        x: 입력 배열
    
    Returns:
        1 / (1 + exp(-x))
    """
    # 오버플로우 방지
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    Tanh 활성화 함수
    
    Args:
        x: 입력 배열
    
    Returns:
        tanh(x)
    """
    return np.tanh(x)


def softmax(x, axis=-1):
    """
    Softmax 함수 (수치적으로 안정)
    
    Args:
        x: 입력 배열
        axis: softmax를 적용할 축
    
    Returns:
        확률 분포 (합이 1)
    
    Example:
        >>> logits = np.random.randn(32, 10)  # 32개 샘플, 10개 클래스
        >>> probs = softmax(logits)
        >>> np.allclose(probs.sum(axis=1), 1)  # True
    """
    # TODO: 구현
    # 힌트:
    # 1. 수치 안정성을 위해 최댓값 빼기
    # 2. exp 계산
    # 3. 정규화
    
    # 최댓값 빼기 (수치 안정성)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    
    # 정규화
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp_x


# ============================================
# 손실 함수
# ============================================

def mse_loss(y_pred, y_true):
    """
    Mean Squared Error Loss
    
    Args:
        y_pred: 예측값 (batch_size, output_dim)
        y_true: 실제값 (batch_size, output_dim)
    
    Returns:
        스칼라 손실값
    """
    return np.mean((y_pred - y_true) ** 2)


def cross_entropy(y_pred, y_true, eps=1e-8):
    """
    Cross Entropy Loss
    
    Args:
        y_pred: 예측 확률 (batch_size, num_classes)
        y_true: 정답 인덱스 (batch_size,) 또는 one-hot (batch_size, num_classes)
        eps: 수치 안정성을 위한 작은 값
    
    Returns:
        스칼라 손실값
    
    TODO: 구현
    힌트:
    1. y_true가 인덱스인지 one-hot인지 확인
    2. log(0) 방지를 위해 clipping
    3. 평균 손실 반환
    """
    # Clipping for numerical stability
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # y_true가 one-hot인지 인덱스인지 확인
    if y_true.ndim == 1 or y_true.shape[1] == 1:
        # 인덱스 형태
        batch_size = y_pred.shape[0]
        log_probs = -np.log(y_pred[np.arange(batch_size), y_true.astype(int)])
    else:
        # One-hot 형태
        log_probs = -np.sum(y_true * np.log(y_pred), axis=1)
    
    return np.mean(log_probs)


def softmax_cross_entropy(logits, y_true):
    """
    Softmax + Cross Entropy (수치적으로 안정)
    
    Args:
        logits: 로짓 (batch_size, num_classes)
        y_true: 정답 인덱스 (batch_size,)
    
    Returns:
        스칼라 손실값
    """
    # LogSumExp trick for numerical stability
    logits_max = np.max(logits, axis=1, keepdims=True)
    log_sum_exp = logits_max + np.log(np.sum(np.exp(logits - logits_max), axis=1, keepdims=True))
    
    batch_size = logits.shape[0]
    log_probs = logits - log_sum_exp
    
    # Cross entropy
    ce = -log_probs[np.arange(batch_size), y_true.astype(int)]
    
    return np.mean(ce)


# ============================================
# 배치 정규화
# ============================================

def batch_norm(x, gamma, beta, eps=1e-5, momentum=0.9, 
               running_mean=None, running_var=None, training=True):
    """
    Batch Normalization
    
    Args:
        x: 입력 (batch_size, features)
        gamma: 스케일 파라미터 (features,)
        beta: 시프트 파라미터 (features,)
        eps: 수치 안정성을 위한 작은 값
        momentum: running statistics 업데이트 비율
        running_mean: 추론용 평균 (features,)
        running_var: 추론용 분산 (features,)
        training: 학습/추론 모드
    
    Returns:
        정규화된 출력, (평균, 분산) - 역전파용
    
    TODO: 구현
    """
    if training:
        # 배치 통계 계산
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        
        # Running statistics 업데이트
        if running_mean is not None:
            running_mean[:] = momentum * running_mean + (1 - momentum) * mean
        if running_var is not None:
            running_var[:] = momentum * running_var + (1 - momentum) * var
    else:
        # 추론 시 running statistics 사용
        mean = running_mean if running_mean is not None else np.mean(x, axis=0)
        var = running_var if running_var is not None else np.var(x, axis=0)
    
    # 정규화
    x_norm = (x - mean) / np.sqrt(var + eps)
    
    # 스케일과 시프트
    out = gamma * x_norm + beta
    
    return out, (mean, var)


# ============================================
# 유틸리티 함수
# ============================================

def one_hot(indices, num_classes):
    """
    One-hot 인코딩
    
    Args:
        indices: 클래스 인덱스 (batch_size,)
        num_classes: 전체 클래스 수
    
    Returns:
        One-hot 벡터 (batch_size, num_classes)
    """
    batch_size = len(indices)
    one_hot_matrix = np.zeros((batch_size, num_classes))
    one_hot_matrix[np.arange(batch_size), indices] = 1
    return one_hot_matrix


def accuracy(y_pred, y_true):
    """
    분류 정확도 계산
    
    Args:
        y_pred: 예측 확률 (batch_size, num_classes) 또는 클래스 (batch_size,)
        y_true: 정답 클래스 (batch_size,)
    
    Returns:
        정확도 (0~1)
    """
    if y_pred.ndim == 2:
        # 확률에서 클래스 추출
        y_pred = np.argmax(y_pred, axis=1)
    
    return np.mean(y_pred == y_true)


def clip_gradients(gradients, max_norm=1.0):
    """
    그래디언트 클리핑
    
    Args:
        gradients: 그래디언트 리스트
        max_norm: 최대 norm
    
    Returns:
        클리핑된 그래디언트
    """
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients))
    
    if total_norm > max_norm:
        scale = max_norm / total_norm
        gradients = [g * scale for g in gradients]
    
    return gradients


# ============================================
# 테스트 코드
# ============================================

if __name__ == "__main__":
    print("🧪 Tensor Operations 테스트")
    print("-" * 50)
    
    # Softmax 테스트
    logits = np.random.randn(4, 3)
    probs = softmax(logits)
    print(f"Softmax 합: {probs.sum(axis=1)}")  # [1, 1, 1, 1]
    
    # Cross Entropy 테스트
    y_true = np.array([0, 1, 2, 1])
    loss = cross_entropy(probs, y_true)
    print(f"Cross Entropy Loss: {loss:.4f}")
    
    # Batch Norm 테스트
    x = np.random.randn(32, 10)
    gamma = np.ones(10)
    beta = np.zeros(10)
    x_norm, _ = batch_norm(x, gamma, beta)
    print(f"Batch Norm - Mean: {x_norm.mean(axis=0).mean():.4f}, "
          f"Var: {x_norm.var(axis=0).mean():.4f}")
    
    print("\n✅ 모든 테스트 통과!")