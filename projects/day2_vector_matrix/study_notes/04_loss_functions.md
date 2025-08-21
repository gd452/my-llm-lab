# 📉 Loss Functions: 학습의 나침반

## 🎯 학습 목표
- 다양한 손실 함수 이해
- 수치적으로 안정적인 구현
- 적절한 손실 함수 선택

## 1. 손실 함수의 역할

손실 함수는 모델의 예측과 정답 사이의 차이를 측정합니다.
이는 최적화의 목표가 되며, 학습의 방향을 결정합니다.

### 좋은 손실 함수의 조건
1. **미분 가능**: 역전파를 위해 필수
2. **볼록성**: 최적화가 용이 (항상 가능한 건 아님)
3. **해석 가능**: 값의 의미가 명확
4. **수치 안정성**: 오버플로우/언더플로우 방지

## 2. 회귀 손실 함수

### Mean Squared Error (MSE)
```python
def mse_loss(y_pred, y_true):
    """
    평균 제곱 오차
    y_pred, y_true: (batch_size, output_dim)
    """
    return np.mean((y_pred - y_true) ** 2)

def mse_grad(y_pred, y_true):
    """MSE의 그래디언트"""
    batch_size = y_pred.shape[0]
    return 2 * (y_pred - y_true) / batch_size
```

### Mean Absolute Error (MAE)
```python
def mae_loss(y_pred, y_true):
    """
    평균 절대 오차 (이상치에 강건)
    """
    return np.mean(np.abs(y_pred - y_true))

def mae_grad(y_pred, y_true):
    """MAE의 그래디언트"""
    batch_size = y_pred.shape[0]
    return np.sign(y_pred - y_true) / batch_size
```

### Huber Loss
```python
def huber_loss(y_pred, y_true, delta=1.0):
    """
    Huber Loss: MSE와 MAE의 장점 결합
    작은 오차에는 MSE, 큰 오차에는 MAE
    """
    error = y_pred - y_true
    is_small_error = np.abs(error) <= delta
    
    small_error_loss = 0.5 * error ** 2
    large_error_loss = delta * (np.abs(error) - 0.5 * delta)
    
    return np.mean(np.where(is_small_error, small_error_loss, large_error_loss))

def huber_grad(y_pred, y_true, delta=1.0):
    """Huber Loss의 그래디언트"""
    error = y_pred - y_true
    batch_size = y_pred.shape[0]
    
    grad = np.where(
        np.abs(error) <= delta,
        error,  # MSE 부분
        delta * np.sign(error)  # MAE 부분
    )
    return grad / batch_size
```

## 3. 분류 손실 함수

### Cross Entropy Loss
```python
def cross_entropy_loss(y_pred, y_true, eps=1e-8):
    """
    Cross Entropy Loss (안정적 구현)
    y_pred: (batch_size, num_classes) - probabilities
    y_true: (batch_size,) - class indices
    """
    batch_size = y_pred.shape[0]
    
    # Clip to prevent log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # 정답 클래스의 확률만 선택
    correct_log_probs = -np.log(y_pred[np.arange(batch_size), y_true])
    
    return np.mean(correct_log_probs)

def cross_entropy_grad(y_pred, y_true):
    """
    Cross Entropy의 그래디언트 (Softmax 출력 가정)
    """
    batch_size = y_pred.shape[0]
    grad = y_pred.copy()
    grad[np.arange(batch_size), y_true] -= 1
    return grad / batch_size
```

### Binary Cross Entropy
```python
def binary_cross_entropy(y_pred, y_true, eps=1e-8):
    """
    이진 분류용 Cross Entropy
    y_pred, y_true: (batch_size,)
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    loss = -(y_true * np.log(y_pred) + 
             (1 - y_true) * np.log(1 - y_pred))
    
    return np.mean(loss)

def binary_cross_entropy_grad(y_pred, y_true, eps=1e-8):
    """BCE의 그래디언트"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    batch_size = len(y_pred)
    
    grad = -(y_true / y_pred - (1 - y_true) / (1 - y_pred))
    return grad / batch_size
```

### Focal Loss
```python
def focal_loss(y_pred, y_true, gamma=2.0, alpha=0.25, eps=1e-8):
    """
    Focal Loss: 클래스 불균형 문제 해결
    어려운 샘플에 더 집중
    """
    batch_size = y_pred.shape[0]
    num_classes = y_pred.shape[1]
    
    # One-hot encoding
    y_true_one_hot = np.zeros_like(y_pred)
    y_true_one_hot[np.arange(batch_size), y_true] = 1
    
    # Clip predictions
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Focal loss 계산
    ce = -y_true_one_hot * np.log(y_pred)
    focal_weight = (1 - y_pred) ** gamma
    fl = alpha * focal_weight * ce
    
    return np.mean(np.sum(fl, axis=1))
```

## 4. 수치 안정성 기법

### LogSumExp Trick
```python
def logsumexp(x, axis=None):
    """
    수치적으로 안정적인 log(sum(exp(x)))
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    return x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))

def stable_softmax_cross_entropy(logits, y_true):
    """
    Softmax + Cross Entropy를 한 번에 (안정적)
    logits: (batch_size, num_classes) - raw scores
    """
    batch_size = logits.shape[0]
    
    # LogSumExp trick
    log_probs = logits - logsumexp(logits, axis=1)
    
    # Cross entropy
    loss = -log_probs[np.arange(batch_size), y_true]
    
    return np.mean(loss)
```

### Gradient Clipping
```python
def clip_gradients(gradients, max_norm=1.0):
    """
    그래디언트 클리핑 (폭발 방지)
    """
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients))
    
    if total_norm > max_norm:
        scale = max_norm / total_norm
        gradients = [g * scale for g in gradients]
    
    return gradients
```

## 5. 정규화 손실

### L2 Regularization
```python
def l2_regularization(params, lambda_reg=0.01):
    """
    L2 정규화 (Weight Decay)
    """
    l2_loss = 0
    for param in params:
        l2_loss += np.sum(param ** 2)
    
    return lambda_reg * l2_loss

def l2_regularization_grad(params, lambda_reg=0.01):
    """L2 정규화의 그래디언트"""
    return [2 * lambda_reg * param for param in params]
```

### L1 Regularization
```python
def l1_regularization(params, lambda_reg=0.01):
    """
    L1 정규화 (Sparsity 유도)
    """
    l1_loss = 0
    for param in params:
        l1_loss += np.sum(np.abs(param))
    
    return lambda_reg * l1_loss

def l1_regularization_grad(params, lambda_reg=0.01):
    """L1 정규화의 그래디언트"""
    return [lambda_reg * np.sign(param) for param in params]
```

## 6. 고급 손실 함수

### Contrastive Loss
```python
def contrastive_loss(embeddings, labels, margin=1.0):
    """
    Contrastive Loss: 유사도 학습
    embeddings: (batch_size, embedding_dim)
    labels: (batch_size,) - 0: different, 1: similar
    """
    # Pairwise distances
    distances = pairwise_distances(embeddings)
    
    # Contrastive loss
    similar_loss = labels * distances ** 2
    dissimilar_loss = (1 - labels) * np.maximum(0, margin - distances) ** 2
    
    return np.mean(similar_loss + dissimilar_loss)
```

### Triplet Loss
```python
def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Triplet Loss: 순위 학습
    anchor: (batch_size, embedding_dim)
    positive: (batch_size, embedding_dim) - 같은 클래스
    negative: (batch_size, embedding_dim) - 다른 클래스
    """
    # 거리 계산
    pos_dist = np.sum((anchor - positive) ** 2, axis=1)
    neg_dist = np.sum((anchor - negative) ** 2, axis=1)
    
    # Triplet loss
    loss = np.maximum(0, pos_dist - neg_dist + margin)
    
    return np.mean(loss)
```

## 7. 손실 함수 조합

```python
class CombinedLoss:
    """여러 손실 함수 조합"""
    
    def __init__(self, losses, weights=None):
        self.losses = losses
        self.weights = weights or [1.0] * len(losses)
    
    def __call__(self, y_pred, y_true):
        total_loss = 0
        
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(y_pred, y_true)
        
        return total_loss

# 사용 예
combined_loss = CombinedLoss(
    losses=[cross_entropy_loss, l2_regularization],
    weights=[1.0, 0.01]
)
```

## 8. 실전 예제: 자동 손실 함수 선택

```python
class AdaptiveLoss:
    """태스크에 맞는 손실 함수 자동 선택"""
    
    def __init__(self, task_type='classification', num_classes=None):
        self.task_type = task_type
        self.num_classes = num_classes
        
        if task_type == 'classification':
            if num_classes == 2:
                self.loss_fn = binary_cross_entropy
            else:
                self.loss_fn = cross_entropy_loss
        elif task_type == 'regression':
            self.loss_fn = mse_loss
        elif task_type == 'ranking':
            self.loss_fn = triplet_loss
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def __call__(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)
    
    def get_info(self):
        """손실 함수 정보 반환"""
        return {
            'task_type': self.task_type,
            'loss_function': self.loss_fn.__name__,
            'num_classes': self.num_classes
        }
```

## 💡 손실 함수 선택 가이드

| 태스크 | 추천 손실 함수 | 이유 |
|--------|---------------|------|
| 이진 분류 | BCE | 확률 해석 가능 |
| 다중 분류 | Cross Entropy | 클래스 간 경쟁 |
| 회귀 | MSE | 미분 용이 |
| 이상치 있는 회귀 | Huber | 강건성 |
| 클래스 불균형 | Focal Loss | 어려운 샘플 집중 |
| 임베딩 학습 | Triplet Loss | 상대적 거리 |

## 🔍 디버깅 팁

```python
def debug_loss(loss_fn, y_pred, y_true):
    """손실 함수 디버깅"""
    loss = loss_fn(y_pred, y_true)
    
    print(f"Loss value: {loss:.6f}")
    print(f"Is NaN: {np.isnan(loss)}")
    print(f"Is Inf: {np.isinf(loss)}")
    
    if hasattr(loss_fn, '__name__'):
        print(f"Loss function: {loss_fn.__name__}")
    
    # 그래디언트 체크
    if loss_fn.__name__.endswith('_loss'):
        grad_fn_name = loss_fn.__name__.replace('_loss', '_grad')
        if grad_fn_name in globals():
            grad = globals()[grad_fn_name](y_pred, y_true)
            print(f"Gradient norm: {np.linalg.norm(grad):.6f}")
```

## 📝 연습 문제

1. KL Divergence 손실 함수를 구현하세요.
2. Smooth L1 Loss를 구현하세요.
3. Label Smoothing을 포함한 Cross Entropy를 구현하세요.

## 🎉 축하합니다!

Day 2의 모든 학습 자료를 완료했습니다!
이제 벡터 연산의 힘을 활용할 수 있습니다.

다음 단계: Day 3 - Attention Mechanism으로 진행하세요!