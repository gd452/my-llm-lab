# 🎪 Batch Processing: 병렬 처리의 힘

## 🎯 학습 목표
- Batch 처리의 장점 이해
- Mini-batch 구현
- 효율적인 데이터 로딩

## 1. 왜 Batch Processing인가?

### 단일 샘플 vs Batch
```python
# 단일 샘플 처리 (느림)
for x in dataset:
    y = model(x)  # 한 번에 하나씩
    loss = compute_loss(y, target)
    update_weights()

# Batch 처리 (빠름)
for batch in dataloader:
    Y = model(batch)  # 여러 개 동시 처리!
    loss = compute_loss(Y, targets)
    update_weights()
```

### Batch 처리의 장점
1. **연산 효율성**: 행렬 연산으로 병렬 처리
2. **메모리 효율성**: 캐시 활용도 증가
3. **학습 안정성**: 노이즈 감소, 부드러운 수렴
4. **GPU 활용**: GPU는 병렬 처리에 최적화

## 2. Batch 차원 이해하기

### 차원 규약
```python
# 일반적인 차원 순서
# Images: (N, H, W, C) - TensorFlow
# Images: (N, C, H, W) - PyTorch
# Sequences: (N, T, D)
# Tabular: (N, D)

# N: Batch size
# H, W: Height, Width
# C: Channels
# T: Time steps / Sequence length
# D: Features / Dimensions
```

### Batch 차원 다루기
```python
import numpy as np

# Batch 차원 추가
single_image = np.random.randn(28, 28)  # (28, 28)
batch_image = single_image[np.newaxis, :]  # (1, 28, 28)

# Batch 차원 제거
predictions = np.random.randn(1, 10)  # (1, 10)
single_pred = predictions.squeeze(0)  # (10,)

# Batch 합치기
batch1 = np.random.randn(32, 784)
batch2 = np.random.randn(32, 784)
combined = np.concatenate([batch1, batch2], axis=0)  # (64, 784)
```

## 3. Mini-batch 구현

### DataLoader 구현
```python
class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        
    def __iter__(self):
        indices = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield self.X[batch_indices], self.y[batch_indices]
    
    def __len__(self):
        return self.n_batches

# 사용 예
X_train = np.random.randn(1000, 784)
y_train = np.random.randint(0, 10, 1000)

dataloader = DataLoader(X_train, y_train, batch_size=32)

for epoch in range(10):
    for X_batch, y_batch in dataloader:
        # 학습 수행
        print(f"Batch shape: {X_batch.shape}")
        break
    break
```

## 4. Batch 연산 최적화

### 벡터화된 활성화 함수
```python
def relu_batch(X):
    """
    Batch ReLU
    X: (batch_size, features)
    """
    return np.maximum(0, X)

def sigmoid_batch(X):
    """
    Batch Sigmoid (수치적으로 안정)
    """
    # 오버플로우 방지
    X = np.clip(X, -500, 500)
    return 1 / (1 + np.exp(-X))

def softmax_batch(X):
    """
    Batch Softmax
    X: (batch_size, num_classes)
    """
    # 수치 안정성을 위해 최댓값 빼기
    X_max = X.max(axis=1, keepdims=True)
    exp_X = np.exp(X - X_max)
    return exp_X / exp_X.sum(axis=1, keepdims=True)
```

### Batch 손실 함수
```python
def mse_loss_batch(y_pred, y_true):
    """
    Batch MSE Loss
    y_pred, y_true: (batch_size, output_dim)
    """
    return np.mean((y_pred - y_true) ** 2)

def cross_entropy_batch(y_pred, y_true):
    """
    Batch Cross Entropy Loss
    y_pred: (batch_size, num_classes) - probabilities
    y_true: (batch_size,) - class indices
    """
    batch_size = y_pred.shape[0]
    # 정답 클래스의 확률만 선택
    log_probs = -np.log(y_pred[np.arange(batch_size), y_true] + 1e-8)
    return np.mean(log_probs)
```

## 5. Batch Normalization

### 구현
```python
class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 학습 가능한 파라미터
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Running statistics (추론용)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        self.training = True
    
    def forward(self, X):
        """
        X: (batch_size, num_features)
        """
        if self.training:
            # 배치 통계 계산
            batch_mean = X.mean(axis=0)
            batch_var = X.var(axis=0)
            
            # Running statistics 업데이트
            self.running_mean = (self.momentum * self.running_mean + 
                                (1 - self.momentum) * batch_mean)
            self.running_var = (self.momentum * self.running_var + 
                               (1 - self.momentum) * batch_var)
            
            # 정규화
            X_norm = (X - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            # 추론 시 running statistics 사용
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        # 스케일과 시프트
        out = self.gamma * X_norm + self.beta
        
        # 역전파를 위해 저장
        self.cache = (X, X_norm, batch_mean, batch_var)
        
        return out
    
    def backward(self, dout):
        """
        역전파
        dout: (batch_size, num_features)
        """
        X, X_norm, mean, var = self.cache
        batch_size = X.shape[0]
        
        # 파라미터 그래디언트
        self.dgamma = (dout * X_norm).sum(axis=0)
        self.dbeta = dout.sum(axis=0)
        
        # 입력 그래디언트 (복잡!)
        dX_norm = dout * self.gamma
        dvar = ((dX_norm * (X - mean) * -0.5 * 
                (var + self.eps) ** (-1.5)).sum(axis=0))
        dmean = (dX_norm * -1 / np.sqrt(var + self.eps)).sum(axis=0)
        
        dX = (dX_norm / np.sqrt(var + self.eps) + 
              dvar * 2 * (X - mean) / batch_size + 
              dmean / batch_size)
        
        return dX
```

## 6. Batch 크기 선택

### Trade-offs
```python
# 작은 배치 (예: 32)
# + 메모리 효율적
# + 정규화 효과 (노이즈)
# - 느린 수렴
# - GPU 활용도 낮음

# 큰 배치 (예: 256, 512)
# + 빠른 학습
# + GPU 효율적
# - 메모리 많이 사용
# - Sharp minima 위험

# 적응적 배치 크기
def get_batch_size(epoch, initial_bs=32, max_bs=256):
    """에폭에 따라 배치 크기 증가"""
    return min(initial_bs * (2 ** (epoch // 10)), max_bs)
```

## 7. 메모리 관리

### Gradient Accumulation
```python
def train_with_gradient_accumulation(model, dataloader, accumulation_steps=4):
    """
    큰 배치를 시뮬레이션하기 위한 그래디언트 누적
    """
    optimizer.zero_grad()
    
    for i, (X_batch, y_batch) in enumerate(dataloader):
        # Forward pass
        predictions = model(X_batch)
        loss = compute_loss(predictions, y_batch)
        
        # Backward pass (그래디언트 누적)
        loss = loss / accumulation_steps
        loss.backward()
        
        # accumulation_steps마다 업데이트
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

### 메모리 효율적인 연산
```python
def memory_efficient_attention(Q, K, V, chunk_size=32):
    """
    청크 단위로 Attention 계산 (메모리 절약)
    Q, K, V: (batch_size, seq_len, d_model)
    """
    batch_size, seq_len, d_model = Q.shape
    outputs = []
    
    for i in range(0, seq_len, chunk_size):
        q_chunk = Q[:, i:i+chunk_size]
        
        # 청크별 attention
        scores = q_chunk @ K.transpose(-2, -1) / np.sqrt(d_model)
        attention_weights = softmax_batch(scores)
        output_chunk = attention_weights @ V
        
        outputs.append(output_chunk)
    
    return np.concatenate(outputs, axis=1)
```

## 8. 실전 예제: Mini-batch SGD

```python
class MiniBatchSGD:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.lr = learning_rate
    
    def train_epoch(self, dataloader):
        epoch_loss = 0
        n_batches = 0
        
        for X_batch, y_batch in dataloader:
            # Forward pass
            predictions = self.model.forward(X_batch)
            
            # Compute loss
            loss = cross_entropy_batch(predictions, y_batch)
            epoch_loss += loss
            
            # Backward pass
            grad_output = self.compute_grad_loss(predictions, y_batch)
            self.model.backward(grad_output)
            
            # Update weights
            self.update_parameters()
            
            n_batches += 1
        
        return epoch_loss / n_batches
    
    def compute_grad_loss(self, predictions, y_true):
        """Cross entropy gradient"""
        batch_size = predictions.shape[0]
        grad = predictions.copy()
        grad[np.arange(batch_size), y_true] -= 1
        return grad / batch_size
    
    def update_parameters(self):
        """파라미터 업데이트"""
        for param, grad in self.model.get_params_and_grads():
            param -= self.lr * grad
```

## 💡 Batch Processing 최적화 팁

1. **2의 거듭제곱**: 배치 크기를 32, 64, 128 등으로
2. **Prefetching**: 다음 배치를 미리 로드
3. **Pin Memory**: GPU 전송 속도 향상
4. **Mixed Precision**: FP16으로 메모리 절약

## 🔍 프로파일링

```python
import time

def profile_batch_sizes(model, X, y):
    """다양한 배치 크기 성능 측정"""
    batch_sizes = [1, 8, 32, 128, 512]
    
    for bs in batch_sizes:
        dataloader = DataLoader(X, y, batch_size=bs)
        
        start = time.time()
        for X_batch, y_batch in dataloader:
            _ = model(X_batch)
        
        elapsed = time.time() - start
        throughput = len(X) / elapsed
        
        print(f"Batch size {bs:3d}: "
              f"{elapsed:.2f}s, "
              f"{throughput:.0f} samples/sec")
```

## 📝 연습 문제

1. Variable batch size를 지원하는 DataLoader를 구현하세요.
2. Batch 단위 Dropout을 구현하세요.
3. Learning rate warm-up을 포함한 스케줄러를 만드세요.

## 다음 단계

손실 함수의 세계로! → [04_loss_functions.md](04_loss_functions.md)