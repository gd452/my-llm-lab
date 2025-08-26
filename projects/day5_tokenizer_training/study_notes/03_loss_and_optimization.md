# Loss Functions and Optimization

## Loss Functions for Language Modeling

### Cross-Entropy Loss
가장 기본적인 language modeling loss

```python
# Mathematical formula
Loss = -Σ(y_true * log(y_pred))

# For language modeling
Loss = -log(P(next_token | context))
```

**특징:**
- Multi-class classification loss
- 각 position에서 다음 토큰 예측
- Softmax와 함께 사용

**구현:**
```python
def cross_entropy(logits, targets):
    # logits: [batch_size, seq_len, vocab_size]
    # targets: [batch_size, seq_len]
    
    # Softmax
    probs = softmax(logits)
    
    # Negative log likelihood
    loss = -log(probs[targets])
    
    return mean(loss)
```

### Perplexity
Language model의 품질 측정 지표

```python
Perplexity = exp(cross_entropy_loss)
```

**해석:**
- 낮을수록 좋음
- PPL=10: 평균적으로 10개 단어 중 선택하는 불확실성
- PPL=1: 완벽한 예측

### Label Smoothing
Overconfidence 방지

```python
def label_smoothing(targets, vocab_size, smoothing=0.1):
    confidence = 1.0 - smoothing
    smooth_value = smoothing / vocab_size
    
    # One-hot with smoothing
    smooth_targets = np.full((len(targets), vocab_size), smooth_value)
    smooth_targets[range(len(targets)), targets] = confidence
    
    return smooth_targets
```

## Optimization Algorithms

### 1. Stochastic Gradient Descent (SGD)
가장 기본적인 optimizer

```python
# Update rule
θ = θ - α * ∇L(θ)

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g
```

**장점:**
- 간단하고 이해하기 쉬움
- 메모리 효율적

**단점:**
- 느린 수렴
- Local minima에 빠지기 쉬움

### 2. SGD with Momentum
이전 업데이트 방향 고려

```python
# Update rule
v = β * v - α * ∇L(θ)
θ = θ + v

class SGDMomentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}
    
    def step(self, params, grads):
        for p, g in zip(params, grads):
            v = self.velocity.get(p, 0)
            v = self.momentum * v - self.lr * g
            p += v
            self.velocity[p] = v
```

### 3. Adam (Adaptive Moment Estimation)
현재 가장 인기 있는 optimizer

```python
# Update rules
m = β₁ * m + (1 - β₁) * ∇L(θ)  # First moment
v = β₂ * v + (1 - β₂) * ∇L(θ)²  # Second moment

# Bias correction
m_hat = m / (1 - β₁ᵗ)
v_hat = v / (1 - β₂ᵗ)

# Update
θ = θ - α * m_hat / (√v_hat + ε)
```

**하이퍼파라미터:**
- `lr`: 0.001 (default)
- `β₁`: 0.9 (momentum)
- `β₂`: 0.999 (RMSprop)
- `ε`: 1e-8 (numerical stability)

### 4. AdamW (Adam with Weight Decay)
Weight decay를 올바르게 적용한 Adam

```python
# Decoupled weight decay
θ = θ - α * (m_hat / (√v_hat + ε) + λ * θ)
```

### 5. Learning Rate Schedulers

**Linear Warmup**
```python
def linear_warmup(step, warmup_steps=1000):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0
```

**Exponential Decay**
```python
lr = initial_lr * decay_rate^(step / decay_steps)
```

**Cosine Annealing**
```python
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * t/T))
```

**OneCycleLR**
```python
# 3 phases:
# 1. Warmup: lr increases
# 2. Annealing: lr decreases  
# 3. Fine-tuning: very small lr
```

## Gradient Problems & Solutions

### 1. Vanishing Gradients
깊은 네트워크에서 gradient가 0에 가까워짐

**해결책:**
- Residual connections
- Better initialization (Xavier, He)
- Batch/Layer normalization
- ReLU activation

### 2. Exploding Gradients
Gradient가 너무 커짐

**해결책:**
```python
def gradient_clipping(gradients, max_norm=1.0):
    total_norm = sqrt(sum(g.norm()**2 for g in gradients))
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for g in gradients:
            g *= clip_coef
```

### 3. Dead Neurons
ReLU 사용 시 일부 뉴런이 비활성화

**해결책:**
- Leaky ReLU
- ELU, SELU
- Careful initialization

## Regularization Techniques

### 1. L2 Regularization (Weight Decay)
```python
loss = cross_entropy_loss + λ * sum(w**2)
```

### 2. Dropout
```python
def dropout(x, p=0.5, training=True):
    if not training:
        return x
    mask = np.random.binomial(1, 1-p, x.shape)
    return x * mask / (1-p)
```

### 3. Early Stopping
Validation loss가 증가하기 시작하면 중단

### 4. Data Augmentation
- Token replacement
- Back-translation
- Paraphrasing

## Optimization Tips

### 1. Hyperparameter Tuning
**중요도 순서:**
1. Learning rate
2. Batch size
3. Model architecture
4. Optimizer
5. Regularization

### 2. Learning Rate Finding
```python
def lr_finder(model, dataloader, start_lr=1e-7, end_lr=10):
    lrs = []
    losses = []
    
    for lr in np.logspace(np.log10(start_lr), np.log10(end_lr), 100):
        optimizer.lr = lr
        loss = train_one_batch(model, dataloader)
        
        lrs.append(lr)
        losses.append(loss)
        
        if loss > min(losses) * 4:  # Stop if loss explodes
            break
    
    # Plot and find optimal LR
    plot(lrs, losses)
```

### 3. Batch Size Selection
**Large batch:**
- 안정적인 gradient
- 빠른 학습 (GPU 활용)
- Generalization 문제 가능

**Small batch:**
- Noisy gradient (regularization 효과)
- 메모리 효율적
- 느린 학습

### 4. Mixed Precision Training
FP16 사용으로 속도 향상

```python
# Scaling to prevent underflow
loss = loss * scale_factor
loss.backward()
gradients = gradients / scale_factor
optimizer.step()
```

## Common Optimization Patterns

### 1. Transformer Training Recipe
```python
# Typical settings for Transformers
optimizer = AdamW(lr=1e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(T_max=num_epochs)
gradient_clip = 1.0
warmup_steps = 1000
```

### 2. Fine-tuning Recipe
```python
# Lower learning rate for pre-trained models
optimizer = AdamW(lr=2e-5)
# Linear decay
# Few epochs (3-5)
```

### 3. Small Data Recipe
```python
# Strong regularization
dropout = 0.5
weight_decay = 0.1
# Data augmentation
# Early stopping
```