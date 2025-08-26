# Training Loop Implementation

## Training Loop의 구조

### 기본 Training Loop
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # 2. Backward pass
        loss.backward()
        
        # 3. Optimizer step
        optimizer.step()
        optimizer.zero_grad()
```

## 핵심 구성 요소

### 1. Data Loading
```python
class DataLoader:
    def __init__(self, data, batch_size, seq_length):
        self.data = data
        self.batch_size = batch_size
        self.seq_length = seq_length
    
    def get_batch(self):
        # Random sampling
        # Sequential sampling
        # Shuffled sampling
```

**Batching Strategies:**
- **Random sampling**: 무작위로 시퀀스 선택
- **Sequential**: 순차적으로 진행
- **Shuffled**: Epoch마다 섞기

### 2. Forward Pass
모델에 입력을 전달하고 출력 계산

```python
# Language Modeling Task
inputs = [batch_size, seq_length]
outputs = model(inputs)  # [batch_size, seq_length, vocab_size]
```

### 3. Loss Calculation
예측과 정답 사이의 차이 계산

```python
# Cross-Entropy Loss for LM
loss = cross_entropy(outputs, targets)
```

### 4. Backward Pass
Gradient 계산 (Backpropagation)

```python
loss.backward()  # Compute gradients
```

### 5. Parameter Update
Optimizer를 통한 가중치 업데이트

```python
optimizer.step()      # Update weights
optimizer.zero_grad()  # Reset gradients
```

## Language Model Training

### Autoregressive Training
다음 토큰 예측 학습

```python
# Input:  "The cat sat on"
# Target: "cat sat on the"

for i in range(seq_length - 1):
    input_token = tokens[i]
    target_token = tokens[i + 1]
    # Predict target from input
```

### Teacher Forcing
학습 시 실제 정답을 다음 입력으로 사용

```python
# Training (Teacher Forcing)
for t in range(seq_length):
    output = model(target[t])  # Use ground truth
    loss += loss_fn(output, target[t+1])

# Inference (No Teacher Forcing)
for t in range(max_length):
    output = model(generated[t])  # Use generated
    next_token = sample(output)
    generated.append(next_token)
```

## Training Techniques

### 1. Gradient Accumulation
메모리 제약 시 여러 미니배치의 gradient 누적

```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. Gradient Clipping
Exploding gradient 방지

```python
def clip_gradients(parameters, max_norm=1.0):
    total_norm = 0
    for p in parameters:
        total_norm += p.grad.norm() ** 2
    total_norm = sqrt(total_norm)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad *= clip_coef
```

### 3. Learning Rate Scheduling

**Linear Warmup**
```python
def warmup_schedule(step, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    return base_lr
```

**Cosine Annealing**
```python
def cosine_schedule(step, total_steps, base_lr, min_lr):
    progress = step / total_steps
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
```

### 4. Mixed Precision Training
계산 속도 향상을 위한 FP16 사용

```python
# Conceptual (실제는 더 복잡)
with autocast():
    output = model(input)
    loss = loss_fn(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Monitoring & Logging

### 필수 Metrics
```python
metrics = {
    'loss': running_loss / num_batches,
    'perplexity': exp(loss),
    'learning_rate': optimizer.param_groups[0]['lr'],
    'gradient_norm': compute_grad_norm(model),
}
```

### Validation Loop
```python
def validate(model, val_dataloader):
    model.eval()  # Evaluation mode
    total_loss = 0
    
    with torch.no_grad():  # No gradient computation
        for batch in val_dataloader:
            output = model(batch)
            loss = loss_fn(output, targets)
            total_loss += loss
    
    model.train()  # Back to training mode
    return total_loss / len(val_dataloader)
```

### Early Stopping
```python
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
    
    def should_stop(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience
```

## Memory Management

### Batch Size vs Sequence Length Trade-off
```
Memory ∝ batch_size × seq_length × d_model²
```

### Gradient Checkpointing
메모리 절약을 위해 일부 activation만 저장

```python
# Forward에서 일부만 저장
# Backward에서 필요시 재계산
```

## Common Issues & Solutions

### 1. Loss Not Decreasing
- Learning rate 조정
- Gradient clipping 확인
- 데이터 문제 확인
- 모델 초기화 검토

### 2. Overfitting
- Dropout 추가
- Weight decay 적용
- 데이터 증강
- Early stopping

### 3. Training Instability
- Gradient clipping
- Learning rate warmup
- Batch normalization
- Smaller learning rate

### 4. Out of Memory
- Batch size 감소
- Sequence length 감소
- Gradient accumulation
- Model parallelism

## Best Practices

1. **Start Simple**
   - 작은 모델로 시작
   - 작은 데이터셋으로 테스트
   - 기본 하이퍼파라미터 사용

2. **Monitor Everything**
   - Loss curves
   - Gradient norms
   - Learning rate
   - Validation metrics

3. **Save Checkpoints**
   ```python
   if epoch % save_interval == 0:
       save_checkpoint(model, optimizer, epoch)
   ```

4. **Reproducibility**
   ```python
   # Set random seeds
   np.random.seed(42)
   random.seed(42)
   ```