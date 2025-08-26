# Text Generation Strategies

## Overview
텍스트 생성은 학습된 언어 모델을 사용하여 새로운 텍스트를 만드는 과정입니다.
품질과 다양성 사이의 균형이 중요합니다.

## Decoding Strategies

### 1. Greedy Decoding
가장 확률이 높은 토큰을 항상 선택

```python
def greedy_decode(model, input_ids, max_length):
    generated = input_ids
    
    for _ in range(max_length):
        logits = model(generated)
        next_token = argmax(logits[-1])
        generated.append(next_token)
        
        if next_token == EOS_TOKEN:
            break
    
    return generated
```

**장점:**
- 빠르고 결정적
- 구현이 간단

**단점:**
- 반복적이고 지루한 텍스트
- Local optima에 빠짐
- 다양성 부족

### 2. Beam Search
상위 k개의 가능성을 동시에 탐색

```python
def beam_search(model, input_ids, beam_size=5):
    beams = [(input_ids, 0)]  # (sequence, score)
    
    for _ in range(max_length):
        candidates = []
        
        for seq, score in beams:
            logits = model(seq)
            probs = softmax(logits[-1])
            
            # Get top k tokens
            top_k = top_k_indices(probs, beam_size)
            
            for token in top_k:
                new_seq = seq + [token]
                new_score = score + log(probs[token])
                candidates.append((new_seq, new_score))
        
        # Keep top beam_size candidates
        beams = sorted(candidates, key=lambda x: x[1])[-beam_size:]
    
    return beams[0][0]  # Best sequence
```

**Beam size 효과:**
- beam_size=1: Greedy decoding
- beam_size=5-10: 일반적인 선택
- beam_size>10: 개선 미미, 계산 비용 증가

### 3. Temperature Sampling
확률 분포의 sharpness 조절

```python
def temperature_sampling(logits, temperature=1.0):
    # Apply temperature
    logits = logits / temperature
    
    # Convert to probabilities
    probs = softmax(logits)
    
    # Sample
    next_token = np.random.choice(vocab_size, p=probs)
    return next_token
```

**Temperature 효과:**
- T < 1.0: 더 확실한 선택 (less random)
- T = 1.0: 원래 분포
- T > 1.0: 더 다양한 선택 (more random)
- T → 0: Greedy decoding
- T → ∞: Uniform random

### 4. Top-k Sampling
상위 k개 토큰 중에서만 샘플링

```python
def top_k_sampling(logits, k=50, temperature=1.0):
    # Get top k tokens
    top_k_indices = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_indices]
    
    # Apply temperature and softmax
    top_k_probs = softmax(top_k_logits / temperature)
    
    # Sample from top k
    idx = np.random.choice(k, p=top_k_probs)
    return top_k_indices[idx]
```

**k 선택:**
- k=1: Greedy
- k=10: Conservative
- k=50: Balanced
- k=100: Creative

### 5. Top-p (Nucleus) Sampling
누적 확률 p까지의 토큰 중 샘플링

```python
def top_p_sampling(logits, p=0.9, temperature=1.0):
    # Sort by probability
    sorted_indices = np.argsort(logits)[::-1]
    sorted_probs = softmax(logits[sorted_indices] / temperature)
    
    # Find cutoff
    cumsum = np.cumsum(sorted_probs)
    cutoff_idx = np.argmax(cumsum >= p) + 1
    
    # Sample from nucleus
    nucleus_probs = sorted_probs[:cutoff_idx]
    nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
    
    idx = np.random.choice(cutoff_idx, p=nucleus_probs)
    return sorted_indices[idx]
```

**p 선택:**
- p=0.9: 상위 90% 확률 mass
- p=0.95: 더 다양한 선택
- p=0.8: 더 보수적

### 6. Combined Strategies
여러 방법을 조합

```python
def combined_sampling(logits, temperature=0.8, top_k=50, top_p=0.9):
    # Apply temperature
    logits = logits / temperature
    
    # Apply top-k filtering
    if top_k > 0:
        top_k_indices = np.argsort(logits)[-top_k:]
        mask = np.zeros_like(logits) - np.inf
        mask[top_k_indices] = logits[top_k_indices]
        logits = mask
    
    # Apply top-p filtering
    if top_p < 1.0:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_probs = softmax(logits[sorted_indices])
        cumsum = np.cumsum(sorted_probs)
        cutoff_idx = np.argmax(cumsum >= top_p) + 1
        
        mask = np.zeros_like(logits) - np.inf
        mask[sorted_indices[:cutoff_idx]] = logits[sorted_indices[:cutoff_idx]]
        logits = mask
    
    # Sample
    probs = softmax(logits)
    return np.random.choice(len(probs), p=probs)
```

## Advanced Techniques

### 1. Repetition Penalty
반복을 줄이기 위한 페널티

```python
def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    for token_id in set(generated_ids):
        logits[token_id] /= penalty
    return logits
```

### 2. N-gram Blocking
n-gram 반복 방지

```python
def block_ngrams(logits, generated, n=3):
    # Get last n-1 tokens
    context = tuple(generated[-(n-1):])
    
    # Find previous occurrences
    for i in range(len(generated) - n + 1):
        if tuple(generated[i:i+n-1]) == context:
            # Block the next token
            next_token = generated[i+n-1]
            logits[next_token] = -float('inf')
    
    return logits
```

### 3. Length Penalty
길이에 따른 스코어 조정

```python
def length_penalty(score, length, alpha=0.6):
    return score / (length ** alpha)
```

### 4. Diverse Beam Search
다양한 결과를 위한 beam search 변형

```python
def diverse_beam_search(model, input_ids, num_groups=5, diversity_penalty=0.5):
    # Split beams into groups
    # Apply diversity penalty between groups
    # Encourage different paths
```

## Controllable Generation

### 1. Prompt Engineering
시작 텍스트로 생성 제어

```python
prompts = {
    'formal': "In a professional manner, ",
    'casual': "Hey there! So basically ",
    'academic': "According to recent studies, "
}
```

### 2. Conditional Generation
특정 조건 하에서 생성

```python
def conditional_generate(model, condition, max_length):
    # Encode condition
    condition_emb = encode_condition(condition)
    
    # Generate with condition
    generated = []
    for _ in range(max_length):
        logits = model(generated, condition_emb)
        next_token = sample(logits)
        generated.append(next_token)
    
    return generated
```

### 3. Attribute Control
스타일, 감정 등 제어

```python
attributes = {
    'sentiment': 'positive',
    'style': 'formal',
    'topic': 'technology'
}
```

## Evaluation Metrics

### 1. Perplexity
모델의 불확실성 측정

```python
perplexity = exp(cross_entropy_loss)
```

### 2. BLEU Score
생성 텍스트와 참조 텍스트 비교

```python
def bleu_score(generated, reference):
    # N-gram precision
    # Brevity penalty
```

### 3. Diversity Metrics
- **Distinct-n**: Unique n-grams 비율
- **Self-BLEU**: 생성 텍스트들 간 유사도
- **Entropy**: 토큰 분포의 엔트로피

### 4. Human Evaluation
- Fluency: 문법적 정확성
- Coherence: 논리적 일관성
- Relevance: 주제 관련성
- Creativity: 창의성

## Common Issues & Solutions

### 1. Repetitive Text
**문제:** 같은 구문 반복
**해결:**
- Repetition penalty
- N-gram blocking
- Higher temperature
- Top-p sampling

### 2. Incoherent Text
**문제:** 논리적 일관성 부족
**해결:**
- Lower temperature
- Beam search
- Better prompting
- Longer context

### 3. Generic Text
**문제:** 너무 일반적인 응답
**해결:**
- Top-k/Top-p sampling
- Temperature tuning
- Diverse beam search
- Better training data

### 4. Cut-off Text
**문제:** 문장이 중간에 끊김
**해결:**
- EOS token handling
- Length penalty
- Minimum length constraint

## Best Practices

1. **Start Conservative**
   - Temperature: 0.7-0.8
   - Top-p: 0.9
   - Top-k: 50

2. **Task-Specific Tuning**
   - Creative writing: Higher temperature
   - Factual: Lower temperature, beam search
   - Dialogue: Top-p sampling

3. **Post-Processing**
   - Remove incomplete sentences
   - Fix capitalization
   - Handle special tokens

4. **Caching**
   ```python
   # Cache previous computations
   past_key_values = None
   for token in generate():
       logits, past_key_values = model(token, past_key_values)
   ```