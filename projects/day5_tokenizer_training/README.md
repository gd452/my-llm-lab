# Day 5: Tokenizer and Training - Making LLM Work

## 학습 목표
- Tokenizer 구현 (Character-level, BPE basics)
- Training loop 구현
- Loss functions (Cross-Entropy)
- Optimizer basics (SGD, Adam)
- Text generation strategies

## 프로젝트 구조
```
day5_tokenizer_training/
├── README.md
├── notebooks/
│   └── tokenizer_training_tutorial.ipynb  # 전체 구현 튜토리얼
├── study_notes/
│   ├── 01_tokenization_basics.md         # 토큰화 개념과 방법
│   ├── 02_training_loop.md               # 학습 루프 구현
│   ├── 03_loss_and_optimization.md       # 손실 함수와 최적화
│   └── 04_text_generation.md             # 텍스트 생성 전략
├── tests/
│   └── test_tokenizer_training.py        # 테스트 코드
└── 50_eval/
    ├── train_mini_lm.py                   # 미니 언어 모델 학습
    └── generate_text.py                   # 텍스트 생성 데모
```

## 핵심 개념

### 1. Tokenization
- **Character-level**: 가장 간단한 토큰화
- **Word-level**: 단어 단위 토큰화
- **Subword (BPE)**: Byte Pair Encoding 기초
- **Vocabulary 관리**: token ↔ id 매핑

### 2. Training Components
```python
# Training loop 구조
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        logits = model(input_ids)
        loss = cross_entropy(logits, targets)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
```

### 3. Loss Functions
- Cross-Entropy Loss for language modeling
- Perplexity as evaluation metric
- Gradient clipping for stability

### 4. Text Generation
- **Greedy decoding**: 가장 확률 높은 토큰 선택
- **Temperature sampling**: 다양성 조절
- **Top-k sampling**: 상위 k개 중 선택
- **Top-p (nucleus) sampling**: 누적 확률 기반

## 실습 내용
1. Character-level tokenizer 구현
2. Simple BPE tokenizer 구현
3. Training loop with data loader
4. Cross-entropy loss 구현
5. SGD와 Adam optimizer 구현
6. Text generation with different strategies

## 학습 순서
1. `study_notes/01_tokenization_basics.md` 읽기
2. `notebooks/tokenizer_training_tutorial.ipynb` 실습
3. `core/tokenizer.py` 구현
4. `core/training.py` 구현
5. `tests/test_tokenizer_training.py` 실행
6. `50_eval/train_mini_lm.py`로 실제 학습
7. `50_eval/generate_text.py`로 생성 테스트

## 필수 구현 사항
- [ ] CharacterTokenizer class
- [ ] SimpleBPETokenizer class
- [ ] DataLoader class
- [ ] CrossEntropyLoss function
- [ ] SGDOptimizer class
- [ ] AdamOptimizer class
- [ ] TextGenerator class with sampling strategies

## 실제 학습 예제
```python
# 간단한 텍스트로 학습
text = "The quick brown fox jumps over the lazy dog"
tokenizer = CharacterTokenizer()
tokenizer.fit(text)

# 모델 학습
model = MiniLanguageModel(vocab_size=tokenizer.vocab_size)
train(model, text, tokenizer, epochs=100)

# 텍스트 생성
generated = generate(model, tokenizer, "The quick", max_length=20)
```

## 다음 단계 (Day 6+)
- Pre-training strategies (Masked LM, Causal LM)
- Fine-tuning techniques
- Advanced tokenization (SentencePiece, WordPiece)
- Distributed training basics
- 