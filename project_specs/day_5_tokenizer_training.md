# Day 5: Tokenizer and Training

## 프로젝트 개요
토크나이저 구현과 학습 파이프라인을 완성합니다. Character-level과 BPE 토크나이저를 구현하고, 학습 루프, 최적화, 텍스트 생성 전략을 실습합니다.

## 학습 목표
- Tokenization 개념과 구현 이해
- Training loop 구성 요소 학습
- Loss functions와 Optimizers 구현
- 다양한 Text generation 전략 비교
- 완전한 학습 파이프라인 구축

## 구현 요구사항

### 1. Tokenizers
```python
class CharacterTokenizer:
    def fit(self, text)
    def encode(self, text) -> List[int]
    def decode(self, ids) -> str

class SimpleBPETokenizer:
    def fit(self, text, num_merges)
    def encode(self, text) -> List[int]
    def decode(self, ids) -> str
```
- Special tokens 관리 (<PAD>, <UNK>, <BOS>, <EOS>)
- Vocabulary 구축 및 매핑
- BPE merge 학습 알고리즘

### 2. DataLoader
```python
class DataLoader:
    def __init__(self, text, tokenizer, batch_size, seq_length)
    def get_batch(batch_idx) -> Tuple[inputs, targets]
    def __iter__()
```
- 배치 데이터 준비
- Input-target pair 생성 (shifted by 1)
- 효율적인 데이터 반복

### 3. Loss Functions
```python
def cross_entropy_loss(logits, targets, ignore_index=-100)
def perplexity(loss)
```
- Language modeling을 위한 cross-entropy
- Perplexity 계산
- Padding 처리

### 4. Optimizers
```python
class SGD:
    def __init__(self, parameters, lr, momentum=0)
    def step()
    def zero_grad()

class Adam:
    def __init__(self, parameters, lr, beta1=0.9, beta2=0.999)
    def step()
    def zero_grad()
```
- Gradient descent 구현
- Momentum 지원
- Adam의 adaptive learning rate

### 5. Text Generation
```python
class TextGenerator:
    def generate(prompt, max_length, temperature, top_k, top_p)
```
- Greedy decoding
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Repetition penalty

## 테스트 요구사항

### 필수 테스트
1. **Tokenizer 테스트**
   - Encode/decode 정확성
   - Special token 처리
   - BPE merge 동작

2. **DataLoader 테스트**
   - Batch 생성 확인
   - Input-target alignment

3. **Loss 테스트**
   - Perfect/random prediction loss
   - Perplexity 계산

4. **Optimizer 테스트**
   - Parameter update 검증
   - Gradient reset

5. **Generation 테스트**
   - 각 sampling 전략 동작
   - Max length 제한

## 실습 예제

### Mini Language Model Training
```python
# 1. 데이터 준비
text = load_text()
tokenizer = CharacterTokenizer()
tokenizer.fit(text)

# 2. 모델 생성
model = MiniLanguageModel(vocab_size)

# 3. 학습
optimizer = Adam(model.parameters())
for epoch in range(epochs):
    for inputs, targets in dataloader:
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        loss.backward()
        optimizer.step()

# 4. 생성
generator = TextGenerator(model, tokenizer)
generated = generator.generate("The ", max_length=50)
```

## Study Notes 구성

### 01_tokenization_basics.md
- Tokenization levels (char, word, subword)
- BPE 알고리즘 상세
- Special tokens 역할
- Vocabulary 크기 선택

### 02_training_loop.md
- Training loop 구조
- Batch processing
- Teacher forcing
- Gradient accumulation
- Learning rate scheduling

### 03_loss_and_optimization.md
- Cross-entropy loss 이해
- Perplexity 의미
- SGD vs Adam
- Gradient clipping
- Regularization

### 04_text_generation.md
- Decoding strategies 비교
- Temperature의 역할
- Top-k vs Top-p
- Beam search
- Controllable generation

## Notebook Tutorial 포함 내용
1. Character tokenizer 구현 및 테스트
2. BPE tokenizer 구현 및 비교
3. DataLoader 사용법
4. Loss 계산 및 시각화
5. Optimizer 비교 (SGD vs Adam)
6. Generation 전략별 결과 비교
7. Mini LM 학습 전체 과정

## 평가 기준
- [ ] 모든 tokenizer 기능 구현
- [ ] DataLoader 정상 동작
- [ ] Loss/Optimizer 구현 완료
- [ ] 모든 generation 전략 구현
- [ ] 테스트 100% 통과
- [ ] Mini LM 학습 성공
- [ ] Study notes 완성
- [ ] Notebook 실행 가능

## 실행 방법
```bash
cd projects/day5_tokenizer_training

# 테스트 실행
make test

# Mini LM 학습
make train

# 텍스트 생성 데모
make generate

# Notebook 실행
make notebook
```

## 다음 단계 (Day 6+)
- Pre-training strategies (MLM, CLM)
- Fine-tuning techniques
- Advanced tokenization (SentencePiece)
- Distributed training
- Model evaluation metrics