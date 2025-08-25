# miniGPT - nanoGPT 스타일 구현

## 🎯 목표
**"GPT가 어떻게 텍스트를 생성하는가?"를 최소 코드로 이해**

## 📚 핵심 개념 설명

### 1. **Self-Attention: "단어들이 서로를 어떻게 바라보는가"**

```python
# Query: "나는 무엇을 찾고 있나?"
# Key: "나는 무엇을 제공하나?"  
# Value: "나는 무엇을 전달하나?"

attention_scores = Query @ Key.T  # 각 토큰이 다른 토큰을 얼마나 주목하는지
attention_weights = softmax(attention_scores)  # 확률로 변환
output = attention_weights @ Value  # 가중 평균으로 정보 집계
```

**왜 필요한가?**
- "The cat sat on the mat" 에서 "it"이 나오면 "cat"을 참조해야 함
- Attention은 이런 관계를 학습

### 2. **Multi-Head Attention: "여러 관점에서 보기"**

```python
# 6개의 head = 6가지 다른 관점
# Head 1: 문법적 관계 (주어-동사)
# Head 2: 의미적 관계 (cat-animal)
# Head 3: 위치적 관계 (가까운 단어)
# ...
```

**왜 필요한가?**
- 단일 attention은 한 가지 패턴만 학습
- 여러 head로 다양한 관계 동시 학습

### 3. **Positional Encoding: "순서 정보"**

```python
x = token_embedding + position_embedding
# "cat"의 임베딩 + "3번째 위치"의 임베딩
```

**왜 필요한가?**
- Transformer는 순서를 모름 (RNN과 달리)
- 위치 정보를 명시적으로 추가해야 함

### 4. **Causal Masking: "미래를 볼 수 없게"**

```python
# "The cat sat" 까지만 보고 다음 단어 예측
# "on the mat"는 마스킹으로 가림
mask = torch.tril(torch.ones(seq_len, seq_len))
```

**왜 필요한가?**
- 생성 시: 미래 정보 없이 다음 단어 예측
- 학습 시: 모든 위치에서 동시에 학습 (효율적)

### 5. **Residual Connection: "정보 손실 방지"**

```python
x = x + attention(x)  # 원본 정보 + 변환된 정보
```

**왜 필요한가?**
- 깊은 네트워크에서 gradient vanishing 방지
- 원본 정보 보존하며 점진적 개선

## 🔧 실행 방법

### 1. 학습
```bash
cd projects/mini_gpt
python train.py
```

**출력 예시:**
```
Step 0    | train loss 4.5234 | val loss 4.5123
Step 500  | train loss 2.4567 | val loss 2.4890
Step 1000 | train loss 1.8234 | val loss 1.9123
...
생성 샘플:
ROMEO:
What shall I do to thee with my love?
```

### 2. 텍스트 생성
```bash
python generate.py --prompt "To be or not to be"
```

## 📊 모델 구조

```
GPT(
  vocab_size=65,     # 문자 수
  n_embd=384,        # 임베딩 크기
  n_head=6,          # attention head 수
  n_layer=6,         # transformer block 수
  block_size=256     # 최대 시퀀스 길이
)
= 약 10M 파라미터
```

## 🧪 실험해볼 것들

### 1. **Attention 시각화**
```python
# attention_weights를 시각화하면 
# 모델이 어떤 단어를 주목하는지 볼 수 있음
```

### 2. **Temperature 조절**
```python
generate(temperature=0.5)  # 보수적 (반복적)
generate(temperature=1.0)  # 균형
generate(temperature=1.5)  # 창의적 (무작위)
```

### 3. **모델 크기 실험**
```python
# 작은 모델 (빠른 학습, 낮은 품질)
n_layer=2, n_head=2, n_embd=128

# 큰 모델 (느린 학습, 높은 품질)  
n_layer=12, n_head=8, n_embd=512
```

## 💡 핵심 인사이트

1. **GPT = Transformer Decoder만 사용**
   - Encoder 없음 (번역 불필요)
   - Causal masking으로 자기회귀 생성

2. **학습 = 다음 토큰 예측**
   - "The cat" → "sat" 예측
   - 모든 위치에서 동시 학습

3. **생성 = 학습한 패턴 재현**
   - 확률 분포에서 샘플링
   - Temperature로 다양성 조절

## 📈 학습 과정 이해

```
Loss 감소 과정:
4.5 → 2.5 → 1.5 → 1.0

의미:
- 4.5: 랜덤 (65개 중 하나 = log(65) ≈ 4.2)
- 2.5: 패턴 학습 시작
- 1.5: 문법 구조 이해
- 1.0: 스타일 모방 가능
```

## 🚀 다음 단계

1. **더 큰 데이터**: Wikipedia, BookCorpus
2. **Tokenizer 개선**: BPE, WordPiece
3. **모델 스케일**: GPT-2 (124M) → GPT-3 (175B)
4. **Fine-tuning**: 특정 태스크 학습

## ❓ 자주 묻는 질문

**Q: 왜 character-level tokenizer?**
A: 간단하고 이해하기 쉬움. 실제로는 BPE/WordPiece 사용

**Q: 왜 셰익스피어 텍스트?**
A: 작은 크기(1MB), 독특한 스타일로 학습 확인 용이

**Q: 얼마나 학습해야 하나?**
A: 5000 iter ≈ 30분 (CPU), 5분 (GPU)

**Q: 생성 품질이 낮은데?**
A: 정상. 10M 모델 + 1MB 데이터의 한계. 개념 이해가 목적