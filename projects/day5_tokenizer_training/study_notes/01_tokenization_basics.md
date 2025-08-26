# Tokenization Basics

## What is Tokenization?
토큰화는 텍스트를 모델이 처리할 수 있는 작은 단위(토큰)로 나누는 과정입니다.
LLM은 텍스트를 직접 이해할 수 없으므로, 숫자로 변환해야 합니다.

## Tokenization Levels

### 1. Character-level Tokenization
가장 간단한 방법 - 각 문자를 하나의 토큰으로 처리

**장점:**
- 구현이 매우 간단
- OOV(Out-of-Vocabulary) 문제 없음
- 작은 vocabulary size

**단점:**
- 긴 시퀀스 생성 (많은 토큰)
- 의미 단위를 포착하기 어려움
- 학습이 더 어려울 수 있음

```python
# Example
"hello" → ['h', 'e', 'l', 'l', 'o']
Token IDs: [7, 4, 11, 11, 14]
```

### 2. Word-level Tokenization
공백이나 구두점으로 단어를 구분

**장점:**
- 직관적이고 이해하기 쉬움
- 의미 단위 보존

**단점:**
- 큰 vocabulary size
- OOV 문제 심각
- 형태 변화 처리 어려움 (run, running, ran)

```python
# Example
"Hello world!" → ['Hello', 'world', '!']
Token IDs: [1532, 876, 23]
```

### 3. Subword Tokenization
단어를 더 작은 의미 단위로 분할

**방법들:**
- BPE (Byte Pair Encoding)
- WordPiece (BERT)
- SentencePiece (T5, LLaMA)
- Unigram Language Model

**장점:**
- OOV 문제 해결
- 적절한 vocabulary size
- 형태소 수준의 이해 가능

**단점:**
- 구현이 복잡
- 학습 시간 필요

## Byte Pair Encoding (BPE)

### 알고리즘
1. 텍스트를 문자 단위로 시작
2. 가장 빈번한 문자 쌍을 찾음
3. 해당 쌍을 하나의 토큰으로 병합
4. 원하는 vocabulary size까지 반복

### 예제
```
초기: ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']

Step 1: 'll' 가장 빈번 → 병합
['h', 'e', 'll', 'o', ' ', 'w', 'o', 'r', 'l', 'd']

Step 2: 'lo' 빈번 → 병합  
['h', 'e', 'llo', ' ', 'w', 'o', 'r', 'l', 'd']

... 계속 ...
```

## Special Tokens

### 필수 Special Tokens
```python
<PAD>  # Padding - 시퀀스 길이 맞추기
<UNK>  # Unknown - vocabulary에 없는 토큰
<BOS>  # Beginning of Sequence - 시작 표시
<EOS>  # End of Sequence - 끝 표시
```

### 추가 Special Tokens (task별)
```python
<MASK>  # Masked Language Modeling (BERT)
<SEP>   # Segment separator
<CLS>   # Classification token
```

## Vocabulary Management

### Token ↔ ID Mapping
```python
vocab = {
    '<PAD>': 0,
    '<UNK>': 1,
    '<BOS>': 2,
    '<EOS>': 3,
    'the': 4,
    'a': 5,
    ...
}

id_to_token = {v: k for k, v in vocab.items()}
```

### Vocabulary Size 선택
- **작은 vocabulary (< 10K)**
  - 빠른 학습
  - 작은 메모리
  - 긴 시퀀스

- **큰 vocabulary (> 50K)**
  - 짧은 시퀀스
  - 더 나은 표현력
  - 큰 메모리 필요

## 실제 사용 예제

### GPT 시리즈
- GPT-2: BPE with 50,257 tokens
- GPT-3: Similar BPE approach
- Merges learned from training data

### BERT
- WordPiece with 30,522 tokens
- Includes [CLS], [SEP], [MASK] tokens

### Modern LLMs
- LLaMA: SentencePiece with 32K tokens
- GPT-4: Likely uses enhanced BPE (~100K tokens)

## Tokenization 성능 측정

### Fertility (Compression Ratio)
```
Fertility = (Number of tokens) / (Number of words)
```
낮을수록 좋음 (더 압축됨)

### Coverage
얼마나 많은 텍스트를 <UNK> 없이 표현할 수 있는가

### Example
```python
text = "The quick brown fox"
char_tokens = 19  # 공백 포함
word_tokens = 4
bpe_tokens = 6  # 예시

char_fertility = 19/4 = 4.75
word_fertility = 4/4 = 1.0  
bpe_fertility = 6/4 = 1.5
```

## 구현 시 주의사항

1. **정규화 (Normalization)**
   - Lowercase 변환
   - Unicode 정규화
   - 특수문자 처리

2. **Pre-tokenization**
   - 공백으로 먼저 분할
   - 언어별 규칙 적용

3. **Post-processing**
   - Special token 추가
   - Truncation/Padding

4. **효율성**
   - Trie 자료구조 사용
   - 캐싱 활용
   - 병렬 처리