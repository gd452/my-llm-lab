# Day 3: Attention Mechanism 프로젝트 스펙

## 🎯 프로젝트 목표
LLM의 핵심인 Attention 메커니즘을 완전히 이해하고 구현합니다.

## 📚 학습 목표
1. Attention의 직관적 이해와 수학적 원리
2. Self-Attention과 Cross-Attention 구현
3. Multi-Head Attention의 필요성과 구현
4. Positional Encoding의 역할과 구현
5. Transformer의 기초 이해

## 🏗️ 프로젝트 구조

```
projects/day3_attention/
├── README.md                       # 프로젝트 개요
├── requirements.txt               # 의존성 패키지
├── Makefile                      # 빌드 자동화
│
├── study_notes/                  # 📖 학습 자료
│   ├── 01_attention_intuition.md    # Attention 직관적 이해
│   ├── 02_self_attention.md         # Self-Attention 메커니즘
│   ├── 03_multi_head.md            # Multi-Head Attention
│   └── 04_positional_encoding.md    # 위치 인코딩
│
├── notebooks/                    # 🔬 실습 노트북
│   └── attention_tutorial.ipynb
│
├── tests/                        # 🧪 테스트
│   └── test_attention.py
│
└── 50_eval/                      # 🎯 평가/데모
    ├── attention_visualizer.py  # Attention 시각화
    └── translation_toy.py       # 간단한 번역 태스크

core/                            # 핵심 구현 (최상위)
└── attention.py                 # Attention 메커니즘 구현
```

## 📋 구현 체크리스트

### Core 모듈 (`core/attention.py`)
- [x] `scaled_dot_product_attention()`: 기본 attention
- [x] `MultiHeadAttention` class: Multi-head 구현
  - [x] Q, K, V projection
  - [x] Head splitting/combining
  - [x] Output projection
- [x] `positional_encoding()`: Sinusoidal PE
- [x] `add_positional_encoding()`: PE 추가
- [x] `create_causal_mask()`: GPT 스타일 마스킹
- [x] `create_padding_mask()`: 패딩 마스킹
- [x] `visualize_attention()`: 가중치 시각화

### 테스트 (`tests/test_attention.py`)
- [x] Scaled dot-product attention 테스트
- [x] Multi-head attention 테스트
- [x] Positional encoding 테스트
- [x] Masking 함수 테스트
- [x] Attention 속성 검증

### 평가 (`50_eval/`)
- [x] Attention 가중치 시각화
  - [x] Self-attention 패턴
  - [x] Causal attention 패턴
  - [x] Multi-head 비교
- [x] 간단한 번역 태스크
  - [x] Encoder-Decoder attention
  - [x] Cross-attention 시각화

## 🎓 핵심 개념

### 1. Attention 수식
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V

Q: Query - 무엇을 찾고 있는가?
K: Key - 무엇을 제공할 수 있는가?
V: Value - 실제 정보
```

### 2. Multi-Head Attention
```python
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 3. Positional Encoding
```python
PE(pos, 2i) = sin(pos/10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
```

## 🔥 도전 과제

### 필수 과제
1. ✅ Self-Attention 구현 및 시각화
2. ✅ Multi-Head Attention 구현
3. ✅ Causal masking 적용
4. ✅ Positional encoding 추가

### 선택 과제
1. ⬜ Relative positional encoding
2. ⬜ Efficient attention (Linear complexity)
3. ⬜ Flash Attention 이해
4. ⬜ RoPE (Rotary PE) 구현

## 📊 이해도 체크리스트

### 개념 이해
- [ ] Q, K, V의 역할을 명확히 설명할 수 있다
- [ ] Scaling factor (√d_k)가 필요한 이유를 안다
- [ ] Multi-Head가 Single-Head보다 좋은 이유를 안다
- [ ] Positional Encoding이 필요한 이유를 안다
- [ ] Self-Attention vs Cross-Attention 차이를 안다

### 구현 능력
- [ ] NumPy로 Attention을 구현할 수 있다
- [ ] Attention weights를 시각화할 수 있다
- [ ] Causal mask를 적용할 수 있다
- [ ] Multi-Head로 확장할 수 있다
- [ ] PE를 추가하고 효과를 확인할 수 있다

## 🔍 주요 인사이트

### Attention의 장점
1. **병렬 처리**: RNN과 달리 모든 위치 동시 처리
2. **장거리 의존성**: 거리와 무관하게 정보 전달
3. **해석 가능성**: Attention 가중치로 모델 이해
4. **효율성**: 행렬 연산으로 GPU 최적화

### Multi-Head의 이점
1. **다양성**: 각 head가 다른 패턴 학습
2. **앙상블 효과**: 여러 관점 동시 고려
3. **표현력**: 더 풍부한 특징 추출

### Positional Encoding
1. **순서 정보**: Attention은 순서를 모름
2. **Sinusoidal**: 학습 불필요, 임의 길이 처리
3. **최신 기법**: RoPE, ALiBi 등

## 💡 구현 팁

1. **수치 안정성**
   ```python
   # Softmax 전 최댓값 빼기
   scores = scores - np.max(scores, axis=-1, keepdims=True)
   ```

2. **효율적인 행렬 연산**
   ```python
   # 배치 처리
   # (batch, seq, d) @ (batch, d, seq) = (batch, seq, seq)
   ```

3. **Shape 디버깅**
   ```python
   print(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
   print(f"Attention: {attention.shape}")
   ```

## 🐛 일반적인 실수

1. **Scaling 빼먹기**: √d_k로 나누지 않으면 gradient 문제
2. **차원 혼동**: seq_len vs d_model 구분
3. **Mask 적용 오류**: -inf 대신 큰 음수 사용

## 📚 참고 자료

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)
- [Visualizing Attention](https://distill.pub/2016/augmented-rnns/)

## ✅ 완료 기준

- [ ] 모든 테스트 통과 (`make test`)
- [ ] Attention 시각화 실행 (`make visualize`)
- [ ] 번역 데모 실행 (`make translate`)
- [ ] 학습 노트 완독 및 이해
- [ ] 노트북 튜토리얼 완료

## 🎯 다음 단계

Day 3를 완료하면 Day 4 (Transformer Block)로 진행합니다.
여러 Attention 레이어를 쌓아 완전한 Transformer를 구현합니다!