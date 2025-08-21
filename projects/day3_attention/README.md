# 🎯 Day 3: Attention Mechanism - LLM의 심장

> "Attention is All You Need" - Transformer 논문의 핵심 메시지

## 🎯 학습 목표

드디어 LLM의 핵심 기술인 Attention 메커니즘을 구현합니다!
이것이 바로 GPT, BERT, LLaMA 등 모든 현대 LLM의 기초입니다.

### 핵심 개념
- ✅ Self-Attention의 원리 이해
- ✅ Query, Key, Value 행렬의 역할
- ✅ Scaled Dot-Product Attention 구현
- ✅ Multi-Head Attention으로 확장
- ✅ Positional Encoding 추가

## 📚 프로젝트 구조

```
day3_attention/
├── README.md                    # 프로젝트 개요
├── requirements.txt            # 의존성
├── Makefile                   # 빌드/테스트 자동화
│
├── study_notes/               # 📖 학습 자료
│   ├── 01_attention_intuition.md   # Attention 직관적 이해
│   ├── 02_self_attention.md        # Self-Attention 메커니즘
│   ├── 03_multi_head.md           # Multi-Head Attention
│   └── 04_positional_encoding.md   # 위치 인코딩
│
├── notebooks/                 # 🔬 실습 노트북
│   └── attention_tutorial.ipynb
│
├── tests/                     # 🧪 테스트
│   └── test_attention.py
│
└── 50_eval/                   # 🎯 평가/데모
    ├── attention_visualizer.py    # Attention 시각화
    └── translation_toy.py         # 간단한 번역 태스크
```

## 🚀 시작하기

### 1. 환경 설정
```bash
cd projects/day3_attention
pip install -r requirements.txt
```

### 2. 학습 순서
1. **이론 학습**: `study_notes/` 순서대로 읽기
2. **실습**: `notebooks/attention_tutorial.ipynb` 따라하기
3. **구현**: `core/attention.py` 완성
4. **테스트**: `make test`로 검증
5. **시각화**: `50_eval/attention_visualizer.py` 실행

### 3. 테스트 실행
```bash
make test  # 모든 테스트 실행
make visualize  # Attention 시각화
```

## 🎓 학습 내용

### 1. Attention의 직관
- 문장의 각 단어가 다른 단어들에 얼마나 "주목"하는지
- "The cat sat on the mat" - 'cat'은 'sat'과 'the'에 주목
- 문맥을 이해하는 메커니즘

### 2. Self-Attention 수식
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V

Q: Query - 무엇을 찾고 있는가?
K: Key - 무엇을 제공할 수 있는가?
V: Value - 실제 정보
```

### 3. Multi-Head Attention
```python
# 여러 개의 Attention을 병렬로
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 4. Positional Encoding
```python
# 위치 정보를 sine/cosine으로 인코딩
PE(pos, 2i) = sin(pos/10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
```

## 💻 구현할 주요 함수

### core/attention.py
```python
# 구현할 함수들
- scaled_dot_product_attention()  # 기본 attention
- MultiHeadAttention class        # Multi-head 구현
- positional_encoding()           # 위치 인코딩
- causal_mask()                  # 미래 마스킹 (GPT용)
```

## 🔥 도전 과제

### 기본 과제
1. ✅ Self-Attention 구현 및 테스트
2. ✅ Multi-Head Attention 완성
3. ✅ Attention 가중치 시각화
4. ✅ 간단한 시퀀스-투-시퀀스 태스크

### 심화 과제
1. 🌟 Relative Positional Encoding
2. 🌟 Flash Attention (효율적 구현)
3. 🌟 Cross-Attention 구현
4. 🌟 Attention with Linear Complexity

## 📊 이해도 체크

### 개념 이해
- [ ] Q, K, V의 역할을 설명할 수 있다
- [ ] Scaling factor (√d_k)가 필요한 이유를 안다
- [ ] Multi-Head가 Single-Head보다 좋은 이유를 안다
- [ ] Positional Encoding이 필요한 이유를 안다

### 구현 능력
- [ ] NumPy로 Self-Attention을 구현할 수 있다
- [ ] Attention 가중치를 시각화할 수 있다
- [ ] Causal Mask를 적용할 수 있다
- [ ] Multi-Head로 확장할 수 있다

## 🔍 주요 인사이트

1. **병렬 처리**: RNN과 달리 모든 위치를 동시에 처리
2. **장거리 의존성**: 거리와 무관하게 정보 전달
3. **해석 가능성**: Attention 가중치로 모델 동작 이해
4. **계산 효율성**: 행렬 연산으로 GPU 최적화

## 📝 체크리스트

- [ ] Self-Attention 이해 및 구현
- [ ] Multi-Head Attention 구현
- [ ] Positional Encoding 적용
- [ ] Attention 가중치 시각화
- [ ] 테스트 모두 통과
- [ ] 번역 태스크 실행

## 🔗 참고 자료

- [Attention Is All You Need (원 논문)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)
- [Visualizing Attention](https://distill.pub/2016/augmented-rnns/)

## 💡 다음 단계

Day 3를 완료하면 Day 4 (Transformer Block)로 진행합니다.
여러 Attention 레이어를 쌓아 완전한 Transformer를 만듭니다!

---

**"Attention is literally all you need to understand modern AI!"**