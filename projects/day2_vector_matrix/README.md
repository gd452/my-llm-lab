# 🔢 Day 2: Vector & Matrix Operations

> "효율적인 벡터 연산은 딥러닝의 기초입니다"

## 🎯 학습 목표

이제까지 스칼라 단위로 처리하던 연산을 벡터/행렬로 확장합니다.
이를 통해 더 빠르고 효율적인 신경망을 구현할 수 있습니다.

### 핵심 개념
- ✅ NumPy 배열과 Broadcasting
- ✅ Batch 처리와 병렬 연산
- ✅ Softmax와 CrossEntropy Loss
- ✅ 행렬곱과 효율적인 연산

## 📚 프로젝트 구조

```
day2_vector_matrix/
├── README.md                 # 프로젝트 개요
├── requirements.txt          # 의존성
├── Makefile                 # 빌드/테스트 자동화
│
├── study_notes/             # 📖 학습 자료
│   ├── 01_numpy_basics.md  # NumPy 기초
│   ├── 02_broadcasting.md  # Broadcasting 이해
│   ├── 03_batch_processing.md  # Batch 처리
│   └── 04_loss_functions.md    # 손실 함수들
│
├── notebooks/               # 🔬 실습 노트북
│   └── vector_ops_tutorial.ipynb
│
├── tests/                   # 🧪 테스트
│   └── test_tensor_ops.py
│
└── 50_eval/                 # 🎯 평가/데모
    ├── mnist_mini.py        # Mini MNIST 분류
    └── benchmark.py         # 성능 비교
```

## 🚀 시작하기

### 1. 환경 설정
```bash
cd projects/day2_vector_matrix
pip install -r requirements.txt
```

### 2. 학습 순서
1. **개념 학습**: `study_notes/` 순서대로 읽기
2. **실습**: `notebooks/vector_ops_tutorial.ipynb` 따라하기
3. **구현**: core 모듈의 tensor_ops.py 완성
4. **테스트**: `make test`로 검증
5. **평가**: `50_eval/` 데모 실행

### 3. 테스트 실행
```bash
make test  # 모든 테스트 실행
```

## 🎓 학습 내용

### 1. NumPy 기초
- ndarray 생성과 조작
- shape, dtype, stride
- 기본 연산과 인덱싱

### 2. Broadcasting
- Broadcasting 규칙
- 효율적인 연산 패턴
- 메모리 최적화

### 3. Batch Processing
- Mini-batch 구현
- 병렬 연산의 이점
- GPU와의 연관성

### 4. 핵심 함수 구현
- Softmax (안정적인 구현)
- Cross-Entropy Loss
- Batch Normalization
- Dropout (optional)

## 💻 구현할 주요 함수

### core/tensor_ops.py
```python
# 구현할 함수들
- matmul(a, b)           # 행렬곱
- softmax(x, axis=-1)    # Softmax 활성화
- cross_entropy(y_pred, y_true)  # 손실 함수
- batch_norm(x, gamma, beta)     # 배치 정규화
```

### core/nn_vectorized.py
```python
# 벡터화된 신경망 레이어
- LinearLayer        # 완전 연결층 (벡터화)
- MLPVectorized     # 벡터화된 MLP
```

## 🔥 도전 과제

### 기본 과제
1. ✅ 벡터화된 XOR 문제 해결
2. ✅ Mini MNIST (숫자 0-4) 분류
3. ✅ 스칼라 vs 벡터 연산 속도 비교

### 심화 과제
1. 🌟 Batch Normalization 구현
2. 🌟 Dropout 구현
3. 🌟 Adam Optimizer 구현
4. 🌟 Learning Rate Scheduling

## 📊 성능 목표

### Mini MNIST (0-4)
- 정확도: > 90%
- 학습 시간: < 30초
- 배치 크기: 32

### 속도 향상
- 벡터 연산이 스칼라 연산보다 10배 이상 빠름
- 배치 처리로 처리량 증가

## 🔍 주요 인사이트

1. **Broadcasting의 힘**: 명시적 루프 없이 연산 가능
2. **Batch의 중요성**: 병렬 처리와 안정적 학습
3. **수치 안정성**: log-sum-exp 트릭 등
4. **메모리 효율성**: view vs copy 이해

## 📝 체크리스트

- [ ] NumPy 기초 이해
- [ ] Broadcasting 규칙 숙지
- [ ] Softmax 안정적 구현
- [ ] Cross-Entropy 구현
- [ ] 벡터화된 신경망 구현
- [ ] Mini MNIST 학습
- [ ] 성능 벤치마크

## 🔗 참고 자료

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/)

## 💡 다음 단계

Day 2를 완료하면 Day 3 (Attention Mechanism)으로 진행합니다.
Attention 메커니즘은 현대 LLM의 핵심입니다!

---

**"벡터화는 딥러닝의 첫 번째 최적화입니다!"**