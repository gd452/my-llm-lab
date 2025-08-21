# Day 2: Vector & Matrix Operations 프로젝트 스펙

## 🎯 프로젝트 목표
스칼라 연산에서 벡터/행렬 연산으로 전환하여 효율적인 신경망을 구현합니다.

## 📚 학습 목표
1. NumPy의 효율적인 배열 연산 이해
2. Broadcasting 메커니즘 마스터
3. Batch 처리와 병렬 연산의 장점 학습
4. 수치적으로 안정적인 구현 방법 습득

## 🏗️ 프로젝트 구조

```
projects/day2_vector_matrix/
├── README.md                       # 프로젝트 개요
├── requirements.txt               # 의존성 패키지
├── Makefile                      # 빌드 자동화
│
├── study_notes/                  # 📖 학습 자료
│   ├── 01_numpy_basics.md       # NumPy 기초
│   ├── 02_broadcasting.md       # Broadcasting 이해
│   ├── 03_batch_processing.md   # Batch 처리
│   └── 04_loss_functions.md     # 손실 함수들
│
├── notebooks/                    # 🔬 실습 노트북
│   └── vector_ops_tutorial.ipynb
│
├── tests/                        # 🧪 테스트
│   └── test_tensor_ops.py
│
└── 50_eval/                      # 🎯 평가/데모
    ├── mnist_mini.py            # Mini MNIST 분류
    └── benchmark.py             # 성능 비교

core/                            # 핵심 구현 (최상위)
├── tensor_ops.py                # 텐서 연산
└── nn_vectorized.py             # 벡터화된 신경망
```

## 📋 구현 체크리스트

### Core 모듈 (`core/tensor_ops.py`)
- [x] `matmul()`: 행렬곱 연산
- [x] `relu()`: ReLU 활성화 (벡터화)
- [x] `sigmoid()`: Sigmoid 활성화 (수치 안정)
- [x] `softmax()`: Softmax 함수 (안정적 구현)
- [x] `mse_loss()`: Mean Squared Error
- [x] `cross_entropy()`: Cross Entropy Loss
- [x] `batch_norm()`: Batch Normalization
- [x] `one_hot()`: One-hot 인코딩
- [x] `accuracy()`: 정확도 계산

### 벡터화된 신경망 (`core/nn_vectorized.py`)
- [x] `LinearLayer`: 벡터화된 완전 연결층
  - [x] Xavier/He 초기화
  - [x] Forward pass (배치 처리)
  - [x] Backward pass (그래디언트 계산)
- [x] `MLPVectorized`: 벡터화된 다층 퍼셉트론
  - [x] 여러 레이어 조합
  - [x] 배치 단위 처리
- [x] `SGDOptimizer`: 최적화기
  - [x] Momentum 지원

### 테스트 (`tests/test_tensor_ops.py`)
- [x] 활성화 함수 테스트
- [x] 손실 함수 테스트
- [x] Batch Normalization 테스트
- [x] 유틸리티 함수 테스트

### 평가 (`50_eval/`)
- [x] Mini MNIST 분류기
  - [x] 데이터 로딩 (sklearn digits)
  - [x] DataLoader 구현
  - [x] 학습 루프
  - [x] 평가 및 시각화
- [x] 성능 벤치마크
  - [x] 스칼라 vs 벡터 연산 비교
  - [x] 배치 크기별 성능 측정
  - [x] 메모리 효율성 분석

## 🎓 핵심 개념

### 1. Broadcasting
```python
# Shape이 다른 배열 간 연산
A = np.array([[1, 2, 3]])  # (1, 3)
B = np.array([[1], [2]])   # (2, 1)
C = A + B                   # (2, 3) - 자동으로 확장!
```

### 2. Batch Processing
```python
# 여러 샘플을 동시에 처리
X = np.random.randn(32, 784)  # 32개 샘플
W = np.random.randn(784, 128)
Y = X @ W  # 32개 결과를 한 번에!
```

### 3. 수치 안정성
```python
# LogSumExp Trick
def stable_softmax(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)  # 오버플로우 방지
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

## 🔥 도전 과제

### 필수 과제
1. ✅ 벡터화된 XOR 문제 해결
2. ✅ Mini MNIST (0-4) 90% 이상 정확도
3. ✅ 스칼라 대비 10배 이상 속도 향상

### 선택 과제
1. ⬜ Dropout 구현
2. ⬜ Adam Optimizer 구현  
3. ⬜ Learning Rate Scheduling
4. ⬜ Data Augmentation

## 📊 성능 목표

| 지표 | 목표 | 달성 여부 |
|------|------|-----------|
| Mini MNIST 정확도 | > 90% | ⬜ |
| 학습 시간 (1000 에폭) | < 30초 | ⬜ |
| 벡터화 속도 향상 | > 10x | ⬜ |
| 메모리 사용량 | < 100MB | ⬜ |

## 🔍 주요 인사이트

### Broadcasting의 힘
- 명시적 루프 없이 연산 가능
- 메모리 효율적 (실제로 복사하지 않음)
- 코드가 간결하고 읽기 쉬움

### Batch의 중요성
- GPU 활용도 극대화
- 학습 안정성 향상 (노이즈 감소)
- 병렬 처리로 처리량 증가

### 수치 안정성
- Overflow/Underflow 방지 필수
- LogSumExp Trick 활용
- Gradient Clipping으로 학습 안정화

## 💡 구현 팁

1. **Shape 확인 습관화**
   ```python
   def debug_shapes(*arrays):
       for i, arr in enumerate(arrays):
           print(f"Array {i}: {arr.shape}")
   ```

2. **keepdims=True 활용**
   ```python
   mean = X.mean(axis=0, keepdims=True)  # Broadcasting 용이
   ```

3. **벡터화 우선**
   ```python
   # Bad: for loop
   result = []
   for x in data:
       result.append(process(x))
   
   # Good: vectorized
   result = process(data)
   ```

## 🐛 일반적인 실수

1. **Shape 불일치**
   - 해결: 연산 전 shape 확인
   
2. **수치 불안정**
   - 해결: log 연산 전 clipping, 정규화

3. **메모리 낭비**
   - 해결: Broadcasting 활용, view vs copy 구분

## 📚 참고 자료

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/)

## ✅ 완료 기준

- [ ] 모든 테스트 통과 (`make test`)
- [ ] Mini MNIST 90% 이상 정확도
- [ ] 벤치마크에서 10배 이상 속도 향상 확인
- [ ] 학습 노트 완독 및 이해

## 🎯 다음 단계

Day 2를 완료하면 Day 3 (Attention Mechanism)으로 진행합니다.
Attention은 현대 LLM의 핵심 기술입니다!