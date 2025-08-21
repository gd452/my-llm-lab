# 🧠 Day 1.5: Neural Network 기초 - From Neuron to MLP

> "우리는 뇌의 뉴런처럼 작동하는 인공 뉴런을 만들어봅니다" - Andrej Karpathy 스타일

## 🎯 학습 목표

이 프로젝트를 완료하면 다음을 이해하게 됩니다:
- 🔸 **뉴런**: 가장 작은 학습 단위
- 🔸 **레이어**: 뉴런들의 조직화
- 🔸 **MLP**: 다층 퍼셉트론의 구조
- 🔸 **학습**: 경사하강법과 역전파
- 🔸 **XOR 문제**: 신경망의 Hello World

## 📚 선수 지식

- ✅ Tiny Autograd 완료 (역전파 이해)
- ✅ Python 기초
- ✅ 고등학교 수학 (미분)

## 🗂️ 프로젝트 구조

```
tiny_neural_net/
├── _10_core/              # 핵심 구현 (밑줄로 시작해서 import 용이)
│   └── nn_tiny/
│       ├── neuron.py      # 단일 뉴런 구현
│       ├── layer.py       # 레이어 (뉴런 집합)
│       ├── mlp.py         # Multi-Layer Perceptron
│       ├── losses.py      # 손실 함수 (MSE, CrossEntropy)
│       └── optimizer.py   # 최적화 알고리즘 (SGD)
├── notebooks/
│   └── nn_tutorial.ipynb # 상세 튜토리얼
├── tests/
│   └── test_nn.py        # 단위 테스트
├── 50_eval/
│   └── xor_demo.py       # XOR 문제 해결
└── study_notes/
    └── nn_concepts.md    # 핵심 개념 정리
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
cd tiny_neural_net
pip install -r requirements.txt
```

### 2. 테스트 실행 (현재 실패 - TODO 구현 필요)
```bash
python -m pytest tests/ -v
```

### 3. 노트북으로 학습
```bash
jupyter notebook notebooks/nn_tutorial.ipynb
```

### 4. XOR 문제 도전
```bash
python 50_eval/xor_demo.py
```

## 📝 구현 체크리스트

### Stage 1: 뉴런 (1시간)
- [ ] `Neuron.__init__`: 가중치와 편향 초기화
- [ ] `Neuron.__call__`: forward pass (wx + b)
- [ ] `Neuron.parameters()`: 학습 가능한 파라미터 반환

### Stage 2: 레이어 (30분)
- [ ] `Layer.__init__`: n개의 뉴런 생성
- [ ] `Layer.__call__`: 모든 뉴런에 입력 전달
- [ ] `Layer.parameters()`: 모든 뉴런의 파라미터

### Stage 3: MLP (1시간)
- [ ] `MLP.__init__`: 다층 구조 생성
- [ ] `MLP.__call__`: 순차적 forward pass
- [ ] `MLP.parameters()`: 전체 네트워크 파라미터

### Stage 4: 학습 (1시간)
- [ ] `mse_loss`: Mean Squared Error
- [ ] `SGD.step()`: 파라미터 업데이트
- [ ] `SGD.zero_grad()`: gradient 초기화

### Stage 5: XOR 해결 (1시간)
- [ ] 데이터 준비
- [ ] 네트워크 생성 (2-4-1 구조)
- [ ] 학습 루프
- [ ] 결과 시각화

## 🎓 핵심 개념

### 1. 뉴런의 수학
```python
# 단일 뉴런의 계산
output = activation(w1*x1 + w2*x2 + ... + b)

# 우리의 구현 (Value 클래스 사용)
output = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
output = output.tanh()  # 활성화 함수
```

### 2. XOR 문제
```
입력: (0,0) → 출력: 0
입력: (0,1) → 출력: 1
입력: (1,0) → 출력: 1
입력: (1,1) → 출력: 0

선형으로 분리 불가능 → 은닉층 필요!
```

### 3. 학습 과정
```python
for epoch in range(1000):
    # Forward pass
    y_pred = model(X)
    
    # Compute loss
    loss = mse_loss(y_pred, y_true)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update parameters
    optimizer.step()
```

## 🔍 디버깅 팁

1. **Gradient 확인**: 모든 파라미터의 grad가 0이 아닌지 확인
2. **Loss 추적**: 학습 중 loss가 감소하는지 모니터링
3. **학습률**: 너무 크면 발산, 너무 작으면 수렴 느림
4. **초기화**: 가중치 초기화가 중요 (Xavier, He 초기화)

## 📊 기대 결과

### XOR 학습 곡선
```
Epoch 0: Loss = 0.25
Epoch 100: Loss = 0.18
Epoch 500: Loss = 0.05
Epoch 1000: Loss = 0.001
```

### 정확도
```
(0, 0) → 0.02 ≈ 0 ✓
(0, 1) → 0.98 ≈ 1 ✓
(1, 0) → 0.97 ≈ 1 ✓
(1, 1) → 0.03 ≈ 0 ✓
```

## 💡 도전 과제

### Level 1: 기본
- [ ] AND, OR 게이트 학습
- [ ] 다른 활성화 함수 (ReLU, Sigmoid)

### Level 2: 중급
- [ ] 3-bit parity 문제
- [ ] 원형 데이터 분류

### Level 3: 고급
- [ ] MNIST 숫자 인식 (784-128-10)
- [ ] 정규화 기법 (L2, Dropout)

## 🔗 참고 자료

- [Andrej Karpathy - micrograd](https://github.com/karpathy/micrograd)
- [Neural Networks from Scratch](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [XOR Problem Visualization](https://playground.tensorflow.org/)

## ⏰ 예상 소요 시간

- **총 시간**: 4-6시간
- **권장 진행**:
  - 1시간: 노트북 따라하기
  - 2시간: 직접 구현
  - 1시간: XOR 문제 해결
  - 1-2시간: 도전 과제

## 🎯 다음 단계

이 프로젝트를 완료하면:
- **Day 2**: 벡터/행렬 연산으로 확장
- **Day 3**: Attention 메커니즘 이해
- **Day 4**: Transformer 구현
- **Day 5**: 실제 텍스트 생성

---

**"The key to understanding deep learning is to build it from scratch!"** - Andrej Karpathy