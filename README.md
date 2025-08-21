# 🚀 My LLM Lab: 나만의 Tiny LLM 만들기

> "The best way to understand something is to build it from scratch" - Andrej Karpathy

## 🎯 목표
나만의 아주 간단한 tiny LLM을 통해 주요 LLM을 이해하고, LLM 전반을 이해하는 expert가 되기 위한 기초 과정

## 📚 전체 학습 경로

### Week 1: 기초 다지기

#### ✅ Day 1: Tiny Autograd Package
- **상태**: 완료
- **폴더**: `tiny_autograd_project/`
- **내용**: 
  - 자동 미분 구현
  - 계산 그래프와 역전파
  - Value 클래스 구현
  - 위상 정렬과 Chain Rule

#### ✅ Day 1.5: Neural Network 기초
- **상태**: 완료
- **폴더**: `projects/day1_5_neural_net/`
- **내용**:
  - Neuron, Layer, MLP 구현
  - XOR 문제 해결
  - 경사하강법과 최적화
  - 활성화 함수와 손실 함수

#### ✅ Day 2: 벡터/행렬 연산
- **상태**: 완료
- **폴더**: `projects/day2_vector_matrix/`
- **내용**:
  - NumPy로 효율적인 연산
  - Batch 처리
  - Softmax와 CrossEntropy
  - 행렬곱과 Broadcasting
  - Mini MNIST 분류기

#### ✅ Day 3: Attention 메커니즘
- **상태**: 생성 완료, 학습 대기
- **폴더**: `projects/day3_attention/`
- **내용**:
  - Self-Attention 구현
  - Query, Key, Value 이해
  - Scaled Dot-Product Attention
  - Multi-Head Attention
  - Positional Encoding

#### ⏳ Day 4: Transformer Block
- **상태**: 대기
- **예정 내용**:
  - 전체 Transformer 아키텍처
  - Positional Encoding
  - Layer Normalization
  - Feed-Forward Network

#### ⏳ Day 5: 토크나이저와 학습
- **상태**: 대기
- **예정 내용**:
  - BPE 토크나이저
  - 텍스트 전처리
  - 학습 루프 구현
  - 간단한 텍스트 생성

### Week 2: Mini GPT 구현

#### ⏳ Day 6-10: Mini GPT
- **예정 내용**:
  - GPT 아키텍처 구현
  - Causal Attention
  - 텍스트 생성 전략
  - Temperature, Top-k, Top-p
  - 작은 데이터셋으로 학습

### Week 3: 파인튜닝과 응용

#### ⏳ Day 11-15: 챗봇 만들기
- **예정 내용**:
  - Instruction Tuning
  - Prompt Engineering
  - 대화 컨텍스트 관리
  - 간단한 챗봇 구현

### Week 4: 프로젝트

#### ⏳ Day 16-20: 도메인 특화 LLM
- **예정 내용**:
  - 특정 도메인 데이터 수집
  - 커스텀 토크나이저
  - 도메인 특화 파인튜닝
  - 성능 평가 및 개선

## 🛠️ 기술 스택
- Python 3.8+
- NumPy (벡터 연산)
- PyTorch (Week 2부터 선택적)
- Jupyter Notebook (인터랙티브 학습)

## 📖 학습 방법

### 각 Day별 진행 순서:
1. **개념 학습**: `study_notes/` 읽기
2. **튜토리얼**: `notebooks/` 따라하기
3. **구현**: 스켈레톤 코드 완성
4. **테스트**: pytest로 검증
5. **실습**: 데모 실행 및 실험

### 권장 학습 시간:
- 하루 2-4시간
- 각 Day 완료 후 충분한 복습
- 이해가 안 되면 다음 단계로 넘어가지 말 것

## 🎓 선수 지식
- Python 기초 문법
- 고등학교 수준 수학 (미분)
- 프로그래밍 기본 개념

## 💡 핵심 원칙
1. **처음부터 구현**: 라이브러리 사용 최소화
2. **이해 중심**: 암기보다 원리 이해
3. **점진적 학습**: 작은 것부터 차근차근
4. **실습 위주**: 코드로 직접 확인

## 🔗 참고 자료
- [Andrej Karpathy - Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 📝 진행 상황
- [x] Day 1: Autograd 완료
- [x] Day 1.5: Neural Network 완료
- [x] Day 2: Vector/Matrix Operations 완료
- [x] Day 3: Attention Mechanism 프로젝트 생성
- [ ] Day 3: Attention 학습 및 구현
- [ ] Day 4-5: 대기
- [ ] Week 2-4: 대기

## 🚦 다음 단계
현재 Day 3 (Attention Mechanism) 학습 준비 완료.
`projects/day3_attention/study_notes/`를 학습한 후 Day 4로 진행 예정.

---

**"작은 걸음이 모여 큰 도약이 됩니다. 포기하지 마세요!"**