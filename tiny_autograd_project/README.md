# Day 1 — Tiny Autograd (Scalar)

**목표:**
- 스칼라용 자동미분 엔진의 최소구성 구현
- 연산자 오버로딩(`+`, `*`, `tanh`, 선택: `relu`)과 `backward()`
- 수치미분과의 상대오차 비교(테스트 제공)

## 프로젝트 구조

```
tiny_autograd_project/
  00_common/                   # 공통 유틸리티 (현재 비어있음, 확장용)
  _10_core/                    # 핵심 비즈니스 로직
    autograd_tiny/
      __init__.py
      value.py                 # 자동미분 엔진 구현
  50_eval/                     # 평가 및 실행 스크립트
    smoke.py                   # 스모크 테스트
  tests/
    test_value.py              # 정답 검증(수치미분 대조)
  .github/
    workflows/ci.yml           # CI: ruff/black/mypy/pytest
    PULL_REQUEST_TEMPLATE.md
  tiny_autograd_tutorial.ipynb # 상세 설명이 포함된 Jupyter 노트북
  Makefile
  pyproject.toml
  requirements.txt
  README.md
```

### 📁 디렉토리 명명 규칙 설명

#### **숫자 접두사 시스템**
- **00_common**: 가장 기초적인 공통 모듈 (유틸리티, 상수 등)
- **10~40**: 핵심 기능 모듈 (낮은 번호 = 기초, 높은 번호 = 응용)
- **50+**: 평가, 테스트, 실행 스크립트

#### **언더스코어 prefix (`_10_core`)**
Python이 숫자로 시작하는 모듈명을 허용하지 않아 `_` 추가
- `import 10_core` ❌ 
- `import _10_core` ✅

#### **각 디렉토리의 역할**

| 디렉토리 | 역할 | 예시 내용 |
|---------|------|----------|
| `00_common/` | 공통 유틸리티 | 로깅, 설정, 상수 등 |
| `_10_core/` | 핵심 비즈니스 로직 | 자동미분 엔진 (`value.py`) |
| `50_eval/` | 평가/실행 스크립트 | 스모크 테스트, 벤치마크, 데모 |
| `tests/` | 단위 테스트 | pytest 기반 정확성 검증 |

#### **주요 파일 설명**

- **`value.py`**: 자동미분을 위한 핵심 `Value` 클래스
  - 스칼라 값과 gradient 저장
  - 연산 그래프 구축 (`_prev`, `_backward`)
  - 역전파 알고리즘 구현

- **`smoke.py`**: "연기가 나는지" 확인하는 기본 동작 테스트
  - 간단한 연산 그래프 생성
  - 역전파 실행
  - 기본 동작 확인용

이러한 구조는 프로젝트가 확장될 때 체계적인 관리를 가능하게 합니다:
```
# 향후 확장 예시
_20_optimizer/   # 최적화 알고리즘
_30_nn/          # 신경망 레이어
_40_distributed/ # 분산 학습
```

## 빠른 시작

### 1. 의존성 설치
```bash
make setup  # 또는 pip install -r requirements.txt
```

### 2. 테스트 실행 (현재 구현으로 통과)
```bash
make test   # pytest -q
```

### 3. 스모크 테스트
```bash
make smoke  # 또는 python 50_eval/smoke.py
```

### 4. Jupyter 노트북으로 학습
```bash
jupyter notebook tiny_autograd_tutorial.ipynb
```

## 구현 요구사항

### Value 클래스
- **속성**: 
  - `data: float` - 실제 스칼라 값
  - `grad: float=0.0` - gradient 값
  - `_prev: set[Value]` - 부모 노드들
  - `_backward: Callable[[], None]` - 역전파 함수

- **연산**: 
  - `__add__`, `__mul__` - 기본 산술 연산
  - `tanh()`, `relu()` - 활성화 함수
  - `__radd__`, `__rmul__` - 역순 연산

- **핵심 메서드**:
  - `backward()` - 그래프를 위상정렬로 순회하며 chain rule 적용

### 테스트
- 함수: `f(a,b) = (a*b + a) * tanh(b)`
- 수치미분과의 상대오차가 `1e-4` 이하

## 핵심 개념

### 1. Chain Rule
각 연산 시 **로컬 미분 규칙**을 `_backward` 클로저로 저장:
- 덧셈: `d(a+b)/da = 1, d(a+b)/db = 1`
- 곱셈: `d(a*b)/da = b, d(a*b)/db = a`
- tanh: `dtanh(x)/dx = 1 - tanh(x)^2`

### 2. 역전파 알고리즘
1. DFS로 그래프 위상정렬
2. 출력 노드의 gradient를 1로 설정
3. 역순으로 각 노드의 `_backward()` 실행

## 명령어 모음

```bash
make setup   # 의존성 설치
make test    # 테스트 실행
make smoke   # 스모크 테스트

# 선택적 (필요시)
make lint    # 기본 오류만 체크
make fmt     # 코드 포맷 체크
make strict  # 엄격한 체크 (lint + format + type)
```

## 학습 가이드

### Jupyter 노트북 구성
1. **기본 개념**: 자동미분, 연산 그래프, Chain Rule
2. **단계별 구현**: 
   - Value 클래스 기본 구조
   - 덧셈/곱셈 연산
   - 활성화 함수 (tanh, ReLU)
   - 역전파 알고리즘
3. **검증**: 수치미분과 비교
4. **시각화**: 연산 그래프 구조
5. **연습 문제**: 추가 연산 구현

### 학습 순서 (권장)
1. Jupyter 노트북을 열어 개념 이해
2. 각 셀을 실행하며 동작 확인
3. 연습 문제 풀이
4. 테스트 통과 확인
5. 추가 기능 구현 시도

## PR 체크리스트

- [ ] 모든 테스트 통과
- [ ] CI(ruff/black/mypy/pytest) 통과
- [ ] 코드 리뷰 반영
- [ ] 문서 업데이트

## 다음 단계

- Day 2: `softmax(log-sum-exp)`와 `cross-entropy` 벡터화 연산으로 확장
- Day 3: 간단한 신경망 구현
- Day 4: 최적화 알고리즘 (SGD, Adam) 추가

## 참고 자료

- [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd)
- [PyTorch Autograd Tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)