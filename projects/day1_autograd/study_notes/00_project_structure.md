# 📂 프로젝트 파일 구조 완전 가이드

## 🎯 학습 경로: 노트북 vs Python 파일

### **주요 학습 도구: Jupyter 노트북** 
```bash
tiny_autograd_tutorial.ipynb  # 👈 메인 학습 자료
```
- **역할**: 단계별 설명과 실습이 모두 포함된 튜토리얼
- **장점**: 코드와 설명을 함께 보며 대화식으로 학습 가능
- **사용법**: 이 파일 하나로 전체 개념 학습 완료

### **Python 파일들의 역할**

## 1️⃣ `_10_core/autograd_tiny/value.py` - 핵심 구현체

**용도**: 실제 프로덕션 코드 (완성된 구현)

```python
# 이 파일의 역할
- ✅ 완성된 Value 클래스 구현
- ✅ 테스트에서 import해서 사용
- ✅ 다른 프로젝트에서 재사용 가능한 모듈
```

**학습 시 활용법**:
```python
# 1. 노트북에서 import해서 실험
from _10_core.autograd_tiny.value import Value

# 2. 구현 참고용 (막힐 때 답안지처럼 활용)
# 3. 나만의 구현과 비교
```

## 2️⃣ `50_eval/smoke.py` - 간단한 동작 확인

**용도**: 구현이 제대로 동작하는지 빠른 확인

```python
# smoke.py의 내용
a, b = Value(1.0), Value(2.0)
y = (a * b + a).tanh()
y.backward()
print(f"a.grad={a.grad}, b.grad={b.grad}")
```

**언제 사용?**
- ✅ 코드 수정 후 빠른 동작 확인
- ✅ 환경 설정이 제대로 되었는지 테스트
- ✅ CI/CD 파이프라인에서 기본 체크

**실행 방법**:
```bash
# 방법 1: Make 명령
make smoke

# 방법 2: 직접 실행
python 50_eval/smoke.py
```

## 3️⃣ `tests/test_value.py` - 정확성 검증

**용도**: 구현의 정확성을 수치적으로 검증

```python
# 주요 테스트 내용
- 자동미분 결과 vs 수치미분 결과 비교
- 상대 오차 < 1e-4 확인
- 여러 테스트 케이스로 검증
```

**언제 사용?**
- ✅ 새로운 연산 추가 후 검증
- ✅ 리팩토링 후 기능 보장
- ✅ PR 전 최종 확인

**실행 방법**:
```bash
# 방법 1: Make 명령
make test

# 방법 2: pytest 직접 실행
pytest tests/test_value.py -v
```

## 4️⃣ `00_common/` - 공통 유틸리티 (현재 비어있음)

**미래 용도**: 프로젝트 확장 시 사용

```python
# 향후 추가될 수 있는 파일들
00_common/
  utils.py        # 로깅, 디버깅 도구
  visualize.py    # 그래프 시각화
  constants.py    # 공통 상수
  benchmark.py    # 성능 측정 도구
```

## 📚 학습 순서 권장사항

### **초보자 경로** (권장)
```
1. tiny_autograd_tutorial.ipynb 실습 (80% 시간)
   ↓
2. smoke.py로 빠른 테스트
   ↓
3. value.py 코드 분석
   ↓
4. test_value.py로 검증
```

### **중급자 경로**
```
1. value.py 코드 직접 분석
   ↓
2. test_value.py 이해
   ↓
3. 노트북으로 실험
   ↓
4. 자체 구현 시도
```

## 🔄 파일 간 관계도

```
tiny_autograd_tutorial.ipynb
        ↓ (학습/이해)
        ↓
_10_core/autograd_tiny/value.py
        ↓ (import)
        ├─→ 50_eval/smoke.py (간단 테스트)
        └─→ tests/test_value.py (정확성 검증)
```

## 💡 실습 팁

### **노트북 중심 학습**
```python
# 노트북에서 이렇게 활용
# Cell 1: 직접 구현
class MyValue:
    # 내 구현...
    pass

# Cell 2: 정답과 비교
from _10_core.autograd_tiny.value import Value as CorrectValue

# Cell 3: 테스트
my_result = test_with_my_value()
correct_result = test_with_correct_value()
assert my_result == correct_result
```

### **파일 수정 실습**
```bash
# 1. value.py 백업
cp _10_core/autograd_tiny/value.py _10_core/autograd_tiny/value_backup.py

# 2. value.py 수정
# 예: 새로운 연산 추가

# 3. 테스트
make test

# 4. 문제 있으면 복구
cp _10_core/autograd_tiny/value_backup.py _10_core/autograd_tiny/value.py
```

## 🎓 학습 체크포인트

### Level 1: 노트북만 사용
- [ ] 노트북 전체 실행
- [ ] 각 셀 이해
- [ ] 연습문제 해결

### Level 2: Python 파일 활용
- [ ] value.py import해서 사용
- [ ] smoke.py 실행 및 이해
- [ ] test 실행 및 통과

### Level 3: 직접 수정
- [ ] value.py에 새 연산 추가
- [ ] 새로운 테스트 작성
- [ ] smoke.py 커스터마이즈

## 🚀 다음 단계

**노트북 학습 완료 후:**
1. `value.py`를 참고해 나만의 버전 작성
2. `tests/` 폴더에 추가 테스트 작성
3. `50_eval/` 폴더에 벤치마크 스크립트 추가
4. `00_common/`에 시각화 도구 구현

**핵심**: 노트북이 메인 학습 도구, 나머지 파일들은 보조/검증 도구!