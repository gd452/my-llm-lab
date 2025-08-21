# 🧠 Attention의 직관적 이해

## 🎯 학습 목표
- Attention이 왜 필요한지 이해
- 기존 방법의 한계 파악
- Attention의 핵심 아이디어 습득

## 1. 왜 Attention이 필요한가?

### 기존 RNN의 문제점

```
문장: "The cat that chased the mouse sat on the mat"

RNN: The → cat → that → chased → ... → mat
      (초기 정보가 점점 희석됨)
```

RNN은 순차적으로 처리하기 때문에:
1. **정보 병목**: 모든 정보를 하나의 hidden state에 압축
2. **장거리 의존성**: 멀리 떨어진 단어 간 관계 포착 어려움
3. **순차 처리**: 병렬화 불가능, 학습이 느림

### Attention의 해결책

```
Attention: 모든 단어가 다른 모든 단어를 직접 "볼" 수 있음

"The cat sat on the mat"
 ↓
cat은 다음에 주목:
- "The" (0.3) - 관사
- "sat" (0.5) - 주요 동작
- "mat" (0.2) - 위치
```

## 2. Attention의 핵심 아이디어

### 일상 생활의 비유

교실에서 선생님의 질문에 답하는 상황:

```
선생님: "수도가 파리인 나라는?"

학생의 뇌:
1. Query (질문): "수도가 파리인 나라"
2. Key (기억 검색): 
   - 프랑스-파리 (매치! 0.9)
   - 한국-서울 (0.1)
   - 일본-도쿄 (0.1)
3. Value (답변 추출): "프랑스"
```

### 프로그래밍 비유

```python
# Dictionary 검색과 유사
memory = {
    "프랑스": "파리",  # Key: Value
    "한국": "서울",
    "일본": "도쿄"
}

query = "파리"
# Soft lookup - 모든 key와 비교
attention_scores = similarity(query, all_keys)
result = weighted_sum(all_values, attention_scores)
```

## 3. Attention 메커니즘 단계별

### Step 1: 유사도 계산

```
Query: "cat"이 무엇과 관련있는지 찾기

       The  cat  sat  on  the  mat
cat:   0.2  1.0  0.7  0.1  0.2  0.3

(자기 자신과 가장 유사, 동사 'sat'과도 높은 관련)
```

### Step 2: 정규화 (Softmax)

```
원시 점수: [0.2, 1.0, 0.7, 0.1, 0.2, 0.3]
     ↓
Softmax 후: [0.10, 0.35, 0.25, 0.08, 0.10, 0.12]
(합이 1이 되도록 정규화)
```

### Step 3: 가중 합

```
각 단어의 정보를 가중치만큼 가져옴:
result = 0.10 * The_vector 
       + 0.35 * cat_vector
       + 0.25 * sat_vector
       + ...
```

## 4. Self-Attention vs Cross-Attention

### Self-Attention
같은 문장 내에서 단어들 간의 관계

```
"The cat sat on the mat"
    ↓
각 단어가 같은 문장의 다른 단어들에 주목
```

### Cross-Attention
다른 문장/소스와의 관계

```
영어: "The cat sat"
     ↓ (번역)
한국어: "고양이가 [?]"
        ↓
"sat"에 주목하여 "앉았다" 생성
```

## 5. 실제 예제: 문장 이해

### 예제 1: 대명사 해결

```
문장: "The dog chased the cat. It was fast."

"It"이 무엇을 가리키는가?
- Attention이 "dog"에 높은 가중치 → "It" = "dog"
```

### 예제 2: 관계 파악

```
문장: "The key to the cabinet is on the table."

"key"의 attention:
- "cabinet" (0.4) - 강한 관계
- "table" (0.3) - 위치 관계
- "to" (0.2) - 문법적 연결
```

## 6. Attention의 장점

### 1. 병렬 처리
```
RNN:  A → B → C → D (순차적)
Attention: A, B, C, D (동시에)
```

### 2. 장거리 의존성
```
"The book that I bought yesterday from the store was interesting"
 ↑                                                      ↑
 └──────────────── 직접 연결 ────────────────────────┘
```

### 3. 해석 가능성
```
Attention 가중치를 시각화하면 모델이 무엇에 주목하는지 볼 수 있음
```

## 7. 시각적 이해

### Attention Matrix

```
        The  cat  sat  on  the  mat
The     0.8  0.1  0.0  0.0  0.1  0.0
cat     0.3  0.5  0.1  0.0  0.0  0.1
sat     0.1  0.4  0.2  0.1  0.1  0.1
on      0.0  0.0  0.2  0.5  0.0  0.3
the     0.7  0.0  0.0  0.0  0.3  0.0
mat     0.1  0.2  0.1  0.3  0.1  0.2

(각 행: Query, 각 열: Key)
```

### Heatmap 표현

```
■ = 높은 attention
□ = 낮은 attention

        T  c  s  o  t  m
    T   ■  □  □  □  □  □
    c   ■  ■  □  □  □  □
    s   □  ■  □  □  □  □
    o   □  □  □  ■  □  ■
    t   ■  □  □  □  □  □
    m   □  □  □  ■  □  □
```

## 💡 핵심 통찰

1. **Attention = 선택적 정보 추출**
   - 모든 정보를 동등하게 보지 않음
   - 중요한 것에 "주목"

2. **유연한 연결**
   - 고정된 구조가 아님
   - 입력에 따라 동적으로 변화

3. **전역적 문맥**
   - 모든 위치의 정보에 접근 가능
   - 거리 제약 없음

## 🔍 이해도 체크

다음 질문에 답해보세요:

1. RNN과 Attention의 가장 큰 차이는?
2. Query, Key, Value의 역할은?
3. Attention 가중치의 합이 1인 이유는?
4. Self-Attention과 Cross-Attention의 차이는?

## 📝 연습 문제

1. "The student who studied hard passed the exam"
   - "passed"가 가장 주목해야 할 단어는?
   - "student"와 "exam"의 관계는?

2. 다음 attention 가중치를 해석하세요:
   ```
   "I love ice cream"
   love → [I: 0.3, love: 0.2, ice: 0.25, cream: 0.25]
   ```

## 다음 단계

이제 수학적으로 정확히 어떻게 구현되는지 알아봅시다!
→ [02_self_attention.md](02_self_attention.md)