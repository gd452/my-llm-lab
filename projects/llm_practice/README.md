# Week 2: 오픈소스 LLM 실습 - Qwen 모델

## 🎯 목표
**실무에서 바로 사용 가능한 LLM 활용 능력 습득**

## 📊 Qwen 모델 선택 이유

1. **최신 성능**: GPT-3.5 수준 또는 그 이상
2. **한국어 지원**: 뛰어난 다국어 능력
3. **다양한 크기**: 0.5B ~ 72B (용도별 선택 가능)
4. **상업적 사용**: Apache 2.0 라이선스

## 🔧 실습 구성

### Phase 1: Local LLM 실행 (Ollama)
```bash
# Qwen 모델 설치 및 실행
ollama pull qwen2:0.5b    # 가장 작은 모델 (빠른 테스트)
ollama pull qwen2:7b      # 실용적 크기
ollama run qwen2:7b
```

### Phase 2: Hugging Face 활용
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Qwen2-1.5B 모델 (코딩 작업용)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B")
```

### Phase 3: Fine-tuning with LoRA
```python
# 효율적인 fine-tuning
from peft import LoraConfig, get_peft_model

# 4-bit quantization + LoRA
# 7B 모델도 일반 GPU에서 학습 가능
```

### Phase 4: RAG 시스템 구축
```python
# 나만의 지식 베이스 연동
from langchain import Qwen2LLM
from langchain.vectorstores import Chroma

# PDF, 노트, 코드 → Vector DB → Qwen
```

## 📁 프로젝트 구조

```
llm_practice/
├── 01_ollama/
│   ├── basic_chat.py        # 기본 대화
│   ├── streaming.py         # 스트리밍 응답
│   └── api_server.py        # REST API 서버
│
├── 02_huggingface/
│   ├── inference.py         # 추론 최적화
│   ├── quantization.py      # 4-bit, 8-bit 양자화
│   └── batch_processing.py  # 배치 처리
│
├── 03_fine_tuning/
│   ├── prepare_data.py      # 데이터 준비
│   ├── lora_training.py     # LoRA 학습
│   └── merge_weights.py     # 가중치 병합
│
└── 04_rag_system/
    ├── document_loader.py   # 문서 로딩
    ├── vector_store.py      # 벡터 DB
    └── rag_chat.py          # RAG 챗봇
```

## 🚀 실습 스케줄

### Day 1-2: Ollama + Qwen 기초
- 로컬 실행 환경 구축
- 다양한 모델 크기 비교
- Prompt engineering 실습

### Day 3-4: Hugging Face 생태계
- 모델 로딩 최적화
- Quantization (메모리 절약)
- Batch inference

### Day 5-6: Fine-tuning
- 커스텀 데이터셋 준비
- LoRA/QLoRA 학습
- 학습 모니터링

### Day 7: RAG 시스템
- 문서 임베딩
- 벡터 검색
- Context-aware 응답

## 💡 실무 활용 예시

### 1. 코드 리뷰 봇
```python
# 코드 분석 및 개선 제안
def review_code(code: str):
    prompt = f"Review this code and suggest improvements:\n{code}"
    return qwen_model.generate(prompt)
```

### 2. 문서 요약기
```python
# 긴 문서 → 핵심 요약
def summarize_document(doc: str):
    prompt = f"Summarize in 3 bullet points:\n{doc}"
    return qwen_model.generate(prompt)
```

### 3. SQL 생성기
```python
# 자연어 → SQL 쿼리
def text_to_sql(question: str, schema: str):
    prompt = f"Schema: {schema}\nQuestion: {question}\nSQL:"
    return qwen_model.generate(prompt)
```

## 🎓 학습 목표 체크리스트

- [ ] Ollama로 로컬 LLM 실행
- [ ] Streaming response 구현
- [ ] 4-bit quantization으로 메모리 최적화
- [ ] LoRA fine-tuning 실행
- [ ] Custom dataset 준비
- [ ] RAG 시스템 구축
- [ ] 실제 업무 적용 사례 1개 구현

## 📈 성능 벤치마크

| 모델 | 크기 | 메모리 | 속도 | 품질 | 용도 |
|------|------|--------|------|------|------|
| Qwen2-0.5B | 0.5B | 1GB | 매우빠름 | 보통 | 테스트, 간단작업 |
| Qwen2-1.5B | 1.5B | 3GB | 빠름 | 좋음 | 코딩, 번역 |
| Qwen2-7B | 7B | 14GB | 보통 | 매우좋음 | 전문작업 |
| Qwen2-72B | 72B | 140GB | 느림 | 최고 | 연구, 고급작업 |

## 🔍 Week 3 Preview: 나만의 메모 비서

```python
# 내 노트 + Qwen = 개인 AI 비서
class PersonalAssistant:
    def __init__(self):
        self.model = load_qwen_model()
        self.notes = load_my_notes()
        self.vector_db = create_vector_store(self.notes)
    
    def answer(self, question):
        # 1. 관련 노트 검색
        context = self.vector_db.search(question)
        # 2. Qwen으로 답변 생성
        return self.model.generate_with_context(question, context)
```

준비되셨나요? 실습을 시작해볼까요?