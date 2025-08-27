# LangChain + Ollama 통합

## 🎯 학습 목표
- LangChain 기본 개념 이해
- Ollama와 LangChain 통합
- 체인(Chain) 구성과 활용
- 프롬프트 템플릿 활용
- 메모리 관리

## 📚 커리큘럼

### 1. 기본 셋업 (`basic_setup.py`)
- LangChain 설치
- Ollama 연동
- 첫 체인 구성

### 2. 프롬프트 엔지니어링 (`prompt_templates.py`)
- PromptTemplate 활용
- FewShotPromptTemplate
- 동적 프롬프트 생성

### 3. 체인 패턴 (`chain_patterns.py`)
- Sequential Chain
- Router Chain
- Transform Chain
- 커스텀 체인

### 4. 메모리 관리 (`memory_management.py`)
- ConversationBufferMemory
- ConversationSummaryMemory
- VectorStoreMemory
- 메모리 영속화

### 5. 출력 파서 (`output_parsers.py`)
- JSON 출력 파싱
- Pydantic 통합
- 구조화된 출력

## 📦 필요 패키지

```bash
pip install langchain langchain-community pydantic
```

## 🚀 빠른 시작

```python
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Ollama 모델 초기화
llm = Ollama(model="qwen3:8b")

# 프롬프트 템플릿
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms:"
)

# 체인 구성
chain = LLMChain(llm=llm, prompt=prompt)

# 실행
result = chain.run(topic="quantum computing")
print(result)
```

## 📝 체크리스트

- [ ] LangChain 설치
- [ ] 기본 체인 구성
- [ ] 프롬프트 템플릿 활용
- [ ] 메모리 구현
- [ ] 출력 파싱 구현