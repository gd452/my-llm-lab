# LLM 실습 프로젝트

실무에서 바로 사용 가능한 LLM 활용 기법을 배워봅니다.

## 🌟 하이라이트

- **Qwen3 최신 모델** 활용 (2025년 4월 출시)
- **Thinking Mode** 지원으로 심층 추론 가능
- **실무 중심** 코드와 예제
- **단계별** 학습 커리큘럼

## 학습 목표

1. **Ollama + Qwen3** 로컬 LLM 설정 및 활용
2. **LangChain** 프레임워크로 체인 구성
3. **Embeddings** 벡터화와 유사도 검색
4. **RAG** 기반 지식 활용 시스템
5. **Fine-tuning** LoRA로 효율적 커스터마이징
6. **Agents** ReAct 프레임워크 자율 에이전트

## 프로젝트 구조

```
llm_practice/
├── 01_ollama/          # Ollama + Qwen3 기초
│   ├── setup.md       # Qwen3 설치 가이드
│   └── basic_chat.py  # 기본 대화 구현
│
├── 02_langchain/       # LangChain 활용
│   ├── basic_setup.py       # 기본 체인 구성
│   └── prompt_templates.py  # 프롬프트 엔지니어링
│
├── 03_embeddings/      # 임베딩과 벡터 검색
│   └── embedding_basics.py  # 임베딩 기초
│
├── 04_rag_basics/      # RAG 시스템
│   └── simple_rag.py        # RAG 파이프라인
│
├── 05_fine_tuning/     # 파인튜닝
│   └── lora_finetuning.py   # LoRA 파인튜닝
│
└── 06_agents/          # AI 에이전트
    └── autonomous_agents.py  # ReAct 에이전트
```

## 커리큘럼

### 📅 Week 1: 기초 세팅
- [x] Ollama 설치 및 Qwen3 모델 설정
- [x] Thinking Mode를 활용한 심층 추론
- [x] 기본 API 호출 및 대화 구현

### 🔗 Week 2: 프레임워크 활용
- [x] LangChain 기본 체인 구성
- [x] 프롬프트 템플릿과 Few-shot Learning
- [x] 임베딩과 의미 검색

### 🎯 Week 3: RAG & 파인튜닝
- [x] 벡터 DB와 문서 검색
- [x] RAG 파이프라인 구축
- [x] LoRA로 효율적 파인튜닝

### 🤖 Week 4: 에이전트 개발
- [x] ReAct 프레임워크 구현
- [x] 도구 사용과 메모리 관리
- [x] 다중 에이전트 시스템

## 설치 가이드

### 1️⃣ 기본 패키지 설치
```bash
# 필수 패키지
pip install langchain langchain-community
pip install sentence-transformers
pip install chromadb
pip install peft transformers
```

### 2️⃣ Ollama 설치
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows (WSL2 권장)
```

### 3️⃣ Qwen3 모델 다운로드
```bash
# Qwen3 최신 모델 (Thinking Mode 지원)
ollama pull qwen3:8b
ollama pull qwen3:30b-a3b  # MoE 모델 (선택)

# 실행 테스트
ollama run qwen3:8b
```

## 빠른 시작

### 기본 대화
```python
# 01_ollama/basic_chat.py
from QwenChat import QwenChat

chat = QwenChat(model="qwen3:8b")
response = chat.chat("What is machine learning?")
print(response)
```

### Thinking Mode 활용
```python
# Qwen3의 특별 기능
response = chat.chat(
    "피보나치 수열의 10번째 항을 구해주세요",
    thinking_mode=True  # 심층 추론 활성화
)
```

## 학습 자료

- [📖 Qwen3 공식 문서](https://qwenlm.github.io/blog/qwen3/)
- [🦙 Ollama 공식 사이트](https://ollama.ai/)
- [🔗 LangChain 튜토리얼](https://python.langchain.com/)
- [🎆 LoRA 논문](https://arxiv.org/abs/2106.09685)
- [🤔 ReAct 프레임워크](https://arxiv.org/abs/2210.03629)

## 실습 환경

- **OS**: macOS/Linux/Windows (WSL2)
- **Python**: 3.9+
- **GPU**: 선택사항 (CPU로도 가능)
- **RAM**: 8GB+ 권장 (16GB 추천)
- **저장공간**: 10GB+ (모델 크기에 따라 상이)

## 트러블슈팅

### Ollama 연결 오류
```bash
# Ollama 서비스 시작
ollama serve

# 또는 백그라운드 실행
ollama serve &
```

### 메모리 부족
```bash
# 더 작은 모델 사용
ollama pull qwen3:4b

# 또는 양자화 모델
ollama pull qwen3:8b-q4_0
```

## 기여

문제가 발생하거나 개선 사항이 있으면 Issues에 등록해주세요.

## 라이선스

MIT License