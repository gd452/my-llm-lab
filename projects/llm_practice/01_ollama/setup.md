# Ollama + Qwen 설정 가이드

## 1. Ollama 설치

### macOS
```bash
# Homebrew로 설치
brew install ollama

# 또는 직접 다운로드
curl -fsSL https://ollama.ai/install.sh | sh
```

### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Windows
```powershell
# WSL2 사용 권장
# 또는 https://ollama.ai/download 에서 다운로드
```

## 2. Qwen 모델 설치

```bash
# 모델 크기별 선택
ollama pull qwen2:0.5b    # 500MB - 테스트용
ollama pull qwen2:1.5b    # 1.5GB - 일반 작업
ollama pull qwen2:7b      # 4GB - 고품질 (추천)

# 설치된 모델 확인
ollama list
```

## 3. 첫 실행

```bash
# 대화형 실행
ollama run qwen2:7b

# API 서버 실행 (백그라운드)
ollama serve

# API 테스트
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2:7b",
  "prompt": "Hello, how are you?",
  "stream": false
}'
```

## 4. Python 연동

```python
# requests로 간단히 사용
import requests
import json

def chat_with_qwen(prompt):
    response = requests.post('http://localhost:11434/api/generate',
        json={
            "model": "qwen2:7b",
            "prompt": prompt,
            "stream": False
        })
    return response.json()['response']

# 테스트
result = chat_with_qwen("What is the capital of France?")
print(result)
```

## 5. 성능 최적화 설정

```bash
# GPU 메모리 제한 (NVIDIA)
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_GPU=1

# CPU 스레드 설정
export OLLAMA_NUM_THREAD=8

# 모델 캐시 위치 변경
export OLLAMA_MODELS=/path/to/models
```

## 6. 모델 커스터마이징

```bash
# Modelfile 생성
cat > Modelfile << EOF
FROM qwen2:7b

# 시스템 프롬프트 설정
SYSTEM """
You are a helpful coding assistant. 
Always provide clear explanations and working code examples.
"""

# 파라미터 조정
PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9
EOF

# 커스텀 모델 생성
ollama create my-qwen -f Modelfile

# 실행
ollama run my-qwen
```

## 7. 문제 해결

### 메모리 부족
```bash
# 더 작은 모델 사용
ollama pull qwen2:0.5b

# 또는 양자화 모델
ollama pull qwen2:7b-q4_0  # 4-bit 양자화
```

### 속도 개선
```bash
# GPU 사용 확인
ollama run qwen2:7b --verbose

# CPU 전용 모드
OLLAMA_NUM_GPU=0 ollama run qwen2:7b
```

### 포트 변경
```bash
# 기본 11434 포트 변경
OLLAMA_HOST=0.0.0.0:8080 ollama serve
```

## 📝 체크리스트

- [ ] Ollama 설치 완료
- [ ] Qwen2 모델 다운로드
- [ ] 첫 대화 테스트
- [ ] Python API 연동
- [ ] 커스텀 모델 생성

준비가 완료되면 `basic_chat.py`로 진행하세요!