"""
Ollama + Qwen 기본 대화 구현
실무에서 바로 사용 가능한 패턴들
"""

import requests
import json
from typing import Optional, Dict, List
import time

class QwenChat:
    """Qwen 모델과 대화하는 클래스"""
    
    def __init__(self, model: str = "qwen3:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.conversation_history = []
        
        # 모델 사용 가능 확인
        if not self.check_model():
            print(f"⚠️  모델 '{model}'을 찾을 수 없습니다.")
            print("먼저 실행하세요: ollama pull " + model)
    
    def check_model(self) -> bool:
        """모델이 설치되어 있는지 확인"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            models = response.json()
            return any(m['name'] == self.model for m in models.get('models', []))
        except:
            return False
    
    def chat(self, prompt: str, temperature: float = 0.7, 
             system_prompt: Optional[str] = None, thinking_mode: bool = False) -> str:
        """단순 대화 (컨텍스트 없음)"""
        
        # Qwen3 Thinking Mode 지원
        if thinking_mode:
            prompt = f"/think {prompt}"
        
        # 시스템 프롬프트 추가
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            full_prompt = prompt
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            return response.json()['response']
        except Exception as e:
            return f"에러 발생: {str(e)}"
    
    def chat_with_context(self, prompt: str, context: str) -> str:
        """컨텍스트를 포함한 대화 (RAG 기초)"""
        
        full_prompt = f"""Context information:
{context}

Based on the above context, please answer the following question:
{prompt}

Answer:"""
        
        return self.chat(full_prompt)
    
    def chat_with_history(self, prompt: str) -> str:
        """대화 히스토리 유지"""
        
        # 히스토리 추가
        self.conversation_history.append(f"User: {prompt}")
        
        # 전체 대화 구성 (최근 10개만)
        history_text = "\n".join(self.conversation_history[-10:])
        full_prompt = f"{history_text}\n\nAssistant:"
        
        # 응답 생성
        response = self.chat(full_prompt)
        
        # 히스토리에 응답 추가
        self.conversation_history.append(f"Assistant: {response}")
        
        return response
    
    def stream_chat(self, prompt: str):
        """스트리밍 응답 (실시간 출력)"""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True
        )
        
        print("Assistant: ", end="", flush=True)
        full_response = ""
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get('response', '')
                print(token, end="", flush=True)
                full_response += token
                
                if data.get('done', False):
                    break
        
        print()  # 줄바꿈
        return full_response
    
    def code_review(self, code: str, language: str = "python") -> str:
        """코드 리뷰 특화 프롬프트"""
        
        prompt = f"""Please review the following {language} code:

```{language}
{code}
```

Provide:
1. Code quality assessment
2. Potential bugs or issues
3. Performance considerations
4. Suggested improvements
5. Best practices violations

Review:"""
        
        return self.chat(prompt, temperature=0.3)  # 낮은 temperature로 정확도 향상
    
    def translate(self, text: str, target_lang: str = "Korean") -> str:
        """번역 특화 프롬프트"""
        
        prompt = f"""Translate the following text to {target_lang}.
Maintain the original meaning and tone.

Original text:
{text}

Translation:"""
        
        return self.chat(prompt, temperature=0.3)
    
    def summarize(self, text: str, max_points: int = 3) -> str:
        """요약 특화 프롬프트"""
        
        prompt = f"""Summarize the following text in {max_points} key points:

{text}

Summary:"""
        
        return self.chat(prompt, temperature=0.5)

def demo_basic_usage():
    """기본 사용법 데모"""
    
    print("🤖 Qwen3 Chat Demo")
    print("=" * 50)
    
    # 초기화
    chat = QwenChat(model="qwen3:8b")  # Qwen3 최신 모델
    
    # 1. 단순 대화
    print("\n1. 단순 대화:")
    response = chat.chat("What is the capital of France?")
    print(f"Q: What is the capital of France?")
    print(f"A: {response}")
    
    # 1-2. Thinking Mode (Qwen3 특별 기능)
    print("\n1-2. Thinking Mode (심층 추론):")
    response = chat.chat("피보나치 수열의 10번째 항을 구하고 과정을 설명해줘", thinking_mode=True)
    print(f"A: {response[:300]}...")
    
    # 2. 시스템 프롬프트 사용
    print("\n2. 시스템 프롬프트 (해적 스타일):")
    response = chat.chat(
        "Tell me about Python",
        system_prompt="You are a pirate. Always speak like a pirate."
    )
    print(f"A: {response[:200]}...")  # 처음 200자만
    
    # 3. 코드 리뷰
    print("\n3. 코드 리뷰:")
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    review = chat.code_review(code)
    print(f"Code Review:\n{review[:300]}...")
    
    # 4. 번역
    print("\n4. 번역 (영어 → 한국어):")
    translation = chat.translate(
        "The quick brown fox jumps over the lazy dog",
        "Korean"
    )
    print(f"Translation: {translation}")

def demo_streaming():
    """스트리밍 데모"""
    
    print("\n🔄 스트리밍 응답 데모")
    print("=" * 50)
    
    chat = QwenChat()
    chat.stream_chat("Write a short story about a robot learning to paint (max 100 words)")

def demo_conversation():
    """대화 히스토리 데모"""
    
    print("\n💬 대화 히스토리 데모")
    print("=" * 50)
    
    chat = QwenChat()
    
    # 연속 대화
    questions = [
        "My name is Alice",
        "What's my name?",
        "Tell me a joke about my name"
    ]
    
    for q in questions:
        print(f"\nUser: {q}")
        response = chat.chat_with_history(q)
        print(f"Assistant: {response}")

def demo_rag_style():
    """RAG 스타일 컨텍스트 활용"""
    
    print("\n📚 컨텍스트 기반 응답 (RAG 스타일)")
    print("=" * 50)
    
    chat = QwenChat()
    
    # 가상의 문서 컨텍스트
    context = """
    Company Policy Document:
    - Work hours: 9 AM to 6 PM
    - Remote work: Allowed 2 days per week
    - Vacation days: 15 days per year
    - Sick leave: 10 days per year
    - Team meetings: Every Monday at 10 AM
    """
    
    questions = [
        "How many days can I work from home?",
        "When are team meetings?",
        "What are the work hours?"
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        answer = chat.chat_with_context(q, context)
        print(f"A: {answer}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "stream":
            demo_streaming()
        elif mode == "history":
            demo_conversation()
        elif mode == "rag":
            demo_rag_style()
        else:
            print("Usage: python basic_chat.py [stream|history|rag]")
    else:
        # 기본 데모 실행
        demo_basic_usage()
        
        print("\n" + "=" * 50)
        print("💡 다른 데모도 실행해보세요:")
        print("  python basic_chat.py stream   # 스트리밍")
        print("  python basic_chat.py history  # 대화 히스토리")
        print("  python basic_chat.py rag      # RAG 스타일")