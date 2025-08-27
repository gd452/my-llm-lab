"""
LangChain + Ollama 기본 셋업
실무에서 자주 사용하는 패턴들
"""

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.schema import BaseOutputParser
import json
from typing import Dict, List, Any

class BasicLangChainSetup:
    """LangChain 기본 셋업 클래스"""
    
    def __init__(self, model: str = "qwen3:8b"):
        self.llm = Ollama(
            model=model,
            temperature=0.7,
            num_predict=256,  # 최대 토큰 수
        )
        
    def simple_chain(self, topic: str) -> str:
        """가장 기본적인 체인"""
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="Explain {topic} in 3 bullet points:"
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(topic=topic)
    
    def chat_prompt_chain(self, user_input: str) -> str:
        """ChatPromptTemplate 사용 (더 구조화된 방식)"""
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant specialized in technology."),
            ("human", "{user_input}"),
        ])
        
        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        return chain.run(user_input=user_input)
    
    def sequential_chain(self, topic: str) -> Dict[str, str]:
        """순차적 체인 - 여러 단계 처리"""
        
        # 첫 번째 체인: 개념 설명
        explain_prompt = PromptTemplate(
            input_variables=["topic"],
            template="Explain {topic} in simple terms:"
        )
        explain_chain = LLMChain(llm=self.llm, prompt=explain_prompt)
        
        # 두 번째 체인: 실제 예시 생성
        example_prompt = PromptTemplate(
            input_variables=["explanation"],
            template="Based on this explanation: {explanation}\n\nProvide a real-world example:"
        )
        example_chain = LLMChain(llm=self.llm, prompt=example_prompt)
        
        # 체인 연결
        overall_chain = SimpleSequentialChain(
            chains=[explain_chain, example_chain],
            verbose=True
        )
        
        result = overall_chain.run(topic)
        
        return {
            "topic": topic,
            "final_output": result
        }
    
    def custom_output_parser(self, question: str):
        """커스텀 출력 파서 사용"""
        
        class BulletPointParser(BaseOutputParser):
            def parse(self, text: str) -> List[str]:
                """텍스트를 bullet point 리스트로 파싱"""
                lines = text.strip().split('\n')
                bullet_points = []
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('•') or 
                                line.startswith('-') or 
                                line.startswith('*') or
                                line[0].isdigit()):
                        # 불릿 포인트 제거하고 텍스트만 추출
                        clean_line = line.lstrip('•-*1234567890. ')
                        if clean_line:
                            bullet_points.append(clean_line)
                return bullet_points
        
        prompt = PromptTemplate(
            input_variables=["question"],
            template="Answer this question with 3 bullet points:\n{question}"
        )
        
        chain = LLMChain(
            llm=self.llm, 
            prompt=prompt,
            output_parser=BulletPointParser()
        )
        
        return chain.run(question=question)
    
    def json_output_chain(self, product: str) -> Dict:
        """JSON 형식으로 출력"""
        
        prompt = PromptTemplate(
            input_variables=["product"],
            template="""
Generate a product description in JSON format for: {product}

The JSON should have these fields:
- name: product name
- category: product category
- features: list of 3 key features
- price_range: estimated price range
- target_audience: who would buy this

Output only valid JSON, no additional text:
"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(product=product)
        
        # JSON 파싱 시도
        try:
            # JSON 블록 추출 (```json ... ``` 처리)
            if '```json' in result:
                result = result.split('```json')[1].split('```')[0]
            elif '```' in result:
                result = result.split('```')[1].split('```')[0]
            
            return json.loads(result.strip())
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON", "raw": result}
    
    def conditional_chain(self, text: str, task: str) -> str:
        """조건부 체인 - task에 따라 다른 처리"""
        
        task_prompts = {
            "summarize": "Summarize this text in 2 sentences: {text}",
            "translate": "Translate this to Korean: {text}",
            "analyze": "Analyze the sentiment and key points: {text}",
            "improve": "Rewrite this text to be more professional: {text}"
        }
        
        if task not in task_prompts:
            return f"Unknown task: {task}. Available: {list(task_prompts.keys())}"
        
        prompt = PromptTemplate(
            input_variables=["text"],
            template=task_prompts[task]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(text=text)
    
    def batch_processing(self, items: List[str], task_template: str) -> List[str]:
        """배치 처리 - 여러 아이템 한번에 처리"""
        
        prompt = PromptTemplate(
            input_variables=["item"],
            template=task_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        results = []
        for item in items:
            result = chain.run(item=item)
            results.append(result)
            print(f"Processed: {item[:30]}...")
        
        return results

def demo_basic_chains():
    """기본 체인 데모"""
    
    print("🔗 LangChain 기본 체인 데모")
    print("=" * 60)
    
    setup = BasicLangChainSetup()
    
    # 1. 간단한 체인
    print("\n1. Simple Chain:")
    result = setup.simple_chain("machine learning")
    print(result)
    
    # 2. Chat 프롬프트
    print("\n2. Chat Prompt Chain:")
    result = setup.chat_prompt_chain("What are the benefits of using Docker?")
    print(result)
    
    # 3. 순차 체인
    print("\n3. Sequential Chain:")
    result = setup.sequential_chain("blockchain")
    print(f"Final output: {result['final_output'][:200]}...")
    
    # 4. JSON 출력
    print("\n4. JSON Output:")
    result = setup.json_output_chain("wireless earbuds")
    if isinstance(result, dict) and 'error' not in result:
        print(json.dumps(result, indent=2))
    else:
        print(result)

def demo_advanced_features():
    """고급 기능 데모"""
    
    print("\n🚀 고급 기능 데모")
    print("=" * 60)
    
    setup = BasicLangChainSetup()
    
    # 1. 커스텀 파서
    print("\n1. Custom Output Parser:")
    result = setup.custom_output_parser("What are the benefits of cloud computing?")
    for i, point in enumerate(result, 1):
        print(f"  {i}. {point}")
    
    # 2. 조건부 처리
    print("\n2. Conditional Chain:")
    text = "Artificial intelligence is transforming industries worldwide."
    
    tasks = ["summarize", "translate", "analyze"]
    for task in tasks:
        print(f"\n  Task: {task}")
        result = setup.conditional_chain(text, task)
        print(f"  Result: {result[:150]}...")
    
    # 3. 배치 처리
    print("\n3. Batch Processing:")
    items = [
        "Python programming",
        "Data science",
        "Cloud computing"
    ]
    template = "Write a one-line definition for: {item}"
    results = setup.batch_processing(items, template)
    for item, result in zip(items, results):
        print(f"\n  {item}:")
        print(f"  → {result}")

def demo_error_handling():
    """에러 처리 데모"""
    
    print("\n⚠️ 에러 처리 데모")
    print("=" * 60)
    
    setup = BasicLangChainSetup()
    
    # Ollama 연결 테스트
    try:
        result = setup.simple_chain("test")
        print("✅ Ollama 연결 성공")
    except Exception as e:
        print(f"❌ Ollama 연결 실패: {e}")
        print("해결방법:")
        print("1. ollama serve 실행 확인")
        print("2. ollama pull qwen3:8b 실행")
        print("3. http://localhost:11434 접속 테스트")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "advanced":
            demo_advanced_features()
        elif mode == "error":
            demo_error_handling()
        else:
            print("Usage: python basic_setup.py [advanced|error]")
    else:
        # 기본 데모
        demo_basic_chains()
        
        print("\n" + "=" * 60)
        print("💡 다른 데모:")
        print("  python basic_setup.py advanced  # 고급 기능")
        print("  python basic_setup.py error     # 에러 처리")