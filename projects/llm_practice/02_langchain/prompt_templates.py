"""
프롬프트 템플릿 고급 활용
실무에서 자주 사용하는 프롬프트 엔지니어링 패턴
"""

from langchain_community.llms import Ollama
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.chains import LLMChain
from typing import List, Dict, Any
import json

class AdvancedPromptTemplates:
    """고급 프롬프트 템플릿 패턴"""
    
    def __init__(self, model: str = "qwen3:8b"):
        self.llm = Ollama(model=model, temperature=0.7)
    
    def few_shot_learning(self, task: str, new_input: str) -> str:
        """Few-shot 학습 템플릿"""
        
        # 예제 데이터
        examples = [
            {
                "input": "The movie was fantastic!",
                "output": "Sentiment: Positive"
            },
            {
                "input": "I hated the service at this restaurant.",
                "output": "Sentiment: Negative"
            },
            {
                "input": "The weather is okay today.",
                "output": "Sentiment: Neutral"
            }
        ]
        
        # 예제 템플릿
        example_template = """
Input: {input}
Output: {output}
"""
        
        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template=example_template
        )
        
        # Few-shot 프롬프트
        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Analyze the sentiment of the text:",
            suffix="Input: {new_input}\nOutput:",
            input_variables=["new_input"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=few_shot_prompt)
        return chain.run(new_input=new_input)
    
    def dynamic_few_shot(self, examples: List[Dict], query: str) -> str:
        """동적 Few-shot (예제 자동 선택)"""
        
        example_prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template="Q: {question}\nA: {answer}"
        )
        
        # 길이 기반 예제 선택기
        example_selector = LengthBasedExampleSelector(
            examples=examples,
            example_prompt=example_prompt,
            max_length=200  # 최대 길이
        )
        
        dynamic_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix="Answer questions based on these examples:",
            suffix="Q: {query}\nA:",
            input_variables=["query"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=dynamic_prompt)
        return chain.run(query=query)
    
    def role_based_prompt(self, role: str, task: str) -> str:
        """역할 기반 프롬프트"""
        
        role_templates = {
            "teacher": {
                "system": "You are a patient teacher who explains concepts clearly with examples.",
                "style": "educational and encouraging"
            },
            "scientist": {
                "system": "You are a scientist who provides accurate, evidence-based information.",
                "style": "analytical and precise"
            },
            "developer": {
                "system": "You are an experienced developer who writes clean, efficient code.",
                "style": "practical with code examples"
            },
            "consultant": {
                "system": "You are a business consultant who provides strategic insights.",
                "style": "professional and solution-oriented"
            }
        }
        
        if role not in role_templates:
            role = "teacher"  # 기본값
        
        role_config = role_templates[role]
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(role_config["system"]),
            HumanMessagePromptTemplate.from_template(
                f"Please help with this task in a {role_config['style']} manner: {{task}}"
            )
        ])
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(task=task)
    
    def structured_output_prompt(self, topic: str) -> Dict:
        """구조화된 출력 프롬프트"""
        
        template = """
Analyze the topic "{topic}" and provide a structured response.

Format your response as a JSON object with the following structure:
{{
    "topic": "{topic}",
    "summary": "Brief 2-sentence summary",
    "key_points": ["point 1", "point 2", "point 3"],
    "pros": ["advantage 1", "advantage 2"],
    "cons": ["disadvantage 1", "disadvantage 2"],
    "applications": ["use case 1", "use case 2"],
    "complexity_level": "beginner|intermediate|advanced"
}}

Provide only the JSON object, no additional text:
"""
        
        prompt = PromptTemplate(
            input_variables=["topic"],
            template=template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(topic=topic)
        
        try:
            # JSON 파싱
            if '```json' in result:
                result = result.split('```json')[1].split('```')[0]
            elif '```' in result:
                result = result.split('```')[1].split('```')[0]
            
            return json.loads(result.strip())
        except:
            return {"error": "Failed to parse", "raw": result}
    
    def chain_of_thought_prompt(self, problem: str) -> str:
        """Chain of Thought (CoT) 프롬프트"""
        
        cot_template = """
Problem: {problem}

Let's solve this step by step.

Step 1: Understand the problem
What are we trying to solve?

Step 2: Identify key information
What information do we have?

Step 3: Plan the approach
How should we approach this?

Step 4: Execute the solution
Work through the solution.

Step 5: Verify the answer
Check if our answer makes sense.

Final Answer:
"""
        
        prompt = PromptTemplate(
            input_variables=["problem"],
            template=cot_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(problem=problem)
    
    def template_composition(self, product: str, audience: str) -> str:
        """템플릿 조합 (여러 템플릿 결합)"""
        
        # 제품 분석 템플릿
        product_analysis = PromptTemplate(
            input_variables=["product"],
            template="Analyze the key features of {product}:"
        )
        
        # 타겟 오디언스 템플릿
        audience_analysis = PromptTemplate(
            input_variables=["audience"],
            template="Describe the needs and preferences of {audience}:"
        )
        
        # 최종 조합 템플릿
        final_template = """
Product Analysis:
{product_analysis}

Target Audience:
{audience_analysis}

Based on the above analysis, create a marketing message that connects the product features with the audience needs:
"""
        
        # 각 부분 실행
        product_result = LLMChain(llm=self.llm, prompt=product_analysis).run(product=product)
        audience_result = LLMChain(llm=self.llm, prompt=audience_analysis).run(audience=audience)
        
        # 최종 조합
        final_prompt = PromptTemplate(
            input_variables=["product_analysis", "audience_analysis"],
            template=final_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=final_prompt)
        return chain.run(
            product_analysis=product_result,
            audience_analysis=audience_result
        )
    
    def conditional_prompt(self, input_text: str, mode: str = "auto") -> str:
        """조건부 프롬프트 (입력에 따라 다른 템플릿)"""
        
        # 입력 분류
        if mode == "auto":
            # 자동 모드 감지
            if "?" in input_text:
                mode = "question"
            elif any(word in input_text.lower() for word in ["analyze", "explain", "describe"]):
                mode = "explanation"
            elif any(word in input_text.lower() for word in ["code", "function", "program"]):
                mode = "code"
            else:
                mode = "general"
        
        templates = {
            "question": """
Question: {input}

Provide a clear, concise answer with supporting details:
""",
            "explanation": """
Topic: {input}

Provide a comprehensive explanation including:
1. Definition
2. Key concepts
3. Examples
4. Common misconceptions
""",
            "code": """
Request: {input}

Provide:
1. Clean, commented code
2. Explanation of the approach
3. Time/space complexity
4. Example usage
""",
            "general": """
Input: {input}

Response:
"""
        }
        
        template = templates.get(mode, templates["general"])
        prompt = PromptTemplate(input_variables=["input"], template=template)
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(input=input_text)
    
    def iterative_refinement_prompt(self, initial_text: str, iterations: int = 3) -> List[str]:
        """반복적 개선 프롬프트"""
        
        versions = [initial_text]
        
        refinement_template = """
Original text:
{text}

Improve this text by:
1. Making it more clear and concise
2. Fixing any errors
3. Enhancing the structure
4. Adding relevant details where needed

Improved version:
"""
        
        prompt = PromptTemplate(
            input_variables=["text"],
            template=refinement_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        current_text = initial_text
        for i in range(iterations):
            print(f"Refinement iteration {i+1}/{iterations}...")
            current_text = chain.run(text=current_text)
            versions.append(current_text)
        
        return versions

def demo_prompt_templates():
    """프롬프트 템플릿 데모"""
    
    print("📝 프롬프트 템플릿 데모")
    print("=" * 60)
    
    templates = AdvancedPromptTemplates()
    
    # 1. Few-shot Learning
    print("\n1. Few-shot Learning:")
    result = templates.few_shot_learning(
        "sentiment analysis",
        "The product quality is decent but the price is too high."
    )
    print(f"Result: {result}")
    
    # 2. 역할 기반 프롬프트
    print("\n2. Role-based Prompts:")
    roles = ["teacher", "developer"]
    task = "explain recursion"
    
    for role in roles:
        print(f"\n  As a {role}:")
        result = templates.role_based_prompt(role, task)
        print(f"  {result[:200]}...")
    
    # 3. Chain of Thought
    print("\n3. Chain of Thought:")
    problem = "If a train travels 120 km in 2 hours, how long will it take to travel 300 km?"
    result = templates.chain_of_thought_prompt(problem)
    print(result[:400] + "...")
    
    # 4. 구조화된 출력
    print("\n4. Structured Output:")
    result = templates.structured_output_prompt("machine learning")
    if isinstance(result, dict) and 'error' not in result:
        print(json.dumps(result, indent=2))
    else:
        print(result)

def demo_advanced_patterns():
    """고급 패턴 데모"""
    
    print("\n🚀 고급 프롬프트 패턴")
    print("=" * 60)
    
    templates = AdvancedPromptTemplates()
    
    # 1. 템플릿 조합
    print("\n1. Template Composition:")
    result = templates.template_composition(
        product="smartwatch",
        audience="fitness enthusiasts"
    )
    print(result[:300] + "...")
    
    # 2. 조건부 프롬프트
    print("\n2. Conditional Prompts:")
    inputs = [
        "What is the capital of France?",
        "Explain photosynthesis",
        "Write a function to reverse a string"
    ]
    
    for input_text in inputs:
        print(f"\n  Input: {input_text[:50]}...")
        result = templates.conditional_prompt(input_text)
        print(f"  Output: {result[:150]}...")
    
    # 3. 반복적 개선
    print("\n3. Iterative Refinement:")
    initial = "AI is technology that makes computers smart and can do things."
    versions = templates.iterative_refinement_prompt(initial, iterations=2)
    
    for i, version in enumerate(versions):
        print(f"\n  Version {i}:")
        print(f"  {version[:150]}...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "advanced":
            demo_advanced_patterns()
        else:
            print("Usage: python prompt_templates.py [advanced]")
    else:
        demo_prompt_templates()
        
        print("\n" + "=" * 60)
        print("💡 더 많은 데모:")
        print("  python prompt_templates.py advanced  # 고급 패턴")