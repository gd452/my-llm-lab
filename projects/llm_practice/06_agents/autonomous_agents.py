"""
자율 에이전트 시스템
ReAct, Tool Use, Memory를 활용한 AI 에이전트
"""

import json
import requests
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime

class ActionType(Enum):
    """에이전트 액션 타입"""
    THINK = "think"
    ACT = "act"
    OBSERVE = "observe"
    ANSWER = "answer"

@dataclass
class AgentAction:
    """에이전트 액션"""
    type: ActionType
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class Tool:
    """에이전트가 사용할 수 있는 도구"""
    
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
    
    def run(self, **kwargs) -> str:
        """도구 실행"""
        try:
            return str(self.func(**kwargs))
        except Exception as e:
            return f"Error: {str(e)}"

class ReActAgent:
    """ReAct (Reasoning + Acting) 에이전트"""
    
    def __init__(self, 
                 llm_model: str = "qwen3:8b",
                 max_iterations: int = 5):
        
        self.llm_model = llm_model
        self.llm_base_url = "http://localhost:11434"
        self.max_iterations = max_iterations
        
        # 도구 저장소
        self.tools: Dict[str, Tool] = {}
        
        # 메모리 (대화 히스토리)
        self.memory: List[AgentAction] = []
        
        # 기본 도구 등록
        self._register_default_tools()
    
    def _register_default_tools(self):
        """기본 도구 등록"""
        
        # 계산기
        def calculator(expression: str) -> float:
            """수학 계산"""
            try:
                return eval(expression, {"__builtins__": {}}, {})
            except:
                return "Invalid expression"
        
        # 검색 (시뮬레이션)
        def search(query: str) -> str:
            """웹 검색 시뮬레이션"""
            mock_results = {
                "python": "Python is a high-level programming language",
                "machine learning": "ML is a subset of AI that enables systems to learn",
                "docker": "Docker is a containerization platform",
                "default": "No specific information found"
            }
            
            for key, value in mock_results.items():
                if key.lower() in query.lower():
                    return value
            return mock_results["default"]
        
        # 날짜/시간
        def get_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
            """현재 날짜/시간 반환"""
            return datetime.now().strftime(format)
        
        # 도구 등록
        self.register_tool("calculator", "Perform mathematical calculations", calculator)
        self.register_tool("search", "Search for information", search)
        self.register_tool("datetime", "Get current date and time", get_datetime)
    
    def register_tool(self, name: str, description: str, func: Callable):
        """도구 등록"""
        self.tools[name] = Tool(name, description, func)
    
    def _call_llm(self, prompt: str) -> str:
        """LLM 호출"""
        try:
            response = requests.post(
                f"{self.llm_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"LLM Error: {response.status_code}"
        except Exception as e:
            return f"LLM Connection Error: {str(e)}"
    
    def _create_react_prompt(self, question: str) -> str:
        """ReAct 프롬프트 생성"""
        
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        prompt = f"""You are an AI assistant that uses the ReAct framework to solve problems.
You can use these tools:
{tools_desc}

Follow this format:
Thought: Think about what you need to do
Action: tool_name
Action Input: {{"param": "value"}}
Observation: [Tool output will appear here]
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: [Your final answer]

Question: {question}

Let's think step by step.
"""
        
        # 이전 액션 추가
        if self.memory:
            prompt += "\nPrevious actions:\n"
            for action in self.memory[-3:]:  # 최근 3개만
                if action.type == ActionType.THINK:
                    prompt += f"Thought: {action.content}\n"
                elif action.type == ActionType.ACT:
                    prompt += f"Action: {action.tool_name}\n"
                    prompt += f"Action Input: {json.dumps(action.tool_input)}\n"
                elif action.type == ActionType.OBSERVE:
                    prompt += f"Observation: {action.content}\n"
        
        return prompt
    
    def _parse_llm_output(self, output: str) -> AgentAction:
        """LLM 출력 파싱"""
        
        # Thought 추출
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", output, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
            
            # Final Answer 체크
            if "Final Answer:" in output:
                answer_match = re.search(r"Final Answer:\s*(.+)", output, re.DOTALL)
                if answer_match:
                    return AgentAction(
                        type=ActionType.ANSWER,
                        content=answer_match.group(1).strip()
                    )
            
            # Action 추출
            action_match = re.search(r"Action:\s*(.+?)(?=Action Input:|$)", output, re.DOTALL)
            if action_match:
                tool_name = action_match.group(1).strip()
                
                # Action Input 추출
                input_match = re.search(r"Action Input:\s*(.+?)(?=Observation:|$)", output, re.DOTALL)
                if input_match:
                    try:
                        tool_input = json.loads(input_match.group(1).strip())
                    except:
                        # JSON 파싱 실패 시 문자열로 처리
                        tool_input = {"input": input_match.group(1).strip()}
                    
                    return AgentAction(
                        type=ActionType.ACT,
                        content=thought,
                        tool_name=tool_name,
                        tool_input=tool_input
                    )
            
            return AgentAction(type=ActionType.THINK, content=thought)
        
        # 파싱 실패 시
        return AgentAction(type=ActionType.THINK, content=output)
    
    def run(self, question: str) -> str:
        """에이전트 실행"""
        
        print(f"\n🤖 ReAct Agent Processing: {question}")
        print("=" * 60)
        
        self.memory = []
        
        for i in range(self.max_iterations):
            print(f"\n[Iteration {i+1}]")
            
            # ReAct 프롬프트 생성
            prompt = self._create_react_prompt(question)
            
            # LLM 호출
            llm_output = self._call_llm(prompt)
            print(f"LLM Output: {llm_output[:200]}...")
            
            # 출력 파싱
            action = self._parse_llm_output(llm_output)
            self.memory.append(action)
            
            # 액션 처리
            if action.type == ActionType.ANSWER:
                print(f"\n✅ Final Answer: {action.content}")
                return action.content
            
            elif action.type == ActionType.ACT:
                print(f"🔧 Using tool: {action.tool_name}")
                print(f"   Input: {action.tool_input}")
                
                # 도구 실행
                if action.tool_name in self.tools:
                    result = self.tools[action.tool_name].run(**action.tool_input)
                    print(f"   Output: {result}")
                    
                    # Observation 추가
                    observation = AgentAction(
                        type=ActionType.OBSERVE,
                        content=result
                    )
                    self.memory.append(observation)
                else:
                    print(f"   Error: Tool '{action.tool_name}' not found")
            
            elif action.type == ActionType.THINK:
                print(f"💭 Thinking: {action.content[:100]}...")
        
        # 최대 반복 도달
        return "Could not find answer within iteration limit"

class MemoryAgent:
    """메모리 기능이 있는 에이전트"""
    
    def __init__(self):
        # 단기 메모리 (현재 대화)
        self.short_term_memory: List[Dict[str, str]] = []
        
        # 장기 메모리 (중요 정보)
        self.long_term_memory: Dict[str, Any] = {}
        
        # 작업 메모리 (현재 작업 컨텍스트)
        self.working_memory: Dict[str, Any] = {}
    
    def remember(self, key: str, value: Any, memory_type: str = "long"):
        """정보 저장"""
        if memory_type == "long":
            self.long_term_memory[key] = value
        elif memory_type == "working":
            self.working_memory[key] = value
    
    def recall(self, key: str, memory_type: str = "long") -> Optional[Any]:
        """정보 회상"""
        if memory_type == "long":
            return self.long_term_memory.get(key)
        elif memory_type == "working":
            return self.working_memory.get(key)
        return None
    
    def add_conversation(self, role: str, message: str):
        """대화 추가"""
        self.short_term_memory.append({
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # 메모리 크기 제한 (최근 10개)
        if len(self.short_term_memory) > 10:
            self.short_term_memory = self.short_term_memory[-10:]
    
    def get_context(self) -> str:
        """현재 컨텍스트 반환"""
        context_parts = []
        
        # 대화 히스토리
        if self.short_term_memory:
            history = "\n".join([
                f"{item['role']}: {item['message']}"
                for item in self.short_term_memory[-5:]
            ])
            context_parts.append(f"Recent conversation:\n{history}")
        
        # 작업 메모리
        if self.working_memory:
            working = "\n".join([
                f"- {k}: {v}"
                for k, v in self.working_memory.items()
            ])
            context_parts.append(f"Current context:\n{working}")
        
        return "\n\n".join(context_parts)

class MultiAgentSystem:
    """다중 에이전트 시스템"""
    
    def __init__(self):
        self.agents: Dict[str, ReActAgent] = {}
        self.coordinator_memory = MemoryAgent()
    
    def add_agent(self, name: str, agent: ReActAgent):
        """에이전트 추가"""
        self.agents[name] = agent
    
    def delegate_task(self, task: str, agent_name: str) -> str:
        """태스크 위임"""
        if agent_name in self.agents:
            result = self.agents[agent_name].run(task)
            self.coordinator_memory.add_conversation(agent_name, result)
            return result
        return f"Agent {agent_name} not found"
    
    def collaborate(self, complex_task: str) -> str:
        """에이전트 협업"""
        
        # 태스크 분해 (간단한 예제)
        subtasks = self._decompose_task(complex_task)
        
        results = []
        for subtask, agent_name in subtasks:
            print(f"\n📋 Delegating to {agent_name}: {subtask}")
            result = self.delegate_task(subtask, agent_name)
            results.append(result)
        
        # 결과 통합
        return self._integrate_results(results)
    
    def _decompose_task(self, task: str) -> List[tuple]:
        """태스크 분해 (간단한 규칙 기반)"""
        subtasks = []
        
        if "calculate" in task.lower():
            subtasks.append((task, "calculator_agent"))
        if "search" in task.lower() or "find" in task.lower():
            subtasks.append((task, "search_agent"))
        
        if not subtasks:
            subtasks.append((task, "general_agent"))
        
        return subtasks
    
    def _integrate_results(self, results: List[str]) -> str:
        """결과 통합"""
        return "\n\n".join([
            f"Result {i+1}: {result}"
            for i, result in enumerate(results)
        ])

def demo_react_agent():
    """ReAct 에이전트 데모"""
    
    print("🤖 ReAct Agent 데모")
    print("=" * 60)
    
    agent = ReActAgent()
    
    # 테스트 질문들
    questions = [
        "What is 25 * 4 + 10?",
        "What is the current date?",
        "Search for information about Python programming"
    ]
    
    for question in questions:
        answer = agent.run(question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print("-" * 40)

def demo_memory_agent():
    """메모리 에이전트 데모"""
    
    print("\n🧠 Memory Agent 데모")
    print("=" * 60)
    
    memory = MemoryAgent()
    
    # 정보 저장
    memory.remember("user_name", "Alice")
    memory.remember("favorite_color", "blue")
    memory.add_conversation("user", "Hello, I'm Alice")
    memory.add_conversation("assistant", "Nice to meet you, Alice!")
    
    # 정보 회상
    print(f"Recalled name: {memory.recall('user_name')}")
    print(f"Recalled color: {memory.recall('favorite_color')}")
    
    # 컨텍스트 출력
    print(f"\nCurrent context:\n{memory.get_context()}")

def explain_agent_concepts():
    """에이전트 개념 설명"""
    
    print("\n📚 AI 에이전트 핵심 개념")
    print("=" * 60)
    
    concepts = {
        "1. ReAct (Reasoning + Acting)": """
    - 추론과 행동을 결합한 프레임워크
    - Thought → Action → Observation 사이클
    - 도구 사용과 추론을 통합
    - Chain-of-Thought와 도구 사용의 결합
        """,
        
        "2. 도구 사용 (Tool Use)": """
    - 계산기, 검색, API 등 외부 도구 활용
    - 에이전트의 능력 확장
    - 정확한 정보와 계산 제공
    - Function Calling과 유사
        """,
        
        "3. 메모리 시스템": """
    - 단기 메모리: 현재 대화 컨텍스트
    - 장기 메모리: 영구 저장 정보
    - 작업 메모리: 현재 작업 관련 정보
    - 에피소드 메모리: 과거 경험
        """,
        
        "4. 다중 에이전트 시스템": """
    - 특화된 에이전트들의 협업
    - 태스크 분해와 위임
    - 결과 통합과 조정
    - 복잡한 문제 해결
        """,
        
        "5. 자율성 레벨": """
    - Level 1: 단순 도구 실행
    - Level 2: 계획 수립과 실행
    - Level 3: 자기 반성과 개선
    - Level 4: 완전 자율 에이전트
        """
    }
    
    for title, content in concepts.items():
        print(f"\n{title}")
        print(content)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "memory":
            demo_memory_agent()
        elif mode == "concepts":
            explain_agent_concepts()
        else:
            print("Usage: python autonomous_agents.py [memory|concepts]")
    else:
        # ReAct 에이전트 데모 (시뮬레이션)
        print("🤖 ReAct Agent 데모 (시뮬레이션)")
        print("=" * 60)
        print("\n실제 LLM 연결 없이 에이전트 동작 시뮬레이션:")
        
        print("\nQ: What is 25 * 4 + 10?")
        print("💭 Thought: I need to calculate 25 * 4 + 10")
        print("🔧 Action: calculator")
        print("   Input: {'expression': '25 * 4 + 10'}")
        print("   Output: 110")
        print("✅ Final Answer: The result is 110")
        
        print("\n" + "=" * 60)
        print("💡 더 많은 데모:")
        print("  python autonomous_agents.py memory    # 메모리 시스템")
        print("  python autonomous_agents.py concepts  # 개념 설명")