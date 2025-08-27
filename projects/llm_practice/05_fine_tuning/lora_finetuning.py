"""
LoRA (Low-Rank Adaptation) 파인튜닝 예제
효율적인 LLM 파인튜닝 기법 실습
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset
import json
from typing import List, Dict, Any, Optional
import numpy as np

class LoRAFineTuner:
    """LoRA 파인튜닝 클래스"""
    
    def __init__(self, 
                 model_name: str = "microsoft/phi-2",  # 작은 모델로 실습
                 use_gpu: bool = torch.cuda.is_available()):
        
        self.model_name = model_name
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드
        self.model = None
        self.peft_model = None
        
    def load_model(self, load_in_8bit: bool = False):
        """모델 로드 및 LoRA 설정"""
        
        print(f"Loading model: {self.model_name}")
        
        # 모델 로드 설정
        if load_in_8bit:
            # 8-bit 양자화 (메모리 절약)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=True,
                device_map="auto"
            )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
        
        print(f"Model loaded: {self.model.__class__.__name__}")
        
    def setup_lora(self, 
                   r: int = 8,
                   lora_alpha: int = 32,
                   lora_dropout: float = 0.1,
                   target_modules: Optional[List[str]] = None):
        """LoRA 구성 설정"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # LoRA 설정
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,  # LoRA rank
            lora_alpha=lora_alpha,  # LoRA scaling
            lora_dropout=lora_dropout,
            target_modules=target_modules or ["q_proj", "v_proj"],  # 타겟 레이어
        )
        
        # PEFT 모델 생성
        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()
        
        return self.peft_model
    
    def prepare_dataset(self, data: List[Dict[str, str]]) -> Dataset:
        """데이터셋 준비"""
        
        def tokenize_function(examples):
            # 입력과 출력을 결합
            texts = []
            for instruction, output in zip(examples['instruction'], examples['output']):
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                texts.append(text)
            
            # 토크나이징
            model_inputs = self.tokenizer(
                texts,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # 레이블 설정 (입력과 동일)
            model_inputs["labels"] = model_inputs["input_ids"].clone()
            
            return model_inputs
        
        # Dataset 생성
        dataset = Dataset.from_dict({
            'instruction': [d['instruction'] for d in data],
            'output': [d['output'] for d in data]
        })
        
        # 토크나이징
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(self, 
              train_dataset: Dataset,
              eval_dataset: Optional[Dataset] = None,
              output_dir: str = "./lora_model",
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 2e-4):
        """모델 훈련"""
        
        if self.peft_model is None:
            raise ValueError("LoRA not configured. Call setup_lora() first.")
        
        # 훈련 인자 설정
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset else "no",
            learning_rate=learning_rate,
            fp16=self.device == "cuda",
            push_to_hub=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer 초기화
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # 훈련 시작
        print("Starting training...")
        trainer.train()
        
        # 모델 저장
        print(f"Saving model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer
    
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """텍스트 생성"""
        
        model = self.peft_model if self.peft_model else self.model
        
        if model is None:
            raise ValueError("No model loaded")
        
        # 프롬프트 토크나이징
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

class DatasetCreator:
    """파인튜닝용 데이터셋 생성"""
    
    @staticmethod
    def create_instruction_dataset(domain: str = "python") -> List[Dict[str, str]]:
        """인스트럭션 데이터셋 생성"""
        
        datasets = {
            "python": [
                {
                    "instruction": "Write a Python function to calculate factorial",
                    "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
                },
                {
                    "instruction": "Explain list comprehension in Python",
                    "output": "List comprehension is a concise way to create lists. It consists of brackets containing an expression followed by a for clause. Example: [x**2 for x in range(10)]"
                },
                {
                    "instruction": "How to handle exceptions in Python?",
                    "output": "Use try-except blocks. The try block contains code that might raise an exception, and except block handles it. Example: try: risky_operation() except Exception as e: handle_error(e)"
                }
            ],
            "sql": [
                {
                    "instruction": "Write a SQL query to find duplicate records",
                    "output": "SELECT column_name, COUNT(*) FROM table_name GROUP BY column_name HAVING COUNT(*) > 1;"
                },
                {
                    "instruction": "Explain JOIN types in SQL",
                    "output": "SQL has several JOIN types: INNER JOIN returns matching records, LEFT JOIN returns all from left table, RIGHT JOIN returns all from right table, FULL JOIN returns all records when there's a match in either table."
                }
            ],
            "general": [
                {
                    "instruction": "Explain machine learning in simple terms",
                    "output": "Machine learning is teaching computers to learn from examples instead of being explicitly programmed. Like showing many pictures of cats to teach recognition."
                },
                {
                    "instruction": "What is the difference between AI and ML?",
                    "output": "AI is the broader concept of machines being able to carry out smart tasks. ML is a subset of AI that focuses on learning from data to improve performance."
                }
            ]
        }
        
        return datasets.get(domain, datasets["general"])
    
    @staticmethod
    def create_qa_dataset() -> List[Dict[str, str]]:
        """Q&A 데이터셋 생성"""
        
        return [
            {
                "instruction": "Q: What is Docker?\nA:",
                "output": "Docker is a platform for developing, shipping, and running applications in containers. Containers package code and dependencies together."
            },
            {
                "instruction": "Q: How does Git work?\nA:",
                "output": "Git is a distributed version control system. It tracks changes in files, allows collaboration, and maintains history through commits, branches, and merges."
            },
            {
                "instruction": "Q: What is REST API?\nA:",
                "output": "REST API is an architectural style for web services using HTTP methods (GET, POST, PUT, DELETE) to perform operations on resources identified by URLs."
            }
        ]

def demo_lora_setup():
    """LoRA 설정 데모"""
    
    print("🔧 LoRA 파인튜닝 설정 데모")
    print("=" * 60)
    
    # 파인튜너 초기화
    finetuner = LoRAFineTuner(model_name="microsoft/phi-2")
    
    print("\n1. 모델 로드:")
    finetuner.load_model(load_in_8bit=False)
    
    print("\n2. LoRA 설정:")
    finetuner.setup_lora(
        r=8,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.1
    )
    
    print("\n3. 데이터셋 준비:")
    dataset_creator = DatasetCreator()
    training_data = dataset_creator.create_instruction_dataset("python")
    
    train_dataset = finetuner.prepare_dataset(training_data)
    print(f"  데이터셋 크기: {len(train_dataset)}")
    print(f"  첫 번째 샘플 키: {list(train_dataset[0].keys())}")
    
    print("\n✅ LoRA 파인튜닝 준비 완료!")
    print("실제 훈련을 시작하려면 'python lora_finetuning.py train' 실행")

def demo_training():
    """실제 훈련 데모 (주의: GPU와 시간 필요)"""
    
    print("🚀 LoRA 파인튜닝 훈련 데모")
    print("=" * 60)
    print("⚠️  주의: 이 데모는 GPU와 상당한 시간이 필요합니다!")
    
    # 간단한 예제만 실행
    print("\n훈련 과정 시뮬레이션:")
    print("1. 모델 로드 ✓")
    print("2. LoRA 어댑터 적용 ✓")
    print("3. 데이터셋 준비 ✓")
    print("4. 훈련 시작...")
    print("   Epoch 1/3: loss=2.34")
    print("   Epoch 2/3: loss=1.89")
    print("   Epoch 3/3: loss=1.45")
    print("5. 모델 저장 ✓")
    
    print("\n실제 훈련 코드:")
    print("""
    finetuner = LoRAFineTuner()
    finetuner.load_model()
    finetuner.setup_lora()
    
    train_data = DatasetCreator.create_instruction_dataset()
    train_dataset = finetuner.prepare_dataset(train_data)
    
    trainer = finetuner.train(
        train_dataset=train_dataset,
        num_epochs=3,
        batch_size=4
    )
    """)

def explain_lora_concepts():
    """LoRA 개념 설명"""
    
    print("\n📚 LoRA (Low-Rank Adaptation) 개념")
    print("=" * 60)
    
    concepts = {
        "1. LoRA란?": """
    - 대규모 언어 모델을 효율적으로 파인튜닝하는 기법
    - 원본 모델 가중치는 고정하고 작은 어댑터만 학습
    - 메모리와 연산량을 크게 줄임 (약 10,000배 적은 파라미터)
        """,
        
        "2. 핵심 원리": """
    - 가중치 업데이트를 저차원 행렬의 곱으로 분해
    - W' = W + BA (B와 A는 작은 행렬)
    - Rank r이 작을수록 파라미터 수 감소
        """,
        
        "3. 장점": """
    - 적은 GPU 메모리 사용
    - 빠른 훈련 속도
    - 여러 태스크에 대한 어댑터 교체 가능
    - 원본 모델 성능 유지
        """,
        
        "4. 주요 하이퍼파라미터": """
    - r (rank): LoRA 행렬의 차원 (일반적으로 4-64)
    - alpha: LoRA 스케일링 파라미터
    - dropout: 과적합 방지
    - target_modules: LoRA를 적용할 레이어
        """,
        
        "5. 사용 사례": """
    - 도메인 특화 모델 생성
    - 다국어 지원 추가
    - 특정 작업 스타일 학습
    - 개인화된 AI 어시스턴트
        """
    }
    
    for title, content in concepts.items():
        print(f"\n{title}")
        print(content)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "train":
            demo_training()
        elif mode == "concepts":
            explain_lora_concepts()
        else:
            print("Usage: python lora_finetuning.py [train|concepts]")
    else:
        demo_lora_setup()
        
        print("\n" + "=" * 60)
        print("💡 더 많은 정보:")
        print("  python lora_finetuning.py train     # 훈련 시뮬레이션")
        print("  python lora_finetuning.py concepts  # LoRA 개념 설명")