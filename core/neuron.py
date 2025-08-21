"""
🧠 Neuron: 신경망의 가장 작은 단위

이 파일에서 구현할 것:
1. 단일 뉴런 (가중치, 편향, 활성화 함수)
2. Forward pass 계산
3. 파라미터 관리

수학적 표현:
    output = activation(Σ(wi * xi) + b)
    
여기서:
    - wi: i번째 가중치
    - xi: i번째 입력
    - b: 편향(bias)
    - activation: 활성화 함수 (tanh, relu 등)
"""

import random

if __name__ == "__main__":
    # 직접 실행할 때만 path 추가
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.autograd import Value


class Neuron:
    """
    단일 뉴런 구현
    
    뉴런은 생물학적 뉴런을 모방한 것으로:
    - 여러 입력을 받아서
    - 가중치를 곱하고
    - 편향을 더한 뒤
    - 활성화 함수를 통과시킵니다
    
    Example:
        >>> neuron = Neuron(3)  # 3개의 입력을 받는 뉴런
        >>> x = [Value(1.0), Value(2.0), Value(3.0)]
        >>> output = neuron(x)  # forward pass
    """
    
    def __init__(self, nin: int, nonlin: bool = True):
        """
        뉴런 초기화
        
        Args:
            nin: 입력 차원 수 (number of inputs)
            nonlin: 비선형 활성화 함수 사용 여부
        
        TODO: 
        1. self.w를 nin개의 랜덤 Value로 초기화 (-1 ~ 1 사이)
        2. self.b를 0으로 초기화된 Value로 설정
        3. self.nonlin 설정
        
        힌트:
            self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        """
        # TODO: 가중치 초기화 (nin개)
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        
        # TODO: 편향 초기화
        self.b = Value(0)
        
        # 비선형 활성화 여부
        self.nonlin = nonlin
    
    def __call__(self, x):
        """
        Forward pass 수행
        
        Args:
            x: 입력 리스트 (Value 객체들 또는 숫자들)
        
        Returns:
            활성화 함수를 통과한 출력 (Value 객체)
        
        TODO:
        1. 입력 x가 Value가 아니면 Value로 변환
        2. Σ(wi * xi) + b 계산
        3. nonlin이 True면 tanh() 적용
        
        힌트:
            # 내적 계산
            act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
            # 활성화 함수
            return act.tanh() if self.nonlin else act
        """
        # TODO: 입력을 Value로 변환 (필요시)
        x = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        
        # TODO: 가중합 계산: Σ(wi * xi) + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        
        # TODO: 활성화 함수 적용
        return act.tanh() if self.nonlin else act
    
    def parameters(self):
        """
        학습 가능한 파라미터 반환
        
        Returns:
            가중치와 편향을 포함한 리스트
            
        TODO: self.w와 self.b를 하나의 리스트로 반환
        """
        # TODO: 모든 파라미터 반환
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


# 테스트 코드
if __name__ == "__main__":
    print("🧪 Neuron 테스트")
    print("-" * 50)
    
    # TODO 구현 후 테스트
    # 2개 입력을 받는 뉴런 생성
    neuron = Neuron(2)
    print(f"뉴런 생성: {neuron}")
    
    # 입력 준비
    x = [Value(1.0), Value(0.5)]
    
    # Forward pass
    output = neuron(x)
    print(f"입력: {[xi.data for xi in x]}")
    print(f"출력: {output.data}")
    
    # 파라미터 확인
    params = neuron.parameters()
    print(f"파라미터 개수: {len(params)}")