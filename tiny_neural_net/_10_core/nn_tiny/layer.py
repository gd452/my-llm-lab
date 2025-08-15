"""
🔲 Layer: 뉴런들의 집합

이 파일에서 구현할 것:
1. 여러 뉴런을 하나의 레이어로 조직화
2. 벡터 입력 → 벡터 출력
3. 모든 뉴런의 파라미터 관리

개념:
    Layer는 같은 입력을 공유하는 뉴런들의 집합입니다.
    각 뉴런은 독립적인 가중치와 편향을 가집니다.
    
    입력: [x1, x2, ..., xn]
    출력: [neuron1(x), neuron2(x), ..., neuronm(x)]
"""

from .neuron import Neuron


class Layer:
    """
    신경망의 한 층(Layer)
    
    여러 개의 뉴런을 포함하며, 각 뉴런은:
    - 같은 입력을 받지만
    - 다른 가중치를 가지고
    - 다른 출력을 생성합니다
    
    Example:
        >>> layer = Layer(3, 2)  # 3차원 입력, 2개 뉴런
        >>> x = [1.0, 2.0, 3.0]
        >>> outputs = layer(x)   # 2개의 출력
    """
    
    def __init__(self, nin: int, nout: int, **kwargs):
        """
        레이어 초기화
        
        Args:
            nin: 입력 차원 수
            nout: 출력 차원 수 (뉴런 개수)
            **kwargs: Neuron에 전달할 추가 인자 (예: nonlin)
        
        TODO:
        1. nout개의 Neuron을 생성하여 self.neurons에 저장
        2. 각 뉴런은 nin개의 입력을 받도록 설정
        
        힌트:
            self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
        """
        # TODO: nout개의 뉴런 생성
        self.neurons = None  # TODO: 구현
    
    def __call__(self, x):
        """
        Forward pass: 모든 뉴런에 입력 전달
        
        Args:
            x: 입력 (리스트 또는 Value 객체들)
        
        Returns:
            각 뉴런의 출력 리스트
            
        TODO:
        1. 각 뉴런에 x를 입력으로 전달
        2. 모든 출력을 리스트로 반환
        
        힌트:
            return [neuron(x) for neuron in self.neurons]
        """
        # TODO: 모든 뉴런의 출력 계산
        return None  # TODO: 구현
    
    def parameters(self):
        """
        레이어의 모든 파라미터 반환
        
        Returns:
            모든 뉴런의 파라미터를 하나의 리스트로
            
        TODO:
        1. 각 뉴런의 parameters()를 호출
        2. 모든 파라미터를 하나의 flat list로 만들기
        
        힌트:
            params = []
            for neuron in self.neurons:
                params.extend(neuron.parameters())
            return params
            
            # 또는 한 줄로:
            return [p for neuron in self.neurons for p in neuron.parameters()]
        """
        # TODO: 모든 뉴런의 파라미터 수집
        return []  # TODO: 구현
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


# 테스트 코드
if __name__ == "__main__":
    print("🧪 Layer 테스트")
    print("-" * 50)
    
    # TODO 구현 후 테스트
    """
    # 3차원 입력, 2개 출력
    layer = Layer(3, 2)
    print(f"레이어 생성: {layer}")
    
    # 입력 준비
    from tiny_autograd_project._10_core.autograd_tiny.value import Value
    x = [Value(1.0), Value(2.0), Value(3.0)]
    
    # Forward pass
    outputs = layer(x)
    print(f"입력: {[xi.data for xi in x]}")
    print(f"출력: {[o.data for o in outputs]}")
    
    # 파라미터 확인
    params = layer.parameters()
    print(f"총 파라미터 개수: {len(params)}")
    # 예상: 3*2 (가중치) + 2 (편향) = 8개
    """
    print("TODO: layer.py 구현 필요!")