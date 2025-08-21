"""
🏗️ MLP (Multi-Layer Perceptron): 다층 신경망

이 파일에서 구현할 것:
1. 여러 레이어를 순차적으로 연결
2. Deep Neural Network 구조
3. 전체 네트워크의 파라미터 관리

구조 예시 (2-4-4-1):
    입력층(2) → 은닉층1(4) → 은닉층2(4) → 출력층(1)
    
    [x1] ─┬─→ [h1] ─┬─→ [h5] ─┬─→ [y]
    [x2] ─┼─→ [h2] ─┼─→ [h6] ─┘
          ├─→ [h3] ─┼─→ [h7]
          └─→ [h4] ─┴─→ [h8]
"""

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.layer import Layer

class MLP:
    """
    Multi-Layer Perceptron (다층 퍼셉트론)
    
    여러 개의 Layer를 순차적으로 연결한 신경망입니다.
    각 층의 출력이 다음 층의 입력이 됩니다.
    
    Example:
        >>> mlp = MLP(2, [4, 4, 1])  # 2-4-4-1 구조
        >>> x = [1.0, 0.5]
        >>> output = mlp(x)  # forward pass through all layers
    """
    
    def __init__(self, nin: int, nouts: list):
        """
        MLP 초기화
        
        Args:
            nin: 입력 차원
            nouts: 각 층의 뉴런 수 리스트
                  예: [4, 4, 1] = 3개 층 (4뉴런, 4뉴런, 1뉴런)
        
        TODO:
        1. 모든 층의 크기를 하나의 리스트로 만들기 [nin] + nouts
        2. 연속된 층들을 연결하여 Layer 객체들 생성
        3. 마지막 층을 제외하고는 nonlin=True 설정
        
        힌트:
            sz = [nin] + nouts  # 예: [2, 4, 4, 1]
            self.layers = []
            for i in range(len(nouts)):
                # 마지막 층은 선형, 나머지는 비선형
                nonlin = (i != len(nouts) - 1)
                self.layers.append(Layer(sz[i], sz[i+1], nonlin=nonlin))
        """
        sz = [nin] + nouts
        
        # TODO: 레이어들 생성
        self.layers = []  # TODO: 구현

        for i in range(len(nouts)):
            nonlin = (i != len(nouts) - 1)
            self.layers.append(Layer(sz[i], sz[i+1], nonlin=nonlin))
        
        # 디버깅용 정보
        self.architecture = sz
    
    def __call__(self, x):
        """
        Forward pass: 모든 레이어를 순차적으로 통과
        
        Args:
            x: 입력 (리스트 또는 Value 객체들)
        
        Returns:
            최종 출력 (단일 값이면 스칼라, 여러 개면 리스트)
            
        TODO:
        1. 첫 번째 레이어에 x 입력
        2. 각 레이어의 출력을 다음 레이어의 입력으로
        3. 마지막 레이어의 출력 반환
        
        힌트:
            for layer in self.layers:
                x = layer(x)
            return x[0] if len(x) == 1 else x
        """
        # TODO: 순차적으로 레이어 통과
        out = x
        # TODO: 구현
        for layer in self.layers:
            out = layer(out)
        
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        """
        전체 네트워크의 파라미터 반환
        
        Returns:
            모든 레이어의 파라미터를 하나의 리스트로
            
        TODO:
        1. 각 레이어의 parameters() 호출
        2. 모두 합쳐서 반환
        
        힌트:
        """
        # TODO: 모든 레이어의 파라미터 수집
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP({self.architecture})"
    
    def summary(self):
        """네트워크 구조 요약 출력"""
        print(f"\n🏗️ Network Architecture: {self.architecture}")
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            n_params = len(layer.parameters())
            total_params += n_params
            print(f"  Layer {i}: {self.architecture[i]} → {self.architecture[i+1]}, "
                  f"Parameters: {n_params}")
        
        print(f"  Total Parameters: {total_params}")
        return total_params


# 테스트 코드
if __name__ == "__main__":
    print("🧪 MLP 테스트")
    print("-" * 50)
    
    # TODO 구현 후 테스트
    # XOR를 위한 네트워크: 2-4-4-1
    mlp = MLP(2, [4, 4, 1])
    print(f"MLP 생성: {mlp}")
    
    # 네트워크 구조 확인
    mlp.summary()
    
    # 입력 준비
    from core.autograd import Value
    x = [Value(0.5), Value(0.5)]
    
    # Forward pass
    output = mlp(x)
    print(f"\n입력: {[xi.data for xi in x]}")
    print(f"출력: {output.data if hasattr(output, 'data') else output}")
    
    # 파라미터 확인
    params = mlp.parameters()
    print(f"\n총 파라미터: {len(params)}개")