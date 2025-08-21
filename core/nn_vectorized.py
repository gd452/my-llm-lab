"""
🚀 Vectorized Neural Network: 벡터화된 신경망

이 파일에서 구현할 것:
1. 벡터화된 Linear Layer
2. 벡터화된 MLP
3. 효율적인 forward/backward pass

NumPy의 Broadcasting을 활용하여
배치 단위로 효율적인 연산을 수행합니다.
"""

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.tensor_ops import relu, sigmoid, tanh, softmax


class LinearLayer:
    """
    벡터화된 완전 연결층
    
    배치 단위로 효율적인 행렬 연산을 수행합니다.
    """
    
    def __init__(self, input_dim, output_dim, activation='relu'):
        """
        초기화
        
        Args:
            input_dim: 입력 차원
            output_dim: 출력 차원
            activation: 활성화 함수 ('relu', 'sigmoid', 'tanh', 'none')
        """
        # Xavier/He 초기화
        if activation == 'relu':
            # He 초기화 (ReLU에 적합)
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        else:
            # Xavier 초기화
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(1 / input_dim)
        
        self.b = np.zeros(output_dim)
        
        # 활성화 함수 설정
        self.activation = activation
        self.activation_fn = {
            'relu': relu,
            'sigmoid': sigmoid,
            'tanh': tanh,
            'none': lambda x: x
        }.get(activation, relu)
        
        # 캐시 (역전파용)
        self.cache = {}
        
        # 그래디언트
        self.dW = None
        self.db = None
    
    def forward(self, X):
        """
        순전파
        
        Args:
            X: 입력 (batch_size, input_dim)
        
        Returns:
            출력 (batch_size, output_dim)
        
        TODO: 구현
        1. 선형 변환: Z = XW + b
        2. 활성화 함수 적용
        3. 역전파를 위해 값들 저장
        """
        # 선형 변환
        Z = X @ self.W + self.b  # Broadcasting으로 bias 추가
        
        # 활성화
        A = self.activation_fn(Z)
        
        # 역전파를 위해 저장
        self.cache = {
            'X': X,
            'Z': Z,
            'A': A
        }
        
        return A
    
    def backward(self, dA):
        """
        역전파
        
        Args:
            dA: 출력에 대한 그래디언트 (batch_size, output_dim)
        
        Returns:
            입력에 대한 그래디언트 (batch_size, input_dim)
        
        TODO: 구현
        """
        X = self.cache['X']
        Z = self.cache['Z']
        batch_size = X.shape[0]
        
        # 활성화 함수의 그래디언트
        if self.activation == 'relu':
            dZ = dA * (Z > 0)
        elif self.activation == 'sigmoid':
            A = self.cache['A']
            dZ = dA * A * (1 - A)
        elif self.activation == 'tanh':
            A = self.cache['A']
            dZ = dA * (1 - A ** 2)
        else:  # none
            dZ = dA
        
        # 파라미터 그래디언트
        self.dW = (X.T @ dZ) / batch_size
        self.db = np.sum(dZ, axis=0) / batch_size
        
        # 입력 그래디언트
        dX = dZ @ self.W.T
        
        return dX
    
    def update_params(self, learning_rate):
        """파라미터 업데이트"""
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
    
    def get_params(self):
        """파라미터 반환"""
        return {'W': self.W, 'b': self.b}
    
    def set_params(self, params):
        """파라미터 설정"""
        self.W = params['W']
        self.b = params['b']


class MLPVectorized:
    """
    벡터화된 Multi-Layer Perceptron
    
    여러 LinearLayer를 조합한 깊은 신경망
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        """
        초기화
        
        Args:
            input_dim: 입력 차원
            hidden_dims: 은닉층 차원들 (리스트)
            output_dim: 출력 차원
            activation: 은닉층 활성화 함수
        
        Example:
            >>> mlp = MLPVectorized(784, [256, 128], 10)
            >>> # 784 -> 256 -> 128 -> 10 구조
        """
        self.layers = []
        
        # 모든 차원을 하나의 리스트로
        dims = [input_dim] + hidden_dims + [output_dim]
        
        # 레이어 생성
        for i in range(len(dims) - 1):
            # 마지막 레이어는 활성화 함수 없음
            if i == len(dims) - 2:
                layer = LinearLayer(dims[i], dims[i+1], activation='none')
            else:
                layer = LinearLayer(dims[i], dims[i+1], activation=activation)
            
            self.layers.append(layer)
    
    def forward(self, X):
        """
        순전파
        
        Args:
            X: 입력 (batch_size, input_dim)
        
        Returns:
            출력 (batch_size, output_dim)
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, dout):
        """
        역전파
        
        Args:
            dout: 출력 그래디언트 (batch_size, output_dim)
        
        Returns:
            입력 그래디언트 (batch_size, input_dim)
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def update_params(self, learning_rate):
        """모든 레이어의 파라미터 업데이트"""
        for layer in self.layers:
            layer.update_params(learning_rate)
    
    def get_all_params(self):
        """모든 파라미터 반환"""
        return [layer.get_params() for layer in self.layers]
    
    def predict(self, X):
        """
        예측 (추론 모드)
        
        Args:
            X: 입력 데이터
        
        Returns:
            예측 결과
        """
        return self.forward(X)
    
    def predict_proba(self, X):
        """
        확률 예측 (분류용)
        
        Args:
            X: 입력 데이터
        
        Returns:
            클래스 확률
        """
        logits = self.forward(X)
        return softmax(logits)


class SGDOptimizer:
    """
    Stochastic Gradient Descent Optimizer
    """
    
    def __init__(self, model, learning_rate=0.01, momentum=0.0):
        """
        초기화
        
        Args:
            model: MLPVectorized 모델
            learning_rate: 학습률
            momentum: 모멘텀 계수
        """
        self.model = model
        self.lr = learning_rate
        self.momentum = momentum
        
        # 모멘텀을 위한 속도 저장
        if momentum > 0:
            self.velocities = []
            for layer in model.layers:
                self.velocities.append({
                    'W': np.zeros_like(layer.W),
                    'b': np.zeros_like(layer.b)
                })
        else:
            self.velocities = None
    
    def step(self):
        """파라미터 업데이트"""
        if self.momentum > 0:
            # 모멘텀 SGD
            for layer, velocity in zip(self.model.layers, self.velocities):
                # 속도 업데이트
                velocity['W'] = self.momentum * velocity['W'] - self.lr * layer.dW
                velocity['b'] = self.momentum * velocity['b'] - self.lr * layer.db
                
                # 파라미터 업데이트
                layer.W += velocity['W']
                layer.b += velocity['b']
        else:
            # 일반 SGD
            self.model.update_params(self.lr)


# ============================================
# 테스트 코드
# ============================================

if __name__ == "__main__":
    print("🧪 Vectorized Neural Network 테스트")
    print("-" * 50)
    
    # 데이터 생성
    np.random.seed(42)
    X = np.random.randn(32, 10)  # 32개 샘플, 10차원
    y = np.random.randint(0, 3, 32)  # 3개 클래스
    
    # 모델 생성
    model = MLPVectorized(
        input_dim=10,
        hidden_dims=[20, 15],
        output_dim=3
    )
    
    print("모델 구조: 10 -> 20 -> 15 -> 3")
    
    # Forward pass
    output = model.forward(X)
    print(f"출력 shape: {output.shape}")
    
    # Softmax 적용
    probs = model.predict_proba(X)
    print(f"확률 합: {probs.sum(axis=1)[:5]}")  # 처음 5개만
    
    # Backward pass 테스트
    from core.tensor_ops import cross_entropy
    
    # 간단한 그래디언트 (실제로는 loss의 그래디언트 사용)
    dout = probs.copy()
    dout[np.arange(32), y] -= 1
    dout /= 32
    
    din = model.backward(dout)
    print(f"입력 그래디언트 shape: {din.shape}")
    
    # 파라미터 업데이트
    optimizer = SGDOptimizer(model, learning_rate=0.1, momentum=0.9)
    optimizer.step()
    
    print("\n✅ 모든 테스트 통과!")


