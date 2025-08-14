"""
⚙️ Optimizer: 최적화 알고리즘

이 파일에서 구현할 것:
1. SGD (Stochastic Gradient Descent) - 확률적 경사 하강법
2. 파라미터 업데이트
3. Gradient 초기화

최적화 알고리즘은 손실 함수의 gradient를 사용해
파라미터를 업데이트합니다.
"""


class SGD:
    """
    Stochastic Gradient Descent Optimizer
    
    가장 기본적인 최적화 알고리즘입니다.
    각 파라미터를 gradient의 반대 방향으로 이동시킵니다.
    
    업데이트 규칙:
        param = param - learning_rate * gradient
    
    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.01)
        >>> loss.backward()  # gradient 계산
        >>> optimizer.step()  # 파라미터 업데이트
        >>> optimizer.zero_grad()  # gradient 초기화
    """
    
    def __init__(self, parameters, lr=0.01):
        """
        SGD 초기화
        
        Args:
            parameters: 최적화할 파라미터 리스트 (Value 객체들)
            lr: 학습률 (learning rate)
            
        TODO:
        1. self.parameters에 파라미터 저장
        2. self.lr에 학습률 저장
        """
        # TODO: 파라미터와 학습률 저장
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        """
        파라미터 업데이트 수행
        
        각 파라미터를 gradient의 반대 방향으로 이동:
        param.data = param.data - lr * param.grad
        
        TODO:
        1. 모든 파라미터에 대해
        2. param.data -= self.lr * param.grad 수행
        
        힌트:
            for param in self.parameters:
                param.data -= self.lr * param.grad
        """
        # TODO: 파라미터 업데이트
        pass  # TODO: 구현
    
    def zero_grad(self):
        """
        모든 gradient를 0으로 초기화
        
        역전파 전에 이전 gradient를 지워야 합니다.
        (gradient는 누적되기 때문)
        
        TODO:
        1. 모든 파라미터의 grad를 0으로 설정
        
        힌트:
            for param in self.parameters:
                param.grad = 0.0
        """
        # TODO: gradient 초기화
        pass  # TODO: 구현


class Adam:
    """
    Adam Optimizer (선택 구현)
    
    Adaptive Moment Estimation
    - Momentum과 RMSProp을 결합한 최적화 알고리즘
    - 각 파라미터마다 적응적 학습률 사용
    
    더 고급 최적화를 원한다면 구현해보세요!
    """
    
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        """
        Adam 초기화 (선택 구현)
        
        Args:
            parameters: 최적화할 파라미터
            lr: 학습률
            betas: (β1, β2) 모멘텀 계수
            eps: 수치 안정성을 위한 작은 값
        """
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        
        # TODO: 각 파라미터마다 m, v 초기화 (선택)
        # self.m = [0] * len(parameters)  # 1차 모멘트
        # self.v = [0] * len(parameters)  # 2차 모멘트
        # self.t = 0  # 시간 스텝
    
    def step(self):
        """Adam 업데이트 (선택 구현)"""
        # TODO: Adam 알고리즘 구현 (선택)
        pass
    
    def zero_grad(self):
        """gradient 초기화"""
        for param in self.parameters:
            param.grad = 0.0


# 테스트 코드
if __name__ == "__main__":
    print("🧪 Optimizer 테스트")
    print("-" * 50)
    
    # TODO 구현 후 테스트
    """
    from tiny_autograd_project._10_core.autograd_tiny.value import Value
    
    # 파라미터 생성
    params = [Value(1.0), Value(2.0), Value(3.0)]
    
    # Gradient 시뮬레이션
    params[0].grad = 0.1
    params[1].grad = -0.2
    params[2].grad = 0.3
    
    print("초기 파라미터:")
    print([p.data for p in params])
    print("Gradients:")
    print([p.grad for p in params])
    
    # SGD 적용
    optimizer = SGD(params, lr=0.1)
    optimizer.step()
    
    print("\n업데이트 후 파라미터:")
    print([p.data for p in params])
    # 예상: [0.99, 2.02, 2.97]
    
    # Gradient 초기화
    optimizer.zero_grad()
    print("\nGradient 초기화 후:")
    print([p.grad for p in params])
    # 예상: [0.0, 0.0, 0.0]
    """
    print("TODO: optimizer.py 구현 필요!")