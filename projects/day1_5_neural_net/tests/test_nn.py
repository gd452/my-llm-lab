"""
🧪 Neural Network 단위 테스트

각 구성 요소가 올바르게 작동하는지 확인합니다.
"""

import pytest
import math

# 깔끔한 import!
from core.autograd import Value
from core.neuron import Neuron
from core.layer import Layer
from core.mlp import MLP
from core.losses import mse_loss
from core.optimizer import SGD


class TestNeuron:
    """Neuron 클래스 테스트"""
    
    @pytest.mark.skip(reason="TODO: Neuron 구현 필요")
    def test_neuron_creation(self):
        """뉴런 생성 테스트"""
        neuron = Neuron(3)
        assert len(neuron.w) == 3
        assert neuron.b is not None
        assert len(neuron.parameters()) == 4  # 3 weights + 1 bias
    
    @pytest.mark.skip(reason="TODO: Neuron 구현 필요")
    def test_neuron_forward(self):
        """뉴런 forward pass 테스트"""
        neuron = Neuron(2)
        x = [Value(1.0), Value(2.0)]
        output = neuron(x)
        
        assert isinstance(output, Value)
        assert -1 <= output.data <= 1  # tanh 출력 범위
    
    @pytest.mark.skip(reason="TODO: Neuron 구현 필요")
    def test_neuron_linear(self):
        """선형 뉴런 테스트"""
        neuron = Neuron(2, nonlin=False)
        x = [Value(1.0), Value(1.0)]
        
        # 가중치를 수동으로 설정
        neuron.w[0].data = 0.5
        neuron.w[1].data = 0.5
        neuron.b.data = 0.0
        
        output = neuron(x)
        assert abs(output.data - 1.0) < 1e-6  # 0.5*1 + 0.5*1 + 0 = 1


class TestLayer:
    """Layer 클래스 테스트"""
    
    @pytest.mark.skip(reason="TODO: Layer 구현 필요")
    def test_layer_creation(self):
        """레이어 생성 테스트"""
        layer = Layer(3, 2)
        assert len(layer.neurons) == 2
        assert len(layer.parameters()) == 8  # (3+1)*2 = 8
    
    @pytest.mark.skip(reason="TODO: Layer 구현 필요")
    def test_layer_forward(self):
        """레이어 forward pass 테스트"""
        layer = Layer(2, 3)
        x = [Value(1.0), Value(2.0)]
        outputs = layer(x)
        
        assert len(outputs) == 3
        assert all(isinstance(o, Value) for o in outputs)


class TestMLP:
    """MLP 클래스 테스트"""
    
    @pytest.mark.skip(reason="TODO: MLP 구현 필요")
    def test_mlp_creation(self):
        """MLP 생성 테스트"""
        mlp = MLP(2, [4, 1])
        assert len(mlp.layers) == 2
        assert mlp.architecture == [2, 4, 1]
    
    @pytest.mark.skip(reason="TODO: MLP 구현 필요")
    def test_mlp_forward(self):
        """MLP forward pass 테스트"""
        mlp = MLP(2, [3, 1])
        x = [Value(0.5), Value(0.5)]
        output = mlp(x)
        
        assert isinstance(output, Value)
    
    @pytest.mark.skip(reason="TODO: MLP 구현 필요")
    def test_mlp_parameters(self):
        """MLP 파라미터 개수 테스트"""
        mlp = MLP(2, [4, 1])
        # 첫 번째 층: (2+1)*4 = 12
        # 두 번째 층: (4+1)*1 = 5
        # 총: 17
        assert len(mlp.parameters()) == 17


class TestLosses:
    """손실 함수 테스트"""
    
    @pytest.mark.skip(reason="TODO: losses 구현 필요")
    def test_mse_loss(self):
        """MSE 손실 함수 테스트"""
        predictions = [Value(0.5), Value(0.8)]
        targets = [Value(0.0), Value(1.0)]
        
        loss = mse_loss(predictions, targets)
        
        # MSE = ((0.5-0)^2 + (0.8-1)^2) / 2 = (0.25 + 0.04) / 2 = 0.145
        expected = 0.145
        assert abs(loss.data - expected) < 1e-6
    
    @pytest.mark.skip(reason="TODO: losses 구현 필요")
    def test_mse_gradient(self):
        """MSE gradient 테스트"""
        pred = Value(0.5)
        target = Value(0.0)
        
        loss = mse_loss([pred], [target])
        loss.backward()
        
        # d(MSE)/d(pred) = 2*(pred - target) / n = 2*0.5 / 1 = 1.0
        assert abs(pred.grad - 1.0) < 1e-6


class TestOptimizer:
    """최적화기 테스트"""
    
    @pytest.mark.skip(reason="TODO: optimizer 구현 필요")
    def test_sgd_step(self):
        """SGD step 테스트"""
        params = [Value(1.0), Value(2.0)]
        params[0].grad = 0.1
        params[1].grad = -0.2
        
        optimizer = SGD(params, lr=0.1)
        optimizer.step()
        
        # param = param - lr * grad
        assert abs(params[0].data - 0.99) < 1e-6  # 1.0 - 0.1*0.1
        assert abs(params[1].data - 2.02) < 1e-6  # 2.0 - 0.1*(-0.2)
    
    @pytest.mark.skip(reason="TODO: optimizer 구현 필요")
    def test_sgd_zero_grad(self):
        """SGD zero_grad 테스트"""
        params = [Value(1.0), Value(2.0)]
        params[0].grad = 0.5
        params[1].grad = 0.5
        
        optimizer = SGD(params)
        optimizer.zero_grad()
        
        assert all(p.grad == 0.0 for p in params)


class TestIntegration:
    """통합 테스트"""
    
    @pytest.mark.skip(reason="TODO: 전체 구현 필요")
    def test_xor_learning(self):
        """XOR 학습 가능성 테스트"""
        # 간단한 XOR 학습 테스트
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [0, 1, 1, 0]
        
        model = MLP(2, [4, 1])
        optimizer = SGD(model.parameters(), lr=0.1)
        
        # 100 epoch 학습
        for _ in range(100):
            loss_val = 0
            
            for inputs, target in zip(X, y):
                x_vals = [Value(x) for x in inputs]
                pred = model(x_vals)
                loss = (pred - Value(target)) ** 2
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_val += loss.data
        
        # 학습 후 loss가 감소했는지 확인
        assert loss_val / 4 < 0.1  # 평균 loss < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])