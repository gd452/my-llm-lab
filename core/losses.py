"""
📉 Loss Functions: 손실 함수

이 파일에서 구현할 것:
1. MSE (Mean Squared Error) - 회귀 문제용
2. Binary Cross-Entropy - 이진 분류용 (선택)

손실 함수는 예측값과 실제값의 차이를 측정합니다.
학습의 목표는 이 손실을 최소화하는 것입니다.
"""

if __name__ == "__main__":
    # 직접 실행할 때만 path 추가
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math


def mse_loss(predictions, targets):
    """
    Mean Squared Error Loss
    
    MSE = (1/n) * Σ(pred_i - target_i)²
    
    Args:
        predictions: 예측값 리스트 (Value 객체들)
        targets: 목표값 리스트 (Value 객체들 또는 숫자들)
    
    Returns:
        평균 제곱 오차 (Value 객체)
        
    TODO:
    1. predictions와 targets가 리스트가 아니면 리스트로 변환
    2. 각 예측값과 목표값의 차이를 제곱
    3. 평균 계산
    """
    
    # 단일 값을 리스트로 변환
    if not isinstance(predictions, list):
        predictions = [predictions]
    if not isinstance(targets, list):
        targets = [targets]
        
    # MSE 계산
    losses = [(pred - target)**2 for pred, target in zip(predictions, targets)]
    return sum(losses) * (1.0 / len(losses))


def binary_cross_entropy(predictions, targets, epsilon=1e-7):
    """
    Binary Cross-Entropy Loss (선택 구현)
    
    BCE = -Σ(target * log(pred) + (1-target) * log(1-pred))
    
    Args:
        predictions: 예측 확률 (0~1 사이, sigmoid 출력)
        targets: 실제 레이블 (0 또는 1)
        epsilon: 수치 안정성을 위한 작은 값
    
    Returns:
        크로스 엔트로피 손실
        
    Note:
        이 함수는 선택사항입니다.
        log 함수가 Value 클래스에 구현되어 있어야 합니다.
    """
    # 선택 구현
    if not isinstance(predictions, list):
        predictions = [predictions]
    if not isinstance(targets, list):
        targets = [targets]
        
    losses = [-target * math.log(pred + epsilon) - (1 - target) * math.log(1 - pred + epsilon) for pred, target in zip(predictions, targets)]
    return sum(losses) * (1.0 / len(losses))


def accuracy(predictions, targets, threshold=0.5):
    """
    정확도 계산 (분류 문제용)
    
    Args:
        predictions: 예측값
        targets: 실제값
        threshold: 이진 분류 임계값
    
    Returns:
        정확도 (0~1 사이의 float)
        
    TODO:
    1. 예측값을 threshold와 비교하여 0/1로 변환
    2. 실제값과 비교하여 맞은 개수 계산
    3. 전체 대비 비율 반환
    """
    return sum(1 for pred, target in zip(predictions, targets) if pred >= threshold and target == 1 or pred < threshold and target == 0) / len(predictions)


# 테스트 코드
if __name__ == "__main__":
    print("🧪 Loss Functions 테스트")
    print("-" * 50)
    
    from core.autograd import Value
    
    # MSE 테스트
    predictions = [Value(0.5), Value(0.8)]
    targets = [Value(0.0), Value(1.0)]
    
    loss = mse_loss(predictions, targets)
    print(f"예측: {[p.data for p in predictions]}")
    print(f"목표: {[t.data for t in targets]}")
    print(f"MSE Loss: {loss.data}")
    
    # Backward pass 테스트
    loss.backward()
    print(f"Gradients: {[p.grad for p in predictions]}")