"""
🎯 XOR Problem: 신경망의 Hello World

XOR (Exclusive OR) 문제는 신경망 학습의 고전적인 예제입니다.
단순한 선형 분류기로는 해결할 수 없어, 은닉층이 필요합니다.

XOR 진리표:
    입력 | 출력
    -----|-----
    0, 0 |  0
    0, 1 |  1
    1, 0 |  1  
    1, 1 |  0

이 패턴은 선형으로 분리 불가능합니다!
"""

# 방법 1: pip install -e . 설치 후 (권장)

from core.autograd import Value
from core.mlp import MLP
from core.losses import mse_loss
from core.optimizer import SGD


def create_xor_data():
    """
    XOR 데이터셋 생성
    
    Returns:
        X: 입력 데이터
        y: 목표 출력
    """
    # XOR 입력
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    
    # XOR 출력
    y = [0.0, 1.0, 1.0, 0.0]
    
    return X, y


def train_xor(epochs=1000, lr=0.1, hidden_size=4, verbose=True):
    """
    XOR 문제를 학습하는 신경망
    
    Args:
        epochs: 학습 반복 횟수
        lr: 학습률
        hidden_size: 은닉층 크기
        verbose: 출력 여부
    
    TODO 구현 후 작동:
    1. 데이터 준비
    2. 모델 생성 (2-hidden-1 구조)
    3. 최적화기 생성
    4. 학습 루프:
       - Forward pass
       - Loss 계산
       - Backward pass
       - 파라미터 업데이트
    """
    
    print("=" * 60)
    print("🎯 XOR Problem Solver")
    print("=" * 60)
    
    # 데이터 준비
    X, y = create_xor_data()
    print(f"\n📊 데이터:")
    for i, (inputs, target) in enumerate(zip(X, y)):
        print(f"  {inputs} → {target}")
    
    # 모델 생성
    model = MLP(2, [hidden_size, 1])  # 2-4-1 구조
    print(f"\n🏗️ 모델 구조: 2-{hidden_size}-1")
    model.summary()
    
    # 최적화기
    optimizer = SGD(model.parameters(), lr=lr)
    print(f"\n⚙️ 최적화: SGD (lr={lr})")
    
    # 학습
    print(f"\n📈 학습 시작 (epochs={epochs})...")
    print("-" * 40)
    
    history = []
    
    for epoch in range(epochs):
        # Forward pass - 모든 데이터에 대해
        predictions = []
        targets = []
        
        for inputs, target in zip(X, y):
            # Value 객체로 변환
            x_vals = [Value(x) for x in inputs]
            y_val = Value(target)
            
            # 예측
            pred = model(x_vals)
            predictions.append(pred)
            targets.append(y_val)
        
        # Loss 계산
        loss = mse_loss(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 파라미터 업데이트
        optimizer.step()
        
        # 기록
        history.append(loss.data)
        
        # 출력
        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:4d}: Loss = {loss.data:.6f}")
        
    
    print("-" * 40)
    print("✅ 학습 완료!")
    
    # 최종 평가
    print("\n📊 최종 결과:")
    print("-" * 40)
    correct = 0
    
    for inputs, target in zip(X, y):
        x_vals = [Value(x) for x in inputs]
        pred = model(x_vals)
        pred_val = pred.data
        
        # 0.5를 기준으로 분류
        pred_class = 1 if pred_val > 0.5 else 0
        is_correct = pred_class == int(target)
        correct += is_correct
        
        symbol = "✓" if is_correct else "✗"
        print(f"  {inputs} → {pred_val:.4f} (예측: {pred_class}, 정답: {int(target)}) {symbol}")
    
    accuracy = correct / len(X) * 100
    print("-" * 40)
    print(f"🎯 정확도: {accuracy:.1f}% ({correct}/{len(X)})")

    
    # 학습 곡선 시각화 (선택)
    if verbose:
        print("\n📈 학습 곡선:")
        print("  Loss")
        print("  ↑")
        
        # 간단한 텍스트 그래프
        max_loss = max(history)
        height = 10
        width = min(50, len(history))
        step = len(history) // width
        
        for h in range(height, -1, -1):
            line = "  │"
            for i in range(0, len(history), step):
                if history[i] / max_loss * height >= h:
                    line += "█"
                else:
                    line += " "
            print(line)
        print("  └" + "─" * width + "→ Epoch")
        print(f"    0{' ' * (width - 5)}{epochs}")
    
    return model, history
    


def visualize_decision_boundary(model):
    """
    결정 경계 시각화 (선택 구현)
    
    2D 공간에서 모델의 결정 경계를 텍스트로 시각화합니다.
    """
    # TODO: 선택 구현
    pass


if __name__ == "__main__":
    # XOR 문제 해결
    model, history = train_xor(
        epochs=1000,
        lr=0.1,
        hidden_size=4,
        verbose=True
    )
    
    # 성공 메시지
    if model:
        print("\n" + "🎉" * 20)
        print("축하합니다! XOR 문제를 해결했습니다!")
        print("이제 더 복잡한 문제에 도전해보세요!")
        print("🎉" * 20)