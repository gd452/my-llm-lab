"""
🎯 Mini MNIST: 벡터화된 신경망으로 숫자 분류

MNIST의 일부(0-4)를 사용하여 벡터화된 신경망을 학습합니다.
sklearn의 데이터셋을 사용하여 간단하게 구현합니다.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from core.nn_vectorized import MLPVectorized, SGDOptimizer
from core.tensor_ops import softmax, cross_entropy, accuracy
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_mini_mnist(num_classes=5):
    """
    Mini MNIST 데이터 로드 (0-4 숫자만)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("📊 데이터 로딩...")
    
    # sklearn의 digits 데이터셋 사용 (8x8 이미지)
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # 0-4 클래스만 선택
    mask = y < num_classes
    X, y = X[mask], y[mask]
    
    # 데이터 정규화
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ 데이터 준비 완료!")
    print(f"   학습 데이터: {X_train.shape}")
    print(f"   테스트 데이터: {X_test.shape}")
    print(f"   클래스: {num_classes}개 (0-{num_classes-1})")
    
    return X_train, X_test, y_train, y_test


class DataLoader:
    """미니배치 데이터 로더"""
    
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)
    
    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            yield self.X[batch_indices], self.y[batch_indices]
    
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def train_epoch(model, dataloader, optimizer):
    """한 에폭 학습"""
    total_loss = 0
    total_acc = 0
    n_batches = 0
    
    for X_batch, y_batch in dataloader:
        # Forward pass
        logits = model.forward(X_batch)
        probs = softmax(logits)
        
        # Loss 계산
        loss = cross_entropy(probs, y_batch)
        acc = accuracy(probs, y_batch)
        
        # Backward pass
        grad_out = probs.copy()
        grad_out[np.arange(len(y_batch)), y_batch] -= 1
        grad_out /= len(y_batch)
        
        model.backward(grad_out)
        
        # 파라미터 업데이트
        optimizer.step()
        
        total_loss += loss
        total_acc += acc
        n_batches += 1
    
    return total_loss / n_batches, total_acc / n_batches


def evaluate(model, X_test, y_test, batch_size=32):
    """모델 평가"""
    dataloader = DataLoader(X_test, y_test, batch_size=batch_size, shuffle=False)
    
    total_acc = 0
    n_batches = 0
    all_preds = []
    
    for X_batch, y_batch in dataloader:
        probs = model.predict_proba(X_batch)
        batch_acc = accuracy(probs, y_batch)
        total_acc += batch_acc
        n_batches += 1
        all_preds.extend(np.argmax(probs, axis=1))
    
    return total_acc / n_batches, np.array(all_preds)


def print_confusion_matrix(y_true, y_pred, num_classes=5):
    """혼동 행렬 출력"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    
    print("\n📊 Confusion Matrix:")
    print("    예측 →")
    print("실제 ", end="")
    for i in range(num_classes):
        print(f"  {i}", end="")
    print()
    
    for i in range(num_classes):
        print(f" {i}  ", end="")
        for j in range(num_classes):
            if i == j:
                print(f" \033[92m{cm[i, j]:2d}\033[0m", end="")  # 녹색
            else:
                print(f" {cm[i, j]:2d}", end="")
        print(f"  ({cm[i, i]/cm[i].sum()*100:.1f}%)")


def main():
    """메인 학습 루프"""
    print("=" * 60)
    print("🔢 Mini MNIST Classifier (Vectorized)")
    print("=" * 60)
    
    # 하이퍼파라미터
    num_classes = 5
    hidden_dims = [128, 64]
    learning_rate = 0.1
    momentum = 0.9
    batch_size = 32
    epochs = 50
    
    # 데이터 로드
    X_train, X_test, y_train, y_test = load_mini_mnist(num_classes)
    
    # 모델 생성
    model = MLPVectorized(
        input_dim=X_train.shape[1],  # 64 (8x8)
        hidden_dims=hidden_dims,
        output_dim=num_classes
    )
    
    print(f"\n🏗️ 모델 구조: {X_train.shape[1]} → {' → '.join(map(str, hidden_dims))} → {num_classes}")
    
    # 옵티마이저
    optimizer = SGDOptimizer(model, learning_rate=learning_rate, momentum=momentum)
    
    # 데이터 로더
    train_loader = DataLoader(X_train, y_train, batch_size=batch_size)
    
    # 학습
    print(f"\n📈 학습 시작...")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Momentum: {momentum}")
    print("-" * 40)
    
    history = {'loss': [], 'acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # 학습
        train_loss, train_acc = train_epoch(model, train_loader, optimizer)
        
        # 평가
        val_acc, _ = evaluate(model, X_test, y_test)
        
        # 기록
        history['loss'].append(train_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 출력
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: "
                  f"Loss={train_loss:.4f}, "
                  f"Train Acc={train_acc:.3f}, "
                  f"Val Acc={val_acc:.3f}")
    
    print("-" * 40)
    print("✅ 학습 완료!")
    
    # 최종 평가
    print("\n📊 최종 평가:")
    test_acc, predictions = evaluate(model, X_test, y_test)
    print(f"테스트 정확도: {test_acc:.3f} ({int(test_acc * len(y_test))}/{len(y_test)})")
    
    # 혼동 행렬
    print_confusion_matrix(y_test, predictions, num_classes)
    
    # 학습 곡선 시각화 (텍스트)
    print("\n📈 학습 곡선:")
    print("Acc ↑")
    
    # 간단한 텍스트 그래프
    max_acc = max(max(history['acc']), max(history['val_acc']))
    height = 10
    
    for h in range(height, -1, -1):
        line = "   │"
        
        # 샘플링 (50 에폭을 20개 포인트로)
        step = max(1, len(history['acc']) // 20)
        
        for i in range(0, len(history['acc']), step):
            if history['acc'][i] / max_acc * height >= h:
                line += "●"  # Train
            elif history['val_acc'][i] / max_acc * height >= h:
                line += "○"  # Val
            else:
                line += " "
        
        if h == height:
            line += f" {max_acc:.2f}"
        elif h == 0:
            line += " 0.00"
            
        print(line)
    
    print("   └" + "─" * 20 + "→ Epoch")
    print("    0" + " " * 16 + f"{epochs}")
    print("\n   ● Train  ○ Validation")
    
    # 성공 메시지
    if test_acc > 0.9:
        print("\n" + "🎉" * 20)
        print("축하합니다! 90% 이상의 정확도를 달성했습니다!")
        print("벡터화의 힘을 경험하셨습니다!")
        print("🎉" * 20)
    
    return model, history


if __name__ == "__main__":
    model, history = main()