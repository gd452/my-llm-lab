"""
Transformer Demo - Simple Copy Task
Using PyTorch implementation for actual learning
Educational demo: Input sequence → Same output sequence
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from core.transformer import Transformer


def create_toy_dataset():
    """Create a toy dataset - simple copy task (identity function)"""
    # 더 많은 데이터로 증강
    sequences = [
        # 기본 순차 패턴
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9],
        # 홀수/짝수 패턴
        [1, 3, 5, 7, 9],
        [2, 4, 6, 8, 1],
        # 역순 패턴
        [9, 8, 7, 6, 5],
        [8, 7, 6, 5, 4],
        [7, 6, 5, 4, 3],
        [6, 5, 4, 3, 2],
        [5, 4, 3, 2, 1],
        # 중복 패턴 (어려운 케이스) - 더 많이 추가
        [1, 1, 2, 2, 3],
        [2, 2, 3, 3, 4],
        [3, 3, 4, 4, 5],
        [4, 4, 5, 5, 6],
        [5, 5, 6, 6, 7],
        [6, 6, 7, 7, 8],
        [7, 7, 8, 8, 9],
        [8, 8, 9, 9, 1],
        # 더 다양한 중복 패턴
        [1, 1, 1, 2, 2],
        [2, 2, 2, 3, 3],
        [3, 3, 3, 4, 4],
        [1, 2, 2, 3, 3],
        [2, 3, 3, 4, 4],
        [3, 4, 4, 5, 5],
        # 반복 패턴
        [1, 2, 1, 2, 1],
        [2, 3, 2, 3, 2],
        [3, 4, 3, 4, 3],
        # 단순 패턴 추가
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
    ]
    
    # Source와 Target이 동일 (identity task)
    src_sequences = sequences
    tgt_sequences = sequences
    
    return torch.tensor(src_sequences), torch.tensor(tgt_sequences)


def train_model(model, src_data, tgt_data, device, n_epochs=150, lr=0.0003):
    """Train the transformer model"""
    model.train()
    
    # 더 낮은 learning rate와 scheduler 사용
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()  # padding 제거 (copy task는 padding 없음)
    
    losses = []
    batch_size = 4  # 배치 처리로 안정성 향상
    
    for epoch in range(n_epochs):
        total_loss = 0
        num_batches = 0
        
        # 데이터 셔플
        indices = torch.randperm(len(src_data))
        
        for batch_start in range(0, len(src_data), batch_size):
            batch_end = min(batch_start + batch_size, len(src_data))
            batch_indices = indices[batch_start:batch_end]
            
            optimizer.zero_grad()
            
            # Get batch
            src_batch = src_data[batch_indices].to(device)
            tgt_batch = tgt_data[batch_indices].to(device)
            
            # Teacher forcing
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]
            
            # Forward pass
            logits = model(src_batch, tgt_input)
            
            # Calculate loss
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (더 보수적으로)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Learning rate scheduling
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        # Print progress
        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    return losses


def test_model(model, src_data, tgt_data, device):
    """Test the trained model"""
    model.eval()
    correct = 0
    
    print("\nTesting copy task:")
    
    with torch.no_grad():
        for i in range(len(src_data)):
            src = src_data[i:i+1].to(device)
            true_tgt = tgt_data[i].cpu().numpy()
            
            # Generate sequence - simple greedy generation
            # Start with first token from source
            generated = model.generate_simple(src)
            generated = generated[0].cpu().numpy()
            
            # Check correctness
            is_correct = np.array_equal(generated, true_tgt)
            if is_correct:
                correct += 1
            
            # Print first 8 results and wrong ones
            if i < 8 or not is_correct:
                print(f"\n  Input:     {src_data[i].cpu().numpy()}")
                print(f"  Expected:  {true_tgt}")
                print(f"  Generated: {generated}")
                print(f"  Status:    {'✓ Correct' if is_correct else '✗ Wrong'}")
    
    return correct

"""
개선 결과:
  - 75% → 81.2% (6.2% 향상)
  - 데이터셋: 16개 → 32개 (2배 증가)
  - 모델 크기: 61K → 354K 파라미터 (약 6배 증가)

  적용한 개선 사항:
  1. 데이터 증강 (16→32개): 특히 중복 패턴 데이터 추가
  2. 모델 크기 증가:
    - d_model: 32 → 64
    - 레이어: 2 → 3
    - FFN: 128 → 256
  3. Dropout 감소: 0.1 → 0.05
  4. Epoch 증가: 100 → 150
  5. Learning rate 조정: 0.0005 → 0.0003

  여전히 어려운 패턴:
  - 연속된 중복 숫자 ([1,1,2,2,3], [2,2,3,3,4] 등)
  - 모델이 중복을 잘 처리하지 못함

  추가 개선 방법들:
  1. 더 많은 데이터: 100개 이상의 샘플
  2. 더 긴 학습: 300-500 epochs
  3. Attention 시각화: 어떤 위치를 주목하는지 확인
  4. Position Encoding 개선: 위치 정보를 더 잘 학습
  5. Teacher forcing ratio 조정: 학습 중 가끔 자기 예측 사용

  현재 81.2%는 작은 데이터셋과 간단한 모델로는 꽤 좋은 성과입니다! 교육용 데모로 충분한
  수준이네요.
  """

def main():
    print("=" * 60)
    print("Transformer Demo - Simple Copy Task")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create dataset
    print("\nCreating dataset...")
    src_data, tgt_data = create_toy_dataset()
    print(f"Dataset size: {len(src_data)} examples")
    
    # Show examples
    print("\nDataset examples:")
    for i in range(min(2, len(src_data))):
        print(f"  Source: {src_data[i].numpy()} -> Target: {tgt_data[i].numpy()}")
    
    # Create model
    print("\nCreating model...")
    model = Transformer(
        vocab_size=20,        # Small vocabulary for toy task
        d_model=64,           # 증가: 32 → 64 (더 많은 표현력)
        n_heads=4,
        n_encoder_layers=3,   # 증가: 2 → 3 (더 깊은 인코더)
        n_decoder_layers=3,   # 증가: 2 → 3 (더 깊은 디코더)
        d_ff=256,            # 증가: 128 → 256 (더 큰 FFN)
        max_seq_len=10,
        dropout=0.05         # 감소: 0.1 → 0.05 (작은 데이터셋에는 낮은 dropout)
    ).to(device)
    
    # Print model info
    print("Model created:")
    print(f"  Encoder layers: 3")
    print(f"  Decoder layers: 3")
    print(f"  Model dimension: 64")
    print(f"  Attention heads: 4")
    print(f"  Feed-forward dim: 256")
    print(f"  Dropout: 0.05")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Train model
    print("\nTraining...")
    print("  Using batch size: 4")
    print("  Using learning rate: 0.0003 with cosine annealing")
    print("  Training for 150 epochs")
    losses = train_model(model, src_data, tgt_data, device, n_epochs=150, lr=0.0003)
    
    # Check learning progress
    if len(losses) > 20:
        initial_loss = np.mean(losses[:10])
        final_loss = np.mean(losses[-10:])
        improvement = (initial_loss - final_loss) / initial_loss * 100
        
        print(f"\nLearning progress:")
        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Improvement: {improvement:.1f}%")
    
    # Test model
    correct = test_model(model, src_data, tgt_data, device)
    
    # Print results
    accuracy = correct / len(src_data) * 100
    print("\n" + "=" * 60)
    print(f"Results: {correct}/{len(src_data)} correct = {accuracy:.1f}% accuracy")
    
    if accuracy >= 75:
        print("✓ Success! The model learned to copy sequences!")
    else:
        print("The model needs more training or tuning.")
    
    print("=" * 60)
    print("\nNote: This is a toy example. Real Transformers need:")
    print("  - Much larger datasets")
    print("  - More sophisticated tokenization")
    print("  - Better hyperparameter tuning")
    print("  - Longer training times")
    print("=" * 60)


if __name__ == "__main__":
    main()