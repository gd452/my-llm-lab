"""
속도 문제 디버깅
"""

import torch
import torch.nn as nn
import time
import numpy as np
from model import GPT

# 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# 모델 파라미터 (train.py와 동일)
vocab_size = 65
n_embd = 384
n_head = 6
n_layer = 6
block_size = 256
batch_size = 64

# 모델 생성
print("\n모델 생성 중...")
model = GPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    dropout=0.0  # 추론 시 dropout 없음
).to(device)

# 파라미터 수
total_params = sum(p.numel() for p in model.parameters())
print(f"총 파라미터: {total_params/1e6:.2f}M")

# 더미 데이터
X = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)
Y = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)

print(f"\n입력 shape: {X.shape}")
print(f"Batch size: {batch_size}, Sequence length: {block_size}")

# Warmup
print("\nWarmup...")
for _ in range(3):
    with torch.no_grad():
        _, _ = model(X, Y)

# 속도 측정
print("\n속도 측정 (10회 평균):")
times = []

with torch.no_grad():
    for i in range(10):
        start = time.time()
        logits, loss = model(X, Y)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        elapsed = end - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}초")

avg_time = np.mean(times)
std_time = np.std(times)

print(f"\n평균: {avg_time:.4f}초 (±{std_time:.4f})")
print(f"처리량: {batch_size / avg_time:.1f} sequences/sec")

# 각 컴포넌트 속도 측정
print("\n각 컴포넌트 속도:")

# 1. Embedding만
with torch.no_grad():
    start = time.time()
    tok_emb = model.token_embedding_table(X)
    pos_emb = model.position_embedding_table(torch.arange(block_size).to(device))
    x = tok_emb + pos_emb
    end = time.time()
    print(f"  Embeddings: {(end-start)*1000:.2f}ms")

# 2. Single block
single_block = model.blocks[0]
with torch.no_grad():
    start = time.time()
    out = single_block(x)
    end = time.time()
    print(f"  Single block: {(end-start)*1000:.2f}ms")
    print(f"  → 6 blocks 예상: {(end-start)*6*1000:.2f}ms")

# 3. Output projection
with torch.no_grad():
    start = time.time()
    logits = model.lm_head(x)
    end = time.time()
    print(f"  Output projection: {(end-start)*1000:.2f}ms")

# 메모리 사용량
if device == 'cpu':
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"\n메모리 사용: {memory_info.rss / 1024**2:.0f} MB")

print("\n진단:")
if avg_time > 1.0:
    print("⚠️ 너무 느림! 가능한 원인:")
    print("  1. CPU 성능 문제")
    print("  2. 메모리 스와핑")
    print("  3. 다른 프로세스가 CPU 사용")
    print("  4. PyTorch 설치 문제")
elif avg_time > 0.5:
    print("⚡ 약간 느림. 최적화 가능:")
    print("  - batch_size 줄이기")
    print("  - sequence length 줄이기")
    print("  - 모델 크기 줄이기")
else:
    print("✅ 정상 속도!")

# PyTorch 스레드 확인
print(f"\nPyTorch threads: {torch.get_num_threads()}")
print("더 빠르게 하려면:")
print("  export OMP_NUM_THREADS=8  # CPU 코어 수에 맞게")