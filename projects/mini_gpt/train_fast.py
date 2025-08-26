"""
최적화된 학습 스크립트
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from model import GPT
import os
import time

# ========== 최적화 설정 ==========
# 더 작은 배치와 시퀀스로 속도 향상
batch_size = 32        # 64 → 32로 줄임
block_size = 128       # 256 → 128로 줄임
max_iters = 1000       # 5000 → 1000 (빠른 테스트)
eval_interval = 100    # 500 → 100
learning_rate = 3e-4   
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20        # 평가 반복 수

# 더 작은 모델
n_embd = 256          # 384 → 256
n_head = 4            # 6 → 4  
n_layer = 4           # 6 → 4
dropout = 0.2

# CPU 최적화
if device == 'cpu':
    torch.set_num_threads(8)  # CPU 스레드 수 설정
    print(f"CPU threads: {torch.get_num_threads()}")

torch.manual_seed(1337)

# ========== 데이터 준비 ==========
print("데이터 로딩 중...")

# 셰익스피어 텍스트
if not os.path.exists('data/shakespeare.txt'):
    os.makedirs('data', exist_ok=True)
    import urllib.request
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    urllib.request.urlretrieve(url, 'data/shakespeare.txt')

with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 데이터 분할
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Vocabulary size: {vocab_size}")
print(f"학습 데이터: {len(train_data):,} 토큰")
print(f"검증 데이터: {len(val_data):,} 토큰")

# ========== 데이터 로더 ==========
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# ========== 평가 함수 ==========
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ========== 모델 생성 ==========
print(f"\n최적화된 모델 생성...")
print(f"  작은 모델: {n_layer} layers, {n_head} heads, {n_embd} dim")
print(f"  작은 배치: {batch_size} sequences × {block_size} tokens")

model = GPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout
).to(device)

# 파라미터 수
total_params = sum(p.numel() for p in model.parameters())
print(f"  총 파라미터: {total_params/1e6:.2f}M (원래 10.8M)")

# 옵티마이저
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ========== 학습 루프 ==========
print(f"\n빠른 학습 시작! (device: {device})")
print("-" * 50)

# 초기 속도 테스트
print("속도 테스트...")
start_time = time.time()
for _ in range(5):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
end_time = time.time()
print(f"5 iterations: {end_time - start_time:.2f}초")
print(f"평균: {(end_time - start_time)/5:.2f}초/iter")
print("-" * 50)

# 실제 학습
for iter in range(max_iters):
    
    # 평가
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
        
        # 샘플 생성
        if iter % 200 == 0:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=100)
            print(f"\n생성 샘플:\n{decode(generated[0].tolist())}\n")
            print("-" * 50)
    
    # 학습
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n학습 완료!")

# 모델 저장
os.makedirs('outputs', exist_ok=True)
torch.save(model.state_dict(), 'outputs/model_fast.pt')
print("모델 저장: outputs/model_fast.pt")

# 최종 생성
print("\n최종 생성:")
print("="*50)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = decode(model.generate(context, max_new_tokens=200)[0].tolist())
print(generated)
print("="*50)