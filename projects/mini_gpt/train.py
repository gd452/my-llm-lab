"""
miniGPT 학습 스크립트
셰익스피어 텍스트로 학습하여 비슷한 스타일의 텍스트를 생성하는 모델 훈련
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from model import GPT
import os

# ========== 설정 ==========
# 하이퍼파라미터 (작은 모델로 빠른 학습)
batch_size = 64        # 한 번에 처리할 시퀀스 수
block_size = 256       # 컨텍스트 길이 (최대 256 토큰까지 기억)
max_iters = 5000       # 학습 반복 횟수
eval_interval = 500    # 평가 주기
learning_rate = 3e-4   # 학습률
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200       # 평가 시 평균낼 반복 수
n_embd = 384          # 임베딩 차원 (작은 모델: 384)
n_head = 6            # attention head 수
n_layer = 6           # transformer block 수
dropout = 0.2         # dropout 비율

torch.manual_seed(1337)

# ========== 데이터 준비 ==========
print("데이터 로딩 중...")

# 셰익스피어 텍스트 다운로드 (없으면)
if not os.path.exists('data/shakespeare.txt'):
    os.makedirs('data', exist_ok=True)
    import urllib.request
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    urllib.request.urlretrieve(url, 'data/shakespeare.txt')
    print("셰익스피어 텍스트 다운로드 완료!")

# 텍스트 읽기
with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Character-level tokenizer 구축
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
print(f"샘플 문자들: {chars[:20]}...")

# 문자 ↔ 정수 매핑
stoi = {ch: i for i, ch in enumerate(chars)}  # string to integer
itos = {i: ch for i, ch in enumerate(chars)}  # integer to string
encode = lambda s: [stoi[c] for c in s]       # 텍스트 → 숫자
decode = lambda l: ''.join([itos[i] for i in l])  # 숫자 → 텍스트

# 학습/검증 데이터 분할
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"학습 데이터: {len(train_data):,} 토큰")
print(f"검증 데이터: {len(val_data):,} 토큰")

# ========== 데이터 로더 ==========
def get_batch(split):
    """학습용 미니배치 생성"""
    data = train_data if split == 'train' else val_data
    # 랜덤 시작 위치 선택
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 입력과 타겟 준비 (타겟은 입력의 다음 토큰)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# ========== 평가 함수 ==========
@torch.no_grad()
def estimate_loss():
    """학습/검증 loss 평가"""
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
print(f"\n모델 생성 중...")
model = GPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout
)
m = model.to(device)

# 파라미터 수 출력
print(f"모델 파라미터 수: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

# ========== 옵티마이저 ==========
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ========== 학습 루프 ==========
print(f"\n학습 시작! (device: {device})")
print("-" * 50)

for iter in range(max_iters):
    
    # 주기적으로 loss 평가
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
        
        # 샘플 생성 (학습 진행 상황 확인)
        if iter % 1000 == 0:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=100)
            print(f"\n생성 샘플:\n{decode(generated[0].tolist())}\n")
            print("-" * 50)
    
    # 미니배치 샘플링
    xb, yb = get_batch('train')
    
    # Forward pass
    logits, loss = model(xb, yb)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n학습 완료!")

# ========== 모델 저장 ==========
torch.save(model.state_dict(), 'outputs/model.pt')
print("모델 저장 완료: outputs/model.pt")

# ========== 최종 생성 테스트 ==========
print("\n" + "="*50)
print("최종 텍스트 생성 테스트:")
print("="*50)

model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)[0].tolist())
print(generated_text)
print("="*50)