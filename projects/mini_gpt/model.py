"""
miniGPT - Karpathy's nanoGPT 스타일로 구현한 최소 GPT
각 컴포넌트는 핵심 개념만 포함하며, 왜 필요한지 명확히 이해할 수 있도록 구성
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """단일 attention head - 한 가지 관점에서 단어들의 관계를 학습"""
    
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 하삼각 행렬: 미래 정보를 볼 수 없게 마스킹
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # Batch, Time(sequence), Channels(embedding)
        
        # 각 토큰이 "무엇을 찾고 있는지"(query)와 "무엇을 제공하는지"(key)
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        
        # Attention scores: "각 토큰이 다른 토큰들을 얼마나 주목하는가"
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # 미래 마스킹
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # 가중 평균으로 정보 집계
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """여러 attention head를 병렬로 실행 - 다양한 관점에서 관계 학습"""
    
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 각 head의 출력을 연결하고 선형 변환
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """단순 MLP - 각 위치에서 독립적으로 특징 변환"""
    
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 확장
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # 축소
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block - attention과 feedforward를 결합"""
    
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual connections: 정보 손실 방지
        x = x + self.sa(self.ln1(x))    # attention 후 잔차 연결
        x = x + self.ffwd(self.ln2(x))  # feedforward 후 잔차 연결
        return x

class GPT(nn.Module):
    """최소 GPT 모델 - 텍스트 생성을 위한 언어 모델"""
    
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        
        # 토큰과 위치 임베딩
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Transformer blocks 쌓기
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        
        # 최종 layer norm과 출력 projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # 가중치 초기화
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 임베딩: 토큰 ID → 벡터
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb  # 토큰 + 위치 정보
        
        # Transformer blocks 통과
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # 각 위치에서 다음 토큰 확률 예측
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # 학습 시 loss 계산
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """텍스트 생성 - 자기회귀적으로 다음 토큰 예측"""
        for _ in range(max_new_tokens):
            # 컨텍스트 윈도우 크기로 자르기
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # 예측
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 마지막 위치의 logits만 사용
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 확률로 변환하고 샘플링
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 생성된 토큰 추가
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx