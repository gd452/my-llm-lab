"""
miniGPT 분석 도구 - 모델이 무엇을 학습했는지 시각화
Attention 패턴을 보여주고 모델의 동작을 이해
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import GPT

def visualize_attention(model, text, layer_idx=0, head_idx=0):
    """특정 레이어/헤드의 attention 패턴 시각화"""
    
    # Tokenizer 준비
    with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
        full_text = f.read()
    chars = sorted(list(set(full_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    
    # 텍스트 인코딩
    tokens = encode(text)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    # Forward pass with hooks to capture attention
    model.eval()
    with torch.no_grad():
        # 임베딩
        tok_emb = model.token_embedding_table(x)
        pos_emb = model.position_embedding_table(torch.arange(len(tokens)))
        x_emb = tok_emb + pos_emb
        
        # 특정 블록의 attention 가져오기
        block = model.blocks[layer_idx]
        x_norm = block.ln1(x_emb)
        
        # 특정 head의 attention 계산
        head = block.sa.heads[head_idx]
        B, T, C = x_norm.shape
        k = head.key(x_norm)
        q = head.query(x_norm)
        
        # Attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(head.tril[:T, :T] == 0, float('-inf'))
        attention = F.softmax(wei, dim=-1).squeeze().numpy()
    
    # 시각화
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention, cmap='Blues', aspect='auto')
    
    # 축 레이블
    ax.set_xticks(range(len(text)))
    ax.set_yticks(range(len(text)))
    ax.set_xticklabels(list(text), rotation=45, ha='right')
    ax.set_yticklabels(list(text))
    
    ax.set_xlabel('Keys (being attended to)')
    ax.set_ylabel('Queries (attending from)')
    ax.set_title(f'Attention Pattern - Layer {layer_idx}, Head {head_idx}')
    
    # Colorbar
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'outputs/attention_L{layer_idx}_H{head_idx}.png')
    plt.show()
    
    print(f"Attention 시각화 저장: outputs/attention_L{layer_idx}_H{head_idx}.png")

def analyze_embeddings(model):
    """토큰 임베딩 분석 - 유사한 문자들이 가까운 벡터를 가지는지"""
    
    # 문자 리스트
    with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    
    # 임베딩 추출
    embeddings = model.token_embedding_table.weight.detach().numpy()
    
    # 일부 문자들의 유사도 계산
    sample_chars = ['a', 'e', 'i', 'o', 'u',  # 모음
                   't', 'h', 's', 'n', 'r',  # 자주 쓰이는 자음
                   '.', ',', '!', '?',       # 구두점
                   ' ', '\n']                # 공백
    
    print("\n문자 임베딩 유사도 분석:")
    print("-" * 50)
    
    for i, char1 in enumerate(sample_chars):
        if char1 not in chars:
            continue
        idx1 = chars.index(char1)
        similarities = []
        
        for char2 in sample_chars:
            if char2 not in chars or char1 == char2:
                continue
            idx2 = chars.index(char2)
            
            # 코사인 유사도
            vec1 = embeddings[idx1]
            vec2 = embeddings[idx2]
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities.append((char2, sim))
        
        # 가장 유사한 문자들
        similarities.sort(key=lambda x: x[1], reverse=True)
        top3 = similarities[:3]
        
        char_repr = repr(char1)
        print(f"{char_repr:5} 와 가장 유사한: {', '.join([f'{repr(c)}({s:.2f})' for c, s in top3])}")

def analyze_generation_probability(model, prompt="To be or not to be"):
    """각 스텝에서의 생성 확률 분포 분석"""
    
    # Tokenizer
    with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    
    # 프롬프트 인코딩
    tokens = encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
        probs = F.softmax(logits[0, -1, :], dim=-1)
    
    # Top 10 확률
    top_probs, top_indices = torch.topk(probs, 10)
    
    print(f"\n'{prompt}' 다음에 올 확률이 높은 문자:")
    print("-" * 50)
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        char = itos[idx.item()]
        char_repr = repr(char)
        bar = '█' * int(prob.item() * 50)
        print(f"{i+1:2}. {char_repr:5} {prob.item():.3f} {bar}")

def main():
    """분석 실행"""
    import os
    
    # 모델 체크
    if not os.path.exists('outputs/model.pt'):
        print("에러: 학습된 모델이 없습니다. 먼저 'python train.py'를 실행하세요.")
        return
    
    # outputs 디렉토리 생성
    os.makedirs('outputs', exist_ok=True)
    
    # 모델 로드
    print("모델 로딩 중...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 모델 설정 (train.py와 동일)
    vocab_size = 65
    n_embd = 384
    block_size = 256
    n_head = 6
    n_layer = 6
    dropout = 0.0
    
    model = GPT(vocab_size, n_embd, block_size, n_head, n_layer, dropout)
    model.load_state_dict(torch.load('outputs/model.pt', map_location=device))
    model.eval()
    
    # 1. Attention 시각화
    print("\n=== Attention 패턴 시각화 ===")
    sample_text = "To be or not"
    print(f"분석할 텍스트: '{sample_text}'")
    visualize_attention(model, sample_text, layer_idx=0, head_idx=0)
    
    # 2. 임베딩 분석
    print("\n=== 문자 임베딩 분석 ===")
    analyze_embeddings(model)
    
    # 3. 생성 확률 분석
    print("\n=== 생성 확률 분석 ===")
    analyze_generation_probability(model, "Romeo")
    analyze_generation_probability(model, "KING")
    
    print("\n분석 완료!")
    print("💡 Tip: layer_idx와 head_idx를 바꿔가며 다른 attention 패턴도 확인해보세요.")

if __name__ == '__main__':
    main()