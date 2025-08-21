"""
👁️ Attention Visualizer: Attention 가중치 시각화

Attention이 실제로 어떻게 작동하는지 시각적으로 확인합니다.
다양한 문장에서 단어들이 서로 어떻게 주목하는지 봅니다.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from core.attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    positional_encoding,
    add_positional_encoding,
    create_causal_mask
)


def create_word_embeddings(words, embedding_dim=64):
    """
    간단한 단어 임베딩 생성 (랜덤)
    실제로는 학습된 임베딩을 사용
    """
    np.random.seed(42)  # 재현성을 위해
    vocab = {}
    embeddings = []
    
    for word in words:
        if word not in vocab:
            vocab[word] = np.random.randn(embedding_dim)
        embeddings.append(vocab[word])
    
    return np.array(embeddings), vocab


def visualize_attention_heatmap(attention_weights, tokens, title="Attention Weights"):
    """
    Attention 가중치를 히트맵으로 시각화
    """
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print('=' * 60)
    
    seq_len = len(tokens)
    
    # 헤더 (Query)
    print("\n" + " " * 12 + "Keys (어디를 볼 것인가?)")
    print(" " * 10, end="")
    for token in tokens:
        print(f"{token[:8]:^10}", end="")
    print()
    print(" " * 10 + "─" * (10 * seq_len))
    
    # 히트맵
    for i, query_token in enumerate(tokens):
        print(f"{query_token[:8]:>8} │", end="")
        
        for j, key_token in enumerate(tokens):
            weight = attention_weights[i, j]
            
            # 색상 강도로 표현
            if weight > 0.5:
                symbol = "████"
            elif weight > 0.3:
                symbol = "▓▓▓▓"
            elif weight > 0.15:
                symbol = "▒▒▒▒"
            elif weight > 0.05:
                symbol = "░░░░"
            else:
                symbol = "····"
            
            # 대각선 (자기 자신) 강조
            if i == j:
                print(f" [{symbol}] ", end="")
            else:
                print(f"  {symbol}  ", end="")
        
        print(f" │ {query_token}")
    
    print(" " * 10 + "─" * (10 * seq_len))
    print("\nQueries")
    print("(무엇을")
    print("찾는가?)")
    
    # 범례
    print("\n범례: ████ (>0.5) ▓▓▓▓ (>0.3) ▒▒▒▒ (>0.15) ░░░░ (>0.05) ···· (<0.05)")


def demonstrate_self_attention():
    """
    Self-Attention 데모
    """
    print("\n" + "🔍 Self-Attention 데모 ".center(60, "="))
    
    # 예제 문장
    sentence = "The cat sat on the mat"
    tokens = sentence.split()
    print(f"\n문장: '{sentence}'")
    print(f"토큰: {tokens}")
    
    # 임베딩 생성
    embeddings, vocab = create_word_embeddings(tokens, embedding_dim=32)
    print(f"\n임베딩 shape: {embeddings.shape}")
    
    # Positional encoding 추가
    embeddings_with_pe = add_positional_encoding(embeddings)
    
    # Self-attention 계산
    output, attention_weights = scaled_dot_product_attention(
        embeddings_with_pe, 
        embeddings_with_pe, 
        embeddings_with_pe
    )
    
    # 시각화
    visualize_attention_heatmap(attention_weights, tokens, 
                               "Self-Attention (모든 단어가 서로를 볼 수 있음)")
    
    # 분석
    print("\n📊 Attention 분석:")
    for i, token in enumerate(tokens):
        top_indices = np.argsort(attention_weights[i])[-3:][::-1]
        top_tokens = [tokens[idx] for idx in top_indices]
        top_weights = [attention_weights[i, idx] for idx in top_indices]
        
        print(f"  '{token}' 주목 → ", end="")
        for t, w in zip(top_tokens, top_weights):
            print(f"{t}({w:.2f}) ", end="")
        print()


def demonstrate_causal_attention():
    """
    Causal Attention 데모 (GPT 스타일)
    """
    print("\n" + "🔮 Causal Attention 데모 ".center(60, "="))
    
    # 예제 문장
    sentence = "I think therefore I am"
    tokens = sentence.split()
    print(f"\n문장: '{sentence}'")
    print(f"토큰: {tokens}")
    
    # 임베딩 생성
    embeddings, _ = create_word_embeddings(tokens, embedding_dim=32)
    embeddings_with_pe = add_positional_encoding(embeddings)
    
    # Causal mask 생성
    seq_len = len(tokens)
    causal_mask = create_causal_mask(seq_len)
    
    # Causal self-attention
    output, attention_weights = scaled_dot_product_attention(
        embeddings_with_pe,
        embeddings_with_pe,
        embeddings_with_pe,
        mask=causal_mask
    )
    
    # 시각화
    visualize_attention_heatmap(attention_weights, tokens,
                               "Causal Attention (미래를 볼 수 없음)")
    
    print("\n💡 특징:")
    print("  - 각 단어는 자신과 이전 단어들만 볼 수 있음")
    print("  - 하삼각 행렬 형태")
    print("  - GPT와 같은 생성 모델에서 사용")


def demonstrate_multi_head_attention():
    """
    Multi-Head Attention 데모
    """
    print("\n" + "🎭 Multi-Head Attention 데모 ".center(60, "="))
    
    # 예제 문장
    sentence = "Time flies like an arrow"
    tokens = sentence.split()
    print(f"\n문장: '{sentence}'")
    print(f"토큰: {tokens}")
    
    # 임베딩 생성
    embeddings, _ = create_word_embeddings(tokens, embedding_dim=64)
    embeddings_with_pe = add_positional_encoding(embeddings)
    
    # Multi-head attention
    num_heads = 4
    mha = MultiHeadAttention(d_model=64, num_heads=num_heads)
    
    output, attention_weights = mha.forward(
        embeddings_with_pe,
        embeddings_with_pe,
        embeddings_with_pe
    )
    
    print(f"\n{num_heads}개의 Attention Head:")
    print("각 head가 다른 패턴을 학습합니다.")
    
    # 각 head 시각화
    for head_idx in range(min(2, num_heads)):  # 처음 2개 head만
        print(f"\n{'─' * 60}")
        visualize_attention_heatmap(
            attention_weights[head_idx],
            tokens,
            f"Head {head_idx + 1}/{num_heads}"
        )
    
    # 평균 attention
    avg_attention = attention_weights.mean(axis=0)
    print(f"\n{'─' * 60}")
    visualize_attention_heatmap(
        avg_attention,
        tokens,
        "평균 Attention (모든 head의 평균)"
    )


def analyze_positional_encoding():
    """
    Positional Encoding 분석
    """
    print("\n" + "📍 Positional Encoding 분석 ".center(60, "="))
    
    seq_len = 20
    d_model = 64
    
    pe = positional_encoding(seq_len, d_model)
    
    print(f"\nPositional Encoding shape: {pe.shape}")
    print(f"값 범위: [{pe.min():.3f}, {pe.max():.3f}]")
    
    # 위치별 패턴 시각화
    print("\n위치별 인코딩 패턴 (처음 8차원):")
    print("Pos ", end="")
    for dim in range(8):
        print(f" Dim{dim:2d}", end="")
    print()
    print("─" * 50)
    
    for pos in range(min(10, seq_len)):
        print(f"{pos:3d} ", end="")
        for dim in range(8):
            val = pe[pos, dim]
            if val > 0.5:
                print("  ██ ", end="")
            elif val > 0:
                print("  ▓▓ ", end="")
            elif val > -0.5:
                print("  ░░ ", end="")
            else:
                print("  ·· ", end="")
        print()
    
    # 주파수 분석
    print("\n📊 차원별 주파수:")
    for dim in [0, d_model//4, d_model//2, d_model-1]:
        # 주기 계산
        if dim % 2 == 0:
            freq = 1.0 / (10000 ** (dim / d_model))
            wavelength = 2 * np.pi / freq
            print(f"  Dim {dim:3d}: 파장 ≈ {wavelength:.1f} positions")


def interactive_attention():
    """
    대화형 Attention 실험
    """
    print("\n" + "🎮 대화형 Attention 실험 ".center(60, "="))
    print("\n자신만의 문장으로 Attention을 실험해보세요!")
    
    while True:
        print("\n" + "─" * 60)
        user_input = input("\n문장 입력 (종료: 'quit'): ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if not user_input:
            continue
        
        tokens = user_input.split()
        if len(tokens) < 2:
            print("최소 2개 이상의 단어를 입력하세요.")
            continue
        
        if len(tokens) > 10:
            print("10개 이하의 단어로 입력하세요.")
            continue
        
        # 임베딩 생성
        embeddings, _ = create_word_embeddings(tokens, embedding_dim=32)
        embeddings_with_pe = add_positional_encoding(embeddings)
        
        # Attention 계산
        output, attention_weights = scaled_dot_product_attention(
            embeddings_with_pe,
            embeddings_with_pe,
            embeddings_with_pe
        )
        
        # 시각화
        visualize_attention_heatmap(attention_weights, tokens, "Your Attention")
        
        # 가장 강한 연결 찾기
        print("\n🔗 가장 강한 연결 (자기 자신 제외):")
        for i in range(len(tokens)):
            attention_weights[i, i] = 0  # 자기 자신 제외
        
        max_idx = np.unravel_index(attention_weights.argmax(), attention_weights.shape)
        max_weight = attention_weights[max_idx]
        
        print(f"  '{tokens[max_idx[0]]}' → '{tokens[max_idx[1]]}' "
              f"(weight: {max_weight:.3f})")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("👁️  Attention Mechanism Visualizer")
    print("=" * 60)
    
    # 1. Self-Attention 데모
    demonstrate_self_attention()
    
    # 2. Causal Attention 데모
    demonstrate_causal_attention()
    
    # 3. Multi-Head Attention 데모
    demonstrate_multi_head_attention()
    
    # 4. Positional Encoding 분석
    analyze_positional_encoding()
    
    # 5. 대화형 실험
    print("\n" + "=" * 60)
    response = input("\n대화형 실험을 시작하시겠습니까? (y/n): ")
    if response.lower() == 'y':
        interactive_attention()
    
    print("\n" + "🎉" * 20)
    print("Attention 시각화를 완료했습니다!")
    print("이제 Attention이 어떻게 작동하는지 이해하셨나요?")
    print("🎉" * 20)


if __name__ == "__main__":
    main()