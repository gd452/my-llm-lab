"""
🌐 Translation Toy: 간단한 번역 태스크

Attention을 사용한 시퀀스-투-시퀀스 학습의 간단한 예제입니다.
숫자를 영어에서 프랑스어로 "번역"하는 태스크를 구현합니다.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from core.attention import (
    MultiHeadAttention,
    add_positional_encoding,
    create_causal_mask,
    visualize_attention
)


# 간단한 "번역" 데이터셋
TRANSLATION_DATA = {
    "one": "un",
    "two": "deux", 
    "three": "trois",
    "four": "quatre",
    "five": "cinq",
    "six": "six",
    "seven": "sept",
    "eight": "huit",
    "nine": "neuf",
    "ten": "dix"
}

# 더 복잡한 예제
PHRASE_DATA = {
    "hello world": "bonjour monde",
    "good morning": "bon matin",
    "thank you": "merci",
    "see you": "à bientôt",
    "how are you": "comment allez vous"
}


class SimpleSeq2SeqWithAttention:
    """
    Attention을 사용한 간단한 Sequence-to-Sequence 모델
    """
    
    def __init__(self, vocab_size, d_model=64, num_heads=4):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        
        # 단어 임베딩 (랜덤 초기화)
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1
        
        # Encoder self-attention
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        
        # Decoder self-attention
        self.decoder_self_attention = MultiHeadAttention(d_model, num_heads)
        
        # Decoder cross-attention (encoder 출력 참조)
        self.decoder_cross_attention = MultiHeadAttention(d_model, num_heads)
        
        # 출력 레이어
        self.output_projection = np.random.randn(d_model, vocab_size) * 0.1
        
        # 어휘 사전
        self.word_to_idx = {}
        self.idx_to_word = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """어휘 사전 구축"""
        words = set()
        
        # 모든 단어 수집
        for en, fr in {**TRANSLATION_DATA, **PHRASE_DATA}.items():
            words.update(en.split())
            words.update(fr.split())
        
        # 특수 토큰 추가
        words.add("<PAD>")
        words.add("<START>")
        words.add("<END>")
        
        # 인덱스 매핑
        for idx, word in enumerate(sorted(words)):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def encode_sentence(self, sentence):
        """문장을 인덱스로 변환"""
        tokens = sentence.split()
        indices = [self.word_to_idx.get(token, 0) for token in tokens]
        return indices
    
    def decode_indices(self, indices):
        """인덱스를 문장으로 변환"""
        tokens = [self.idx_to_word.get(idx, "<UNK>") for idx in indices]
        return " ".join(tokens)
    
    def embed_tokens(self, token_indices):
        """토큰 인덱스를 임베딩으로 변환"""
        embeddings = []
        for idx in token_indices:
            if idx < len(self.embedding):
                embeddings.append(self.embedding[idx])
            else:
                embeddings.append(np.zeros(self.d_model))
        return np.array(embeddings)
    
    def encode(self, source_sentence):
        """
        Encoder: 소스 문장 인코딩
        """
        # 토큰화 및 임베딩
        indices = self.encode_sentence(source_sentence)
        embeddings = self.embed_tokens(indices)
        
        # Positional encoding 추가
        embeddings = add_positional_encoding(embeddings)
        
        # Self-attention
        encoded, enc_attention = self.encoder_attention.forward(
            embeddings, embeddings, embeddings
        )
        
        return encoded, enc_attention, indices
    
    def decode(self, target_sentence, encoder_output):
        """
        Decoder: 타겟 문장 디코딩
        """
        # 토큰화 및 임베딩
        indices = self.encode_sentence(target_sentence)
        embeddings = self.embed_tokens(indices)
        
        # Positional encoding 추가
        embeddings = add_positional_encoding(embeddings)
        
        # Causal mask (미래 단어 차단)
        seq_len = len(indices)
        causal_mask = create_causal_mask(seq_len)
        
        # Self-attention (with causal mask)
        self_attn_out, self_attn_weights = self.decoder_self_attention.forward(
            embeddings, embeddings, embeddings, mask=causal_mask
        )
        
        # Cross-attention (encoder 참조)
        cross_attn_out, cross_attn_weights = self.decoder_cross_attention.forward(
            self_attn_out, encoder_output, encoder_output
        )
        
        # 출력 projection
        logits = np.matmul(cross_attn_out, self.output_projection)
        
        return logits, self_attn_weights, cross_attn_weights, indices
    
    def translate(self, source_sentence, visualize=True):
        """
        번역 수행
        """
        print(f"\n{'=' * 60}")
        print(f"소스 (영어): {source_sentence}")
        
        # Encoding
        encoder_output, enc_attention, src_indices = self.encode(source_sentence)
        src_tokens = source_sentence.split()
        
        # 실제 번역 (여기서는 정답 사용)
        if source_sentence in TRANSLATION_DATA:
            target_sentence = TRANSLATION_DATA[source_sentence]
        elif source_sentence in PHRASE_DATA:
            target_sentence = PHRASE_DATA[source_sentence]
        else:
            target_sentence = "unknown"
        
        print(f"타겟 (프랑스어): {target_sentence}")
        
        # Decoding
        logits, self_attn, cross_attn, tgt_indices = self.decode(
            target_sentence, encoder_output
        )
        tgt_tokens = target_sentence.split()
        
        # Attention 시각화
        if visualize:
            print("\n📊 Encoder Self-Attention:")
            self._visualize_attention(enc_attention.mean(axis=0), src_tokens, src_tokens)
            
            print("\n📊 Decoder Self-Attention:")
            self._visualize_attention(self_attn.mean(axis=0), tgt_tokens, tgt_tokens)
            
            print("\n📊 Cross-Attention (Decoder → Encoder):")
            self._visualize_attention(cross_attn.mean(axis=0), tgt_tokens, src_tokens)
        
        return target_sentence
    
    def _visualize_attention(self, weights, query_tokens, key_tokens):
        """Attention 가중치 시각화"""
        print("\n      ", end="")
        for token in key_tokens:
            print(f"{token[:6]:^8}", end="")
        print()
        
        for i, q_token in enumerate(query_tokens):
            print(f"{q_token[:6]:>6}", end="")
            for j, k_token in enumerate(key_tokens):
                weight = weights[i, j] if i < len(weights) and j < len(weights[0]) else 0
                
                if weight > 0.5:
                    print("  ████  ", end="")
                elif weight > 0.3:
                    print("  ▓▓▓  ", end="")
                elif weight > 0.15:
                    print("  ▒▒  ", end="")
                elif weight > 0.05:
                    print("  ░░  ", end="")
                else:
                    print("  ··  ", end="")
            print()


def demonstrate_translation():
    """
    번역 데모
    """
    print("=" * 60)
    print("🌐 Attention을 사용한 번역 데모")
    print("=" * 60)
    
    # 모델 생성
    model = SimpleSeq2SeqWithAttention(vocab_size=50, d_model=32, num_heads=4)
    
    print("\n📚 번역 데이터셋:")
    print("-" * 40)
    for en, fr in list(TRANSLATION_DATA.items())[:5]:
        print(f"  {en:10} → {fr}")
    print("  ...")
    
    # 단순 단어 번역
    print("\n" + "=" * 60)
    print("1️⃣ 단일 단어 번역")
    print("=" * 60)
    
    for word in ["one", "five", "ten"]:
        model.translate(word, visualize=True)
    
    # 구문 번역
    print("\n" + "=" * 60)
    print("2️⃣ 구문 번역")
    print("=" * 60)
    
    for phrase in ["hello world", "thank you"]:
        model.translate(phrase, visualize=True)


def analyze_attention_patterns():
    """
    Attention 패턴 분석
    """
    print("\n" + "=" * 60)
    print("📈 Attention 패턴 분석")
    print("=" * 60)
    
    model = SimpleSeq2SeqWithAttention(vocab_size=50, d_model=32, num_heads=4)
    
    # 다양한 길이의 문장으로 테스트
    test_sentences = [
        ("one", "un"),
        ("one two", "un deux"),
        ("one two three", "un deux trois")
    ]
    
    for src, tgt in test_sentences:
        print(f"\n길이 {len(src.split())} → {len(tgt.split())} 번역:")
        print(f"  {src} → {tgt}")
        
        # Encoding
        encoder_output, _, _ = model.encode(src)
        
        # Decoding
        _, _, cross_attn, _ = model.decode(tgt, encoder_output)
        
        # Cross-attention 분석
        avg_cross_attn = cross_attn.mean(axis=0)
        
        print("\nCross-Attention 패턴:")
        src_tokens = src.split()
        tgt_tokens = tgt.split()
        
        for i, t_tok in enumerate(tgt_tokens):
            if i < len(avg_cross_attn):
                max_idx = np.argmax(avg_cross_attn[i])
                if max_idx < len(src_tokens):
                    print(f"  {t_tok} → {src_tokens[max_idx]} "
                          f"(attention: {avg_cross_attn[i, max_idx]:.3f})")


def interactive_translation():
    """
    대화형 번역
    """
    print("\n" + "=" * 60)
    print("🎮 대화형 번역 실험")
    print("=" * 60)
    
    model = SimpleSeq2SeqWithAttention(vocab_size=50, d_model=32, num_heads=4)
    
    print("\n사용 가능한 단어:")
    print("-" * 40)
    print("영어:", ", ".join(list(TRANSLATION_DATA.keys())[:10]))
    print("\n사용 가능한 구문:")
    print("영어:", ", ".join(list(PHRASE_DATA.keys())[:5]))
    
    while True:
        print("\n" + "-" * 60)
        user_input = input("\n영어 입력 (종료: 'quit'): ").strip().lower()
        
        if user_input == 'quit':
            break
        
        if user_input in TRANSLATION_DATA or user_input in PHRASE_DATA:
            model.translate(user_input, visualize=True)
        else:
            print("⚠️ 미등록 단어/구문입니다. 등록된 단어를 사용하세요.")


def main():
    """메인 실행 함수"""
    print("🌐" * 30)
    print("Translation with Attention".center(60))
    print("🌐" * 30)
    
    # 1. 기본 번역 데모
    demonstrate_translation()
    
    # 2. Attention 패턴 분석
    analyze_attention_patterns()
    
    # 3. 대화형 실험
    print("\n" + "=" * 60)
    response = input("\n대화형 번역을 시작하시겠습니까? (y/n): ")
    if response.lower() == 'y':
        interactive_translation()
    
    print("\n" + "🎉" * 20)
    print("번역 데모를 완료했습니다!")
    print("Cross-Attention이 어떻게 소스와 타겟을 연결하는지 보셨나요?")
    print("🎉" * 20)


if __name__ == "__main__":
    main()