"""
학습된 miniGPT 모델로 텍스트 생성
다양한 생성 전략 실험 가능
"""

import torch
from model import GPT
import argparse
import os

def load_model(model_path='outputs/model.pt'):
    """저장된 모델 로드"""
    
    # 학습 때와 동일한 설정
    vocab_size = 65  # 셰익스피어 텍스트의 고유 문자 수
    n_embd = 384
    block_size = 256
    n_head = 6
    n_layer = 6
    dropout = 0.0  # 생성 시에는 dropout 사용 안 함
    
    # 모델 생성 및 가중치 로드
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(vocab_size, n_embd, block_size, n_head, n_layer, dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    return model, device

def get_tokenizer():
    """Character tokenizer 준비"""
    # 셰익스피어 텍스트에서 vocabulary 재구성
    with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    return encode, decode, stoi

def generate_text(model, device, prompt, max_new_tokens, temperature, top_k):
    """텍스트 생성"""
    
    encode, decode, stoi = get_tokenizer()
    
    # 프롬프트 인코딩
    if prompt:
        # 프롬프트의 문자가 vocabulary에 있는지 확인
        for char in prompt:
            if char not in stoi:
                print(f"경고: '{char}'는 vocabulary에 없음. 공백으로 대체.")
                prompt = prompt.replace(char, ' ')
        
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        print(f"\n프롬프트: {prompt}")
    else:
        # 빈 컨텍스트로 시작 (완전 자유 생성)
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print("\n프롬프트 없이 생성 시작...")
    
    print(f"Temperature: {temperature}, Top-k: {top_k}")
    print("-" * 50)
    
    # 생성
    with torch.no_grad():
        generated = model.generate(
            context, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # 디코딩 및 출력
    generated_text = decode(generated[0].tolist())
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='miniGPT 텍스트 생성')
    parser.add_argument('--prompt', type=str, default='', 
                       help='시작 텍스트 (빈 문자열이면 처음부터 생성)')
    parser.add_argument('--length', type=int, default=500,
                       help='생성할 토큰 수 (기본: 500)')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='생성 다양성 (0.1=보수적, 1.0=균형, 2.0=창의적)')
    parser.add_argument('--top_k', type=int, default=40,
                       help='Top-k 샘플링 (작을수록 안전, 클수록 다양)')
    parser.add_argument('--model', type=str, default='outputs/model.pt',
                       help='모델 파일 경로')
    
    args = parser.parse_args()
    
    # 모델 체크
    if not os.path.exists(args.model):
        print(f"에러: 모델 파일 '{args.model}'을 찾을 수 없습니다.")
        print("먼저 'python train.py'를 실행하여 모델을 학습시키세요.")
        return
    
    # 모델 로드
    print("모델 로딩 중...")
    model, device = load_model(args.model)
    
    # 텍스트 생성
    generated = generate_text(
        model, device,
        prompt=args.prompt,
        max_new_tokens=args.length,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    print(generated)
    print("-" * 50)
    
    # 생성 옵션 실험 제안
    if args.temperature == 0.8 and args.top_k == 40:
        print("\n💡 다른 설정도 시도해보세요:")
        print("  - 보수적: --temperature 0.5 --top_k 10")
        print("  - 창의적: --temperature 1.2 --top_k 100")
        print("  - 극단적: --temperature 2.0 --top_k 200")

if __name__ == '__main__':
    main()