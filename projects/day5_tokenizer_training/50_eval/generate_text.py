"""
Text generation demo with different sampling strategies
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
from core.tokenizer import CharacterTokenizer
from core.training import TextGenerator


class PretrainedMiniLM:
    """Simulated pre-trained model for demo"""
    
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        # Simulate learned patterns
        self.common_bigrams = {
            'T': ['h', 'o', 'a'],
            'h': ['e', 'a', 'i'],
            'e': [' ', 'r', 's'],
            't': ['h', 'o', ' '],
            ' ': ['t', 'a', 'i'],
            'a': ['n', 't', ' '],
            'n': ['d', 'g', ' '],
            'i': ['n', 's', 't'],
            's': [' ', 't', 'e'],
            'o': ['n', 'r', ' '],
        }
    
    def forward(self, input_ids):
        """Generate logits with learned patterns"""
        if isinstance(input_ids, list):
            seq_len = len(input_ids)
            batch_size = 1
        else:
            batch_size = 1 if len(input_ids.shape) == 1 else input_ids.shape[0]
            seq_len = input_ids.shape[-1] if len(input_ids.shape) > 1 else len(input_ids)
        
        # Generate logits
        logits = np.random.randn(batch_size, seq_len, self.vocab_size) * 0.5
        
        # Boost likely next characters based on patterns
        # This simulates a trained model
        if seq_len > 0:
            last_token_id = input_ids[-1] if isinstance(input_ids, list) else input_ids.flat[-1]
            
            # Add some structure to make generation more interesting
            # Boost common English patterns
            for i in range(self.vocab_size):
                if i < 10:  # Boost special tokens less
                    logits[:, -1, i] -= 2
                elif 10 <= i < 40:  # Boost common characters
                    logits[:, -1, i] += np.random.rand() * 2
                else:  # Reduce uncommon characters
                    logits[:, -1, i] -= 1
        
        return logits


def demonstrate_generation_strategies():
    """Show different text generation strategies"""
    
    # Setup
    print("Setting up model and tokenizer...")
    
    # Create tokenizer
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    Actions speak louder than words.
    Time flies when you're having fun.
    All that glitters is not gold.
    Better late than never.
    """
    
    tokenizer = CharacterTokenizer()
    tokenizer.fit(sample_text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model
    model = PretrainedMiniLM(tokenizer.vocab_size)
    generator = TextGenerator(model, tokenizer)
    
    # Test prompts
    prompts = [
        "The ",
        "Time ",
        "All ",
    ]
    
    print("\n" + "=" * 60)
    print("Text Generation Strategies Comparison")
    print("=" * 60)
    
    for prompt in prompts:
        print(f"\n{'='*40}")
        print(f"Prompt: '{prompt}'")
        print(f"{'='*40}")
        
        # 1. Greedy Decoding
        print("\n1. Greedy Decoding (temperature=0):")
        generated = generator.generate(
            prompt, 
            max_length=40,
            temperature=0
        )
        print(f"   → {generated}")
        
        # 2. Low Temperature
        print("\n2. Low Temperature (T=0.5):")
        generated = generator.generate(
            prompt,
            max_length=40,
            temperature=0.5
        )
        print(f"   → {generated}")
        
        # 3. Normal Temperature
        print("\n3. Normal Temperature (T=1.0):")
        generated = generator.generate(
            prompt,
            max_length=40,
            temperature=1.0
        )
        print(f"   → {generated}")
        
        # 4. High Temperature
        print("\n4. High Temperature (T=2.0):")
        generated = generator.generate(
            prompt,
            max_length=40,
            temperature=2.0
        )
        print(f"   → {generated}")
        
        # 5. Top-k Sampling
        print("\n5. Top-k Sampling (k=5, T=0.8):")
        generated = generator.generate(
            prompt,
            max_length=40,
            temperature=0.8,
            top_k=5
        )
        print(f"   → {generated}")
        
        # 6. Top-p (Nucleus) Sampling
        print("\n6. Top-p Sampling (p=0.9, T=0.8):")
        generated = generator.generate(
            prompt,
            max_length=40,
            temperature=0.8,
            top_p=0.9
        )
        print(f"   → {generated}")
        
        # 7. Combined with Repetition Penalty
        print("\n7. With Repetition Penalty (penalty=1.2):")
        generated = generator.generate(
            prompt,
            max_length=40,
            temperature=0.8,
            top_k=10,
            repetition_penalty=1.2
        )
        print(f"   → {generated}")


def interactive_generation():
    """Interactive text generation"""
    print("\n" + "=" * 60)
    print("Interactive Text Generation")
    print("=" * 60)
    print("Enter 'quit' to exit")
    print("-" * 60)
    
    # Setup
    sample_text = "The quick brown fox jumps over the lazy dog. " * 10
    tokenizer = CharacterTokenizer()
    tokenizer.fit(sample_text)
    model = PretrainedMiniLM(tokenizer.vocab_size)
    generator = TextGenerator(model, tokenizer)
    
    while True:
        # Get user input
        prompt = input("\nEnter prompt: ")
        if prompt.lower() == 'quit':
            break
        
        # Get generation parameters
        print("\nGeneration parameters (press Enter for defaults):")
        
        try:
            max_len = input("  Max length (50): ")
            max_len = int(max_len) if max_len else 50
            
            temp = input("  Temperature (0.8): ")
            temp = float(temp) if temp else 0.8
            
            top_k_input = input("  Top-k (0 for disabled): ")
            top_k = int(top_k_input) if top_k_input else None
            
            top_p_input = input("  Top-p (0.9): ")
            top_p = float(top_p_input) if top_p_input else 0.9
            
        except ValueError:
            print("Invalid input, using defaults")
            max_len, temp, top_k, top_p = 50, 0.8, None, 0.9
        
        # Generate
        print("\nGenerating...")
        generated = generator.generate(
            prompt,
            max_length=max_len,
            temperature=temp,
            top_k=top_k,
            top_p=top_p
        )
        
        print(f"\nGenerated text:\n{generated}")


def analyze_sampling_diversity():
    """Analyze diversity of different sampling methods"""
    print("\n" + "=" * 60)
    print("Sampling Diversity Analysis")
    print("=" * 60)
    
    # Setup
    sample_text = "The quick brown fox jumps over the lazy dog."
    tokenizer = CharacterTokenizer()
    tokenizer.fit(sample_text)
    model = PretrainedMiniLM(tokenizer.vocab_size)
    generator = TextGenerator(model, tokenizer)
    
    prompt = "The "
    num_samples = 10
    
    strategies = [
        ("Greedy (T=0)", {"temperature": 0}),
        ("Low Temp (T=0.5)", {"temperature": 0.5}),
        ("Normal (T=1.0)", {"temperature": 1.0}),
        ("High Temp (T=2.0)", {"temperature": 2.0}),
        ("Top-k=5", {"top_k": 5, "temperature": 0.8}),
        ("Top-p=0.9", {"top_p": 0.9, "temperature": 0.8}),
    ]
    
    for name, params in strategies:
        print(f"\n{name}:")
        
        # Generate multiple samples
        samples = []
        for _ in range(num_samples):
            generated = generator.generate(
                prompt,
                max_length=30,
                **params
            )
            samples.append(generated)
        
        # Calculate diversity metrics
        unique_samples = len(set(samples))
        
        # Character diversity
        all_chars = ''.join(samples)
        unique_chars = len(set(all_chars))
        
        print(f"  Unique samples: {unique_samples}/{num_samples}")
        print(f"  Unique characters used: {unique_chars}")
        
        # Show first 3 samples
        print("  Sample outputs:")
        for i, sample in enumerate(samples[:3]):
            print(f"    {i+1}. {sample[:40]}...")


def main():
    print("=" * 60)
    print("Text Generation Demo")
    print("=" * 60)
    
    # Menu
    print("\nSelect demo mode:")
    print("1. Compare generation strategies")
    print("2. Interactive generation")
    print("3. Analyze sampling diversity")
    print("4. Run all demos")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == '1':
        demonstrate_generation_strategies()
    elif choice == '2':
        interactive_generation()
    elif choice == '3':
        analyze_sampling_diversity()
    elif choice == '4':
        demonstrate_generation_strategies()
        analyze_sampling_diversity()
        print("\n(Skipping interactive mode in batch run)")
    else:
        print("Invalid choice. Running strategy comparison...")
        demonstrate_generation_strategies()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()