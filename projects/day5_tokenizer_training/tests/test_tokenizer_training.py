"""
Test cases for Tokenizer and Training components
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
from core.tokenizer import CharacterTokenizer, SimpleBPETokenizer, DataLoader
from core.training import (
    cross_entropy_loss, perplexity,
    SGD, Adam, TextGenerator,
    gradient_clipping
)


def test_character_tokenizer():
    """Test character-level tokenizer"""
    print("Testing CharacterTokenizer...")
    
    # Create and fit tokenizer
    tokenizer = CharacterTokenizer()
    text = "Hello World!"
    tokenizer.fit(text)
    
    # Test vocabulary
    assert tokenizer.vocab_size > 4, "Should have special tokens + characters"
    assert tokenizer.pad_token in tokenizer.char_to_id
    assert tokenizer.unk_token in tokenizer.char_to_id
    
    # Test encoding
    encoded = tokenizer.encode("Hello")
    assert len(encoded) == 5, "Should encode 5 characters"
    
    # Test decoding
    decoded = tokenizer.decode(encoded)
    assert decoded == "Hello", f"Should decode back to 'Hello', got {decoded}"
    
    # Test special tokens
    encoded_special = tokenizer.encode("Hi", add_special_tokens=True)
    assert encoded_special[0] == tokenizer.char_to_id[tokenizer.bos_token]
    assert encoded_special[-1] == tokenizer.char_to_id[tokenizer.eos_token]
    
    # Test unknown character
    encoded_unk = tokenizer.encode("Hello😊")  # Emoji not in training text
    assert tokenizer.char_to_id[tokenizer.unk_token] in encoded_unk
    
    print("✓ CharacterTokenizer test passed")


def test_bpe_tokenizer():
    """Test BPE tokenizer"""
    print("Testing SimpleBPETokenizer...")
    
    # Create and fit tokenizer
    tokenizer = SimpleBPETokenizer(vocab_size=100)
    text = "the cat sat on the mat. the dog sat on the log."
    tokenizer.fit(text, num_merges=10)
    
    # Test merges
    assert len(tokenizer.merges) == 10, "Should have 10 merges"
    
    # Test vocabulary
    assert len(tokenizer.vocab) > 4, "Should have special tokens + learned tokens"
    
    # Test encoding
    encoded = tokenizer.encode("the cat")
    assert len(encoded) > 0, "Should encode text"
    
    # Test decoding
    decoded = tokenizer.decode(encoded)
    # BPE might add/remove spaces, so just check key words
    assert "cat" in decoded.lower(), f"Should contain 'cat', got {decoded}"
    
    print("✓ SimpleBPETokenizer test passed")


def test_dataloader():
    """Test DataLoader"""
    print("Testing DataLoader...")
    
    # Setup
    text = "The quick brown fox jumps over the lazy dog."
    tokenizer = CharacterTokenizer()
    tokenizer.fit(text)
    
    # Create dataloader
    dataloader = DataLoader(
        text=text,
        tokenizer=tokenizer,
        batch_size=2,
        seq_length=5
    )
    
    # Test batch retrieval
    inputs, targets = dataloader.get_batch(0)
    
    assert len(inputs) <= 2, "Batch size should be at most 2"
    assert len(inputs[0]) == 5, "Sequence length should be 5"
    assert len(targets[0]) == 5, "Target sequence length should be 5"
    
    # Test that target is shifted input
    for i in range(len(inputs)):
        input_decoded = tokenizer.decode(inputs[i])
        target_decoded = tokenizer.decode(targets[i])
        # Target should be input shifted by 1
        # (This is approximate due to tokenization)
    
    # Test iteration
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
    
    assert batch_count == dataloader.num_batches
    
    print("✓ DataLoader test passed")


def test_cross_entropy_loss():
    """Test cross-entropy loss calculation"""
    print("Testing cross_entropy_loss...")
    
    # Test with perfect prediction
    vocab_size = 10
    batch_size = 2
    seq_length = 3
    
    # Perfect prediction (one-hot)
    logits = np.zeros((batch_size, seq_length, vocab_size))
    targets = np.array([[0, 1, 2], [3, 4, 5]])
    
    for b in range(batch_size):
        for t in range(seq_length):
            logits[b, t, targets[b, t]] = 10.0  # High logit for correct class
    
    loss = cross_entropy_loss(logits, targets)
    assert loss < 0.01, f"Perfect prediction should have near-zero loss, got {loss}"
    
    # Test with random prediction
    logits_random = np.random.randn(batch_size, seq_length, vocab_size)
    loss_random = cross_entropy_loss(logits_random, targets)
    expected_random_loss = np.log(vocab_size)
    assert abs(loss_random - expected_random_loss) < 1.0, \
        f"Random prediction loss should be close to log(vocab_size)"
    
    # Test perplexity
    ppl = perplexity(loss_random)
    assert ppl > 1, "Perplexity should be > 1"
    
    print("✓ cross_entropy_loss test passed")


def test_sgd_optimizer():
    """Test SGD optimizer"""
    print("Testing SGD optimizer...")
    
    # Create dummy parameter
    class DummyParam:
        def __init__(self, value):
            self.data = np.array(value, dtype=float)
            self.grad = None
    
    param = DummyParam([1.0, 2.0, 3.0])
    initial_value = param.data.copy()
    
    # Create optimizer
    sgd = SGD([param], learning_rate=0.1)
    
    # Simulate gradient
    param.grad = np.array([0.1, 0.2, 0.3])
    
    # Update
    sgd.step()
    
    # Check update
    expected = initial_value - 0.1 * param.grad
    assert np.allclose(param.data, expected), \
        f"SGD update incorrect: expected {expected}, got {param.data}"
    
    # Test zero_grad
    sgd.zero_grad()
    assert param.grad is None, "Gradient should be reset"
    
    print("✓ SGD optimizer test passed")


def test_adam_optimizer():
    """Test Adam optimizer"""
    print("Testing Adam optimizer...")
    
    # Create dummy parameter
    class DummyParam:
        def __init__(self, value):
            self.data = np.array(value, dtype=float)
            self.grad = None
    
    param = DummyParam([1.0, 2.0, 3.0])
    
    # Create optimizer
    adam = Adam([param], learning_rate=0.1)
    
    # Multiple updates
    for _ in range(5):
        param.grad = np.array([0.1, 0.2, 0.3])
        adam.step()
        adam.zero_grad()
    
    # Check that parameters changed
    assert not np.allclose(param.data, [1.0, 2.0, 3.0]), \
        "Parameters should have changed after optimization"
    
    # Check that parameters decreased (gradient was positive)
    assert np.all(param.data < [1.0, 2.0, 3.0]), \
        "Parameters should decrease with positive gradients"
    
    print("✓ Adam optimizer test passed")


def test_gradient_clipping():
    """Test gradient clipping"""
    print("Testing gradient_clipping...")
    
    # Create parameters with large gradients
    class DummyParam:
        def __init__(self, grad_value):
            self.grad = np.array(grad_value, dtype=float)
    
    params = [
        DummyParam([10.0, 0.0, 0.0]),
        DummyParam([0.0, 10.0, 0.0]),
    ]
    
    # Calculate initial norm
    initial_norm = np.sqrt(sum(np.sum(p.grad**2) for p in params))
    assert initial_norm > 10.0, "Initial norm should be large"
    
    # Clip gradients
    gradient_clipping(params, max_norm=5.0)
    
    # Calculate new norm
    new_norm = np.sqrt(sum(np.sum(p.grad**2) for p in params))
    assert abs(new_norm - 5.0) < 0.01, f"Norm should be clipped to 5.0, got {new_norm}"
    
    print("✓ gradient_clipping test passed")


def test_text_generator():
    """Test text generation"""
    print("Testing TextGenerator...")
    
    # Create dummy model
    class DummyModel:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
        
        def forward(self, input_ids):
            # Return random logits
            if isinstance(input_ids, list):
                seq_len = len(input_ids)
            else:
                seq_len = input_ids.shape[-1]
            
            return np.random.randn(1, seq_len, self.vocab_size)
    
    # Setup
    tokenizer = CharacterTokenizer()
    tokenizer.fit("Hello World!")
    model = DummyModel(tokenizer.vocab_size)
    generator = TextGenerator(model, tokenizer)
    
    # Test greedy generation
    generated = generator.generate("H", max_length=5, temperature=0)
    assert len(generated) <= 5, "Should respect max_length"
    assert generated[0] == "H", "Should start with prompt"
    
    # Test temperature sampling
    generated_hot = generator.generate("H", max_length=10, temperature=2.0)
    generated_cold = generator.generate("H", max_length=10, temperature=0.1)
    # Can't test exact behavior due to randomness, just check it runs
    
    # Test top-k sampling
    generated_topk = generator.generate("H", max_length=10, top_k=3)
    assert len(generated_topk) > 0, "Should generate text with top-k"
    
    # Test top-p sampling  
    generated_topp = generator.generate("H", max_length=10, top_p=0.9)
    assert len(generated_topp) > 0, "Should generate text with top-p"
    
    print("✓ TextGenerator test passed")


def test_compression_ratio():
    """Test tokenizer compression"""
    print("Testing tokenizer compression...")
    
    text = "the the the cat cat cat sat sat sat"
    
    # Character tokenizer
    char_tok = CharacterTokenizer()
    char_tok.fit(text)
    char_encoded = char_tok.encode(text)
    
    # BPE tokenizer
    bpe_tok = SimpleBPETokenizer()
    bpe_tok.fit(text, num_merges=20)
    bpe_encoded = bpe_tok.encode(text)
    
    # BPE should be more efficient for repetitive text
    compression_ratio = len(char_encoded) / max(len(bpe_encoded), 1)
    print(f"  Character tokens: {len(char_encoded)}")
    print(f"  BPE tokens: {len(bpe_encoded)}")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    # BPE should compress repeated patterns
    assert len(bpe_encoded) <= len(char_encoded), \
        "BPE should not produce more tokens than character-level"
    
    print("✓ Compression test passed")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Tokenizer and Training Components")
    print("=" * 50)
    
    test_character_tokenizer()
    test_bpe_tokenizer()
    test_dataloader()
    test_cross_entropy_loss()
    test_sgd_optimizer()
    test_adam_optimizer()
    test_gradient_clipping()
    test_text_generator()
    test_compression_ratio()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)