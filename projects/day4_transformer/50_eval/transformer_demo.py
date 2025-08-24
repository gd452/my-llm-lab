"""
Simple Transformer demo - Sequence to Sequence translation toy example
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
from core.transformer import Transformer, Tensor


def create_toy_dataset():
    """Create a toy translation dataset (number reversal task)"""
    # Simple task: reverse sequences of numbers
    # e.g., [1, 2, 3] -> [3, 2, 1]
    
    src_sequences = [
        [1, 2, 3, 0, 0],      # padding with 0
        [4, 5, 6, 7, 0],
        [8, 9, 0, 0, 0],
        [1, 2, 3, 4, 5],
    ]
    
    tgt_sequences = [
        [3, 2, 1, 0, 0],      # reversed
        [7, 6, 5, 4, 0],
        [9, 8, 0, 0, 0],
        [5, 4, 3, 2, 1],
    ]
    
    return np.array(src_sequences), np.array(tgt_sequences)


def train_step(model, src, tgt, learning_rate=0.001):
    """Single training step"""
    # Teacher forcing: use true target as input (shifted)
    tgt_input = tgt[:, :-1]  # All except last
    tgt_output = tgt[:, 1:]  # All except first
    
    # Forward pass
    logits = model.forward(src, tgt_input)
    
    # Simple cross-entropy loss (simplified)
    batch_size, seq_len, vocab_size = logits.shape
    loss = Tensor(0.0)
    
    for b in range(batch_size):
        for t in range(seq_len):
            target_idx = tgt_output[b, t]
            if target_idx != 0:  # Skip padding
                # Softmax
                exp_logits = (logits.data[b, t] - np.max(logits.data[b, t]))
                exp_vals = np.exp(exp_logits)
                probs = exp_vals / np.sum(exp_vals)
                
                # Cross-entropy
                loss = loss + Tensor(-np.log(probs[target_idx] + 1e-8))
    
    # Backward pass
    loss.backward()
    
    # Simple gradient descent update
    for param in [model.src_embedding, model.tgt_embedding, model.output_proj]:
        if param.grad is not None:
            param.data -= learning_rate * param.grad
            param.grad = None
    
    return loss.data


def generate(model, src, max_len=10):
    """Generate output sequence"""
    # Start with begin token (let's use 1)
    tgt = np.array([[1]])  # Shape: (1, 1)
    
    for _ in range(max_len - 1):
        # Get predictions
        logits = model.forward(src[np.newaxis, :], tgt)
        
        # Get last token prediction
        last_logits = logits.data[0, -1]  # (vocab_size,)
        
        # Greedy decoding
        next_token = np.argmax(last_logits)
        
        # Stop if we hit padding or end
        if next_token == 0:
            break
        
        # Append to target sequence
        tgt = np.concatenate([tgt, [[next_token]]], axis=1)
    
    return tgt[0]


def main():
    print("=" * 60)
    print("Transformer Demo - Number Reversal Task")
    print("=" * 60)
    
    # Create toy dataset
    src_data, tgt_data = create_toy_dataset()
    print("\nDataset examples:")
    for i in range(2):
        print(f"  Source: {src_data[i]} -> Target: {tgt_data[i]}")
    
    # Create small transformer
    model = Transformer(
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_model=32,
        n_heads=4,
        d_ff=64,
        vocab_size=20,  # Small vocabulary
        max_seq_len=10,
        dropout=0.0
    )
    
    print("\nModel created:")
    print(f"  Encoder layers: 2")
    print(f"  Decoder layers: 2")
    print(f"  Model dimension: 32")
    print(f"  Attention heads: 4")
    
    # Training loop (simplified)
    print("\nTraining...")
    n_epochs = 50
    
    for epoch in range(n_epochs):
        total_loss = 0
        for i in range(len(src_data)):
            src_batch = src_data[i:i+1]
            tgt_batch = tgt_data[i:i+1]
            
            loss = train_step(model, src_batch, tgt_batch)
            total_loss += loss
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(src_data)
            print(f"  Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    # Test generation
    print("\nTesting generation:")
    for i in range(len(src_data)):
        src = src_data[i]
        true_tgt = tgt_data[i]
        
        # Generate output
        generated = generate(model, src)
        
        print(f"\n  Input:     {src}")
        print(f"  Expected:  {true_tgt}")
        print(f"  Generated: {generated}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("Note: This is a toy example. Real Transformers need:")
    print("  - Proper optimization (Adam)")
    print("  - Better initialization")
    print("  - More training data")
    print("  - Beam search for generation")
    print("=" * 60)


if __name__ == "__main__":
    main()