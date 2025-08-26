"""
Train a mini language model - complete training pipeline demo
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import matplotlib.pyplot as plt
from core.tokenizer import CharacterTokenizer, DataLoader
from core.training import (
    cross_entropy_loss, perplexity,
    Adam, gradient_clipping
)


class MiniLanguageModel:
    """Simple language model for demonstration"""
    
    def __init__(self, vocab_size, hidden_size=128, num_layers=2):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initialize parameters
        self.embedding = np.random.randn(vocab_size, hidden_size) * 0.02
        
        # Simple RNN-like layers
        self.rnn_weights = []
        self.rnn_biases = []
        for _ in range(num_layers):
            self.rnn_weights.append(
                np.random.randn(hidden_size, hidden_size) * 0.02
            )
            self.rnn_biases.append(np.zeros(hidden_size))
        
        # Output projection
        self.output_weight = np.random.randn(hidden_size, vocab_size) * 0.02
        self.output_bias = np.zeros(vocab_size)
        
        # Hidden state
        self.hidden_states = None
    
    def forward(self, input_ids, training=True):
        """Forward pass"""
        if isinstance(input_ids, list):
            input_ids = np.array(input_ids)
        
        if len(input_ids.shape) == 1:
            input_ids = input_ids[np.newaxis, :]
        
        batch_size, seq_length = input_ids.shape
        
        # Embedding lookup
        embedded = np.zeros((batch_size, seq_length, self.hidden_size))
        for b in range(batch_size):
            for t in range(seq_length):
                embedded[b, t] = self.embedding[input_ids[b, t]]
        
        # Process through RNN-like layers
        hidden = embedded
        for layer in range(self.num_layers):
            new_hidden = np.zeros_like(hidden)
            
            for t in range(seq_length):
                if t == 0:
                    # First timestep
                    h = np.tanh(hidden[:, t] @ self.rnn_weights[layer] + 
                               self.rnn_biases[layer])
                else:
                    # Use previous hidden state
                    h = np.tanh((hidden[:, t] + new_hidden[:, t-1]) @ 
                               self.rnn_weights[layer] + self.rnn_biases[layer])
                
                new_hidden[:, t] = h
            
            hidden = new_hidden
        
        # Output projection
        logits = hidden @ self.output_weight + self.output_bias
        
        return logits
    
    def parameters(self):
        """Get all parameters for optimization"""
        params = [self.embedding, self.output_weight, self.output_bias]
        params.extend(self.rnn_weights)
        params.extend(self.rnn_biases)
        return params
    
    def backward(self, loss):
        """Simplified backward pass - just create dummy gradients"""
        # In real implementation, this would compute actual gradients
        for param in self.parameters():
            param.grad = np.random.randn(*param.shape) * 0.01


def load_training_data():
    """Load and prepare training data"""
    # Sample text - in practice, load from file
    text = """
    The quick brown fox jumps over the lazy dog.
    A journey of a thousand miles begins with a single step.
    To be or not to be, that is the question.
    All that glitters is not gold.
    Actions speak louder than words.
    The early bird catches the worm.
    Better late than never.
    Practice makes perfect.
    Time flies when you're having fun.
    Knowledge is power.
    """ * 5  # Repeat for more data
    
    return text


def train_model(model, train_data, tokenizer, epochs=10, batch_size=4, seq_length=32):
    """Train the language model"""
    
    # Create dataloader
    dataloader = DataLoader(
        text=train_data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_length=seq_length
    )
    
    # Create optimizer
    optimizer = Adam(model.parameters(), learning_rate=0.001)
    
    # Training history
    train_losses = []
    train_perplexities = []
    
    print("Starting training...")
    print(f"Total batches per epoch: {len(dataloader)}")
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Convert to arrays
            inputs = np.array(inputs)
            targets = np.array(targets)
            
            # Forward pass
            logits = model.forward(inputs, training=True)
            loss = cross_entropy_loss(logits, targets)
            
            # Backward pass
            model.backward(loss)
            
            # Gradient clipping
            gradient_clipping(model.parameters(), max_norm=5.0)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss
            batch_count += 1
            
            # Print progress
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss:.4f}")
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        ppl = perplexity(avg_loss)
        
        train_losses.append(avg_loss)
        train_perplexities.append(ppl)
        
        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {ppl:.2f}")
        print("-" * 40)
    
    return train_losses, train_perplexities


def plot_training_curves(losses, perplexities):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Perplexity curve
    ax2.plot(perplexities, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Training Perplexity')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=100)
    plt.show()


def evaluate_model(model, tokenizer, test_text):
    """Evaluate model on test text"""
    print("\nEvaluating model...")
    
    # Tokenize test text
    test_ids = tokenizer.encode(test_text)
    
    # Create batches
    batch_size = 1
    seq_length = 32
    
    total_loss = 0
    num_batches = 0
    
    for i in range(0, len(test_ids) - seq_length - 1, seq_length):
        inputs = test_ids[i:i+seq_length]
        targets = test_ids[i+1:i+seq_length+1]
        
        inputs = np.array([inputs])
        targets = np.array([targets])
        
        # Forward pass
        logits = model.forward(inputs, training=False)
        loss = cross_entropy_loss(logits, targets)
        
        total_loss += loss
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    ppl = perplexity(avg_loss)
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Perplexity: {ppl:.2f}")
    
    return avg_loss, ppl


def main():
    print("=" * 60)
    print("Mini Language Model Training Pipeline")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading training data...")
    train_data = load_training_data()
    print(f"   Training data size: {len(train_data)} characters")
    
    # Create tokenizer
    print("\n2. Creating tokenizer...")
    tokenizer = CharacterTokenizer()
    tokenizer.fit(train_data)
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model
    print("\n3. Creating model...")
    model = MiniLanguageModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2
    )
    print(f"   Model parameters:")
    print(f"   - Vocab size: {tokenizer.vocab_size}")
    print(f"   - Hidden size: 64")
    print(f"   - Num layers: 2")
    
    # Train model
    print("\n4. Training model...")
    losses, perplexities = train_model(
        model=model,
        train_data=train_data,
        tokenizer=tokenizer,
        epochs=5,
        batch_size=4,
        seq_length=32
    )
    
    # Plot curves
    print("\n5. Plotting training curves...")
    plot_training_curves(losses, perplexities)
    
    # Evaluate
    print("\n6. Evaluating model...")
    test_text = "The quick brown fox jumps over the lazy dog."
    test_loss, test_ppl = evaluate_model(model, tokenizer, test_text)
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final Training Loss: {losses[-1]:.4f}")
    print(f"Final Training Perplexity: {perplexities[-1]:.2f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {test_ppl:.2f}")
    print("=" * 60)
    
    # Save model info
    print("\nModel training completed successfully!")
    print("Note: In practice, you would save model weights here.")


if __name__ == "__main__":
    main()