"""
Day 5: Training Components
Loss functions, optimizers, and training utilities
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from .autograd import Value


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray, 
                       ignore_index: int = -100) -> float:
    """
    Cross-entropy loss for language modeling
    
    Args:
        logits: Predictions (batch_size, seq_len, vocab_size)
        targets: Ground truth token ids (batch_size, seq_len)
        ignore_index: Index to ignore in loss calculation (e.g., padding)
    
    Returns:
        Average loss value
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for easier calculation
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    # Softmax
    exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Calculate loss
    total_loss = 0.0
    valid_count = 0
    
    for i, target in enumerate(targets_flat):
        if target != ignore_index:
            # Negative log likelihood
            total_loss += -np.log(probs[i, target] + 1e-8)
            valid_count += 1
    
    return total_loss / valid_count if valid_count > 0 else 0.0


def perplexity(loss: float) -> float:
    """Calculate perplexity from loss"""
    return np.exp(loss)


class Optimizer:
    """Base optimizer class"""
    
    def __init__(self, parameters: List, learning_rate: float = 0.001):
        self.parameters = parameters
        self.learning_rate = learning_rate
    
    def step(self):
        """Update parameters"""
        raise NotImplementedError
    
    def zero_grad(self):
        """Reset gradients to zero"""
        for param in self.parameters:
            if hasattr(param, 'grad'):
                param.grad = None


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, parameters: List, learning_rate: float = 0.01, 
                 momentum: float = 0.0):
        super().__init__(parameters, learning_rate)
        self.momentum = momentum
        self.velocities = {}
        
        # Initialize velocities
        for i, param in enumerate(parameters):
            self.velocities[i] = np.zeros_like(param.data if hasattr(param, 'data') else param)
    
    def step(self):
        """Update parameters using SGD with optional momentum"""
        for i, param in enumerate(self.parameters):
            if hasattr(param, 'grad') and param.grad is not None:
                # Get gradient
                grad = param.grad
                
                # Apply momentum
                if self.momentum > 0:
                    self.velocities[i] = self.momentum * self.velocities[i] - self.learning_rate * grad
                    update = self.velocities[i]
                else:
                    update = -self.learning_rate * grad
                
                # Update parameter
                if hasattr(param, 'data'):
                    param.data += update
                else:
                    param += update


class Adam(Optimizer):
    """Adam optimizer"""
    
    def __init__(self, parameters: List, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(parameters, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # Time step
        
        # Initialize moments
        self.m = {}  # First moment (mean)
        self.v = {}  # Second moment (variance)
        
        for i, param in enumerate(parameters):
            self.m[i] = np.zeros_like(param.data if hasattr(param, 'data') else param)
            self.v[i] = np.zeros_like(param.data if hasattr(param, 'data') else param)
    
    def step(self):
        """Update parameters using Adam"""
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if hasattr(param, 'grad') and param.grad is not None:
                grad = param.grad
                
                # Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                
                # Update biased second raw moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                # Update parameter
                update = -self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
                
                if hasattr(param, 'data'):
                    param.data += update
                else:
                    param += update


class TextGenerator:
    """Text generation with various sampling strategies"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate(self, 
                prompt: str,
                max_length: int = 100,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                repetition_penalty: float = 1.0) -> str:
        """
        Generate text with various sampling strategies
        
        Args:
            prompt: Starting text
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
        
        Returns:
            Generated text
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        generated_ids = input_ids.copy()
        
        # Track token frequencies for repetition penalty
        token_freq = {}
        for token_id in input_ids:
            token_freq[token_id] = token_freq.get(token_id, 0) + 1
        
        # Generate tokens
        for _ in range(max_length - len(input_ids)):
            # Get model predictions
            logits = self.model.forward(np.array([generated_ids]))
            next_token_logits = logits[0, -1, :]  # Get last position
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id, freq in token_freq.items():
                    next_token_logits[token_id] /= (repetition_penalty * freq)
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Convert to probabilities
            exp_logits = np.exp(next_token_logits - np.max(next_token_logits))
            probs = exp_logits / np.sum(exp_logits)
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_indices = np.argsort(probs)[-top_k:]
                top_k_probs = probs[top_k_indices]
                top_k_probs = top_k_probs / np.sum(top_k_probs)
                
                # Sample from top-k
                next_token_idx = np.random.choice(len(top_k_probs), p=top_k_probs)
                next_token = top_k_indices[next_token_idx]
            
            # Apply top-p (nucleus) filtering
            elif top_p is not None:
                sorted_indices = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_indices]
                
                # Find cutoff
                cumsum = np.cumsum(sorted_probs)
                cutoff_idx = np.argmax(cumsum >= top_p) + 1
                
                # Keep only tokens above cutoff
                nucleus_indices = sorted_indices[:cutoff_idx]
                nucleus_probs = sorted_probs[:cutoff_idx]
                nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
                
                # Sample from nucleus
                next_token_idx = np.random.choice(len(nucleus_probs), p=nucleus_probs)
                next_token = nucleus_indices[next_token_idx]
            
            # Greedy decoding
            elif temperature == 0:
                next_token = np.argmax(probs)
            
            # Standard sampling
            else:
                next_token = np.random.choice(len(probs), p=probs)
            
            # Add to generated sequence
            generated_ids.append(next_token)
            
            # Update frequency for repetition penalty
            token_freq[next_token] = token_freq.get(next_token, 0) + 1
            
            # Check for EOS token
            if hasattr(self.tokenizer, 'eos_token'):
                eos_id = self.tokenizer.char_to_id.get(self.tokenizer.eos_token)
                if eos_id and next_token == eos_id:
                    break
        
        # Decode to text
        generated_text = self.tokenizer.decode(generated_ids)
        return generated_text


def gradient_clipping(parameters: List, max_norm: float = 1.0):
    """
    Clip gradients to prevent exploding gradients
    
    Args:
        parameters: List of parameters with gradients
        max_norm: Maximum gradient norm
    """
    total_norm = 0.0
    
    # Calculate total norm
    for param in parameters:
        if hasattr(param, 'grad') and param.grad is not None:
            param_norm = np.sum(param.grad ** 2)
            total_norm += param_norm
    
    total_norm = np.sqrt(total_norm)
    
    # Clip if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        for param in parameters:
            if hasattr(param, 'grad') and param.grad is not None:
                param.grad *= clip_coef


def train_epoch(model, dataloader, optimizer, max_grad_norm: float = 1.0) -> float:
    """
    Train for one epoch
    
    Returns:
        Average loss for the epoch
    """
    total_loss = 0.0
    num_batches = 0
    
    for inputs, targets in dataloader:
        # Convert to numpy arrays
        inputs = np.array(inputs)
        targets = np.array(targets)
        
        # Forward pass
        logits = model.forward(inputs)
        loss = cross_entropy_loss(logits, targets)
        
        # Backward pass (simplified)
        model.backward(loss)
        
        # Gradient clipping
        gradient_clipping(model.parameters(), max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0