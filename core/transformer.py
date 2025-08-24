"""
Day 4: Transformer Implementation
Complete Transformer architecture with Encoder-Decoder structure
"""

import numpy as np
from typing import Optional, Tuple
from .attention import MultiHeadAttention


class Tensor:
    """Simple Tensor wrapper for numpy arrays with autograd-like interface"""
    
    def __init__(self, data):
        if isinstance(data, (int, float)):
            self.data = np.array(data)
        else:
            self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.grad = None
        self.shape = self.data.shape
    
    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        return Tensor(self.data + other)
    
    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        return Tensor(self.data - other)
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        return Tensor(self.data * other)
    
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        return Tensor(self.data / other)
    
    def __matmul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data @ other.data)
        return Tensor(self.data @ other)
    
    def __pow__(self, power):
        return Tensor(self.data ** power)
    
    def mean(self, axis=None, keepdims=False):
        return Tensor(np.mean(self.data, axis=axis, keepdims=keepdims))
    
    def sum(self):
        return Tensor(np.sum(self.data))
    
    def relu(self):
        return Tensor(np.maximum(0, self.data))
    
    def backward(self):
        # Simplified backward for demo
        self.grad = np.ones_like(self.data)
    
    def __getitem__(self, key):
        return Tensor(self.data[key])


class LayerNorm:
    """Layer Normalization for stable training"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps
        
        # Learnable parameters
        self.gamma = Tensor(np.ones(d_model))  # Scale
        self.beta = Tensor(np.zeros(d_model))  # Shift
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        Returns:
            Normalized tensor
        """
        # Calculate mean and variance along last dimension
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / Tensor(np.sqrt(var.data + self.eps))
        
        # Scale and shift
        return self.gamma * x_norm + self.beta


class PositionwiseFeedForward:
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (typically 4 * d_model)
            dropout: Dropout rate
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Two linear transformations with ReLU
        self.w1 = Tensor(np.random.randn(d_model, d_ff) * 0.02)
        self.b1 = Tensor(np.zeros(d_ff))
        self.w2 = Tensor(np.random.randn(d_ff, d_model) * 0.02)
        self.b2 = Tensor(np.zeros(d_model))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply position-wise feed-forward network
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        Returns:
            Output tensor
        """
        # First linear + ReLU
        hidden = x @ self.w1 + self.b1
        hidden = hidden.relu()
        
        # Dropout (training only)
        if self.dropout > 0:
            # Simple dropout implementation
            mask = np.random.binomial(1, 1 - self.dropout, hidden.shape) / (1 - self.dropout)
            hidden = hidden * Tensor(mask)
        
        # Second linear
        output = hidden @ self.w2 + self.b2
        
        return output


class TransformerBlock:
    """Single Transformer block (used in both Encoder and Decoder)"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float = 0.1, is_decoder: bool = False):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            is_decoder: Whether this is a decoder block
        """
        self.is_decoder = is_decoder
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        
        # Cross-attention (decoder only)
        if is_decoder:
            self.cross_attn = MultiHeadAttention(d_model, n_heads)
            self.norm2 = LayerNorm(d_model)
        
        # Position-wise feed-forward
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = LayerNorm(d_model)
        
        self.dropout = dropout
    
    def forward(self, x: Tensor, 
                encoder_output: Optional[Tensor] = None,
                self_mask: Optional[Tensor] = None,
                cross_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through transformer block
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            encoder_output: Encoder output for cross-attention (decoder only)
            self_mask: Mask for self-attention
            cross_mask: Mask for cross-attention
        Returns:
            Output tensor
        """
        # Self-attention with residual connection
        # Convert Tensor to numpy for MultiHeadAttention
        x_np = x.data if isinstance(x, Tensor) else x
        mask_np = self_mask.data if isinstance(self_mask, Tensor) and self_mask is not None else self_mask
        attn_output, _ = self.self_attn.forward(x_np, x_np, x_np, mask_np)
        attn_output = Tensor(attn_output)
        x = self.norm1.forward(x + self._dropout(attn_output))
        
        # Cross-attention (decoder only)
        if self.is_decoder and encoder_output is not None:
            x_np = x.data if isinstance(x, Tensor) else x
            enc_np = encoder_output.data if isinstance(encoder_output, Tensor) else encoder_output
            cross_mask_np = cross_mask.data if isinstance(cross_mask, Tensor) and cross_mask is not None else cross_mask
            cross_output, _ = self.cross_attn.forward(x_np, enc_np, enc_np, cross_mask_np)
            cross_output = Tensor(cross_output)
            x = self.norm2.forward(x + self._dropout(cross_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn.forward(x)
        x = self.norm3.forward(x + self._dropout(ffn_output))
        
        return x
    
    def _dropout(self, x: Tensor) -> Tensor:
        """Apply dropout"""
        if self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, x.shape) / (1 - self.dropout)
            return x * Tensor(mask)
        return x


class TransformerEncoder:
    """Stack of Transformer encoder blocks"""
    
    def __init__(self, n_layers: int, d_model: int, n_heads: int, 
                 d_ff: int, dropout: float = 0.1):
        """
        Args:
            n_layers: Number of encoder layers
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        self.layers = [
            TransformerBlock(d_model, n_heads, d_ff, dropout, is_decoder=False)
            for _ in range(n_layers)
        ]
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through encoder stack
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask
        Returns:
            Encoder output
        """
        for layer in self.layers:
            x = layer.forward(x, self_mask=mask)
        return x


class TransformerDecoder:
    """Stack of Transformer decoder blocks"""
    
    def __init__(self, n_layers: int, d_model: int, n_heads: int, 
                 d_ff: int, dropout: float = 0.1):
        """
        Args:
            n_layers: Number of decoder layers
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        self.layers = [
            TransformerBlock(d_model, n_heads, d_ff, dropout, is_decoder=True)
            for _ in range(n_layers)
        ]
    
    def forward(self, x: Tensor, encoder_output: Tensor,
                self_mask: Optional[Tensor] = None,
                cross_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through decoder stack
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            encoder_output: Output from encoder
            self_mask: Mask for self-attention (causal mask)
            cross_mask: Mask for cross-attention
        Returns:
            Decoder output
        """
        for layer in self.layers:
            x = layer.forward(x, encoder_output, self_mask, cross_mask)
        return x


class Transformer:
    """Complete Transformer model with Encoder-Decoder architecture"""
    
    def __init__(self, 
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 d_model: int = 512,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 vocab_size: int = 10000,
                 max_seq_len: int = 100,
                 dropout: float = 0.1):
        """
        Args:
            n_encoder_layers: Number of encoder layers
            n_decoder_layers: Number of decoder layers  
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            vocab_size: Vocabulary size
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.src_embedding = Tensor(np.random.randn(vocab_size, d_model) * 0.02)
        self.tgt_embedding = Tensor(np.random.randn(vocab_size, d_model) * 0.02)
        
        # Positional encoding (from day3)
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Encoder and Decoder stacks
        self.encoder = TransformerEncoder(n_encoder_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(n_decoder_layers, d_model, n_heads, d_ff, dropout)
        
        # Output projection
        self.output_proj = Tensor(np.random.randn(d_model, vocab_size) * 0.02)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> Tensor:
        """Create sinusoidal positional encoding"""
        pos = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)
        
        return Tensor(pe)
    
    def forward(self, src: np.ndarray, tgt: np.ndarray,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through Transformer
        
        Args:
            src: Source sequence (batch_size, src_len) - token indices
            tgt: Target sequence (batch_size, tgt_len) - token indices
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal)
        Returns:
            Output logits (batch_size, tgt_len, vocab_size)
        """
        batch_size, src_len = src.shape
        _, tgt_len = tgt.shape
        
        # Source embedding + positional encoding
        src_emb = self._embed(src, self.src_embedding)
        src_emb = src_emb + self.pos_encoding[:src_len]
        
        # Target embedding + positional encoding  
        tgt_emb = self._embed(tgt, self.tgt_embedding)
        tgt_emb = tgt_emb + self.pos_encoding[:tgt_len]
        
        # Encode source
        encoder_output = self.encoder.forward(src_emb, src_mask)
        
        # Decode with cross-attention to encoder output
        decoder_output = self.decoder.forward(
            tgt_emb, encoder_output, tgt_mask, src_mask
        )
        
        # Project to vocabulary
        logits = decoder_output @ self.output_proj
        
        return logits
    
    def _embed(self, indices: np.ndarray, embedding: Tensor) -> Tensor:
        """Convert token indices to embeddings"""
        batch_size, seq_len = indices.shape
        embedded = np.zeros((batch_size, seq_len, self.d_model))
        
        for b in range(batch_size):
            for t in range(seq_len):
                embedded[b, t] = embedding.data[indices[b, t]]
        
        return Tensor(embedded) * Tensor(np.sqrt(self.d_model))
    
    @staticmethod
    def create_causal_mask(seq_len: int) -> Tensor:
        """Create causal mask for decoder self-attention"""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return Tensor(mask)