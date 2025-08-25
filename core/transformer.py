"""
Day 4: Transformer Implementation
Complete Transformer architecture with Encoder-Decoder structure
PyTorch version for actual learning capability
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. Linear projections in batch
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Apply attention
        attn_output, _ = self.attention(Q, K, V, mask)
        
        # 3. Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 4. Final linear layer
        output = self.W_o(attn_output)
        
        return output
    
    def attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class LayerNorm(nn.Module):
    """Layer Normalization"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Single Transformer block"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float = 0.1, is_decoder: bool = False):
        super().__init__()
        self.is_decoder = is_decoder
        
        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        
        # Cross-attention (decoder only)
        if is_decoder:
            self.cross_attn = MultiHeadAttention(d_model, n_heads)
            self.norm2 = LayerNorm(d_model)
        
        # Feed-forward
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output=None, self_mask=None, cross_mask=None):
        # Self-attention with residual
        attn_output = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention (decoder only)
        if self.is_decoder and encoder_output is not None:
            cross_output = self.cross_attn(x, encoder_output, encoder_output, cross_mask)
            x = self.norm2(x + self.dropout(cross_output))
        
        # Feed-forward with residual
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder blocks"""
    
    def __init__(self, n_layers: int, d_model: int, n_heads: int, 
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, is_decoder=False)
            for _ in range(n_layers)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, self_mask=mask)
        return x


class TransformerDecoder(nn.Module):
    """Stack of Transformer decoder blocks"""
    
    def __init__(self, n_layers: int, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, is_decoder=True)
            for _ in range(n_layers)
        ])
    
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask)
        return x


class Transformer(nn.Module):
    """Complete Transformer model with Encoder-Decoder architecture"""
    
    def __init__(self,
                 vocab_size: int = 10000,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_len: int = 100,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.register_buffer('pos_encoding', self._create_positional_encoding(max_seq_len, d_model))
        
        # Encoder and Decoder
        self.encoder = TransformerEncoder(n_encoder_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(n_decoder_layers, d_model, n_heads, d_ff, dropout)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _create_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass
        
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
            src_mask: Source mask
            tgt_mask: Target mask (causal)
        
        Returns:
            Output logits (batch_size, tgt_len, vocab_size)
        """
        # Embed and add positional encoding
        src_emb = self.src_embedding(src) * np.sqrt(self.d_model)
        src_emb = src_emb + self.pos_encoding[:, :src.size(1), :]
        src_emb = self.dropout(src_emb)
        
        tgt_emb = self.tgt_embedding(tgt) * np.sqrt(self.d_model)
        tgt_emb = tgt_emb + self.pos_encoding[:, :tgt.size(1), :]
        tgt_emb = self.dropout(tgt_emb)
        
        # Create masks if not provided
        if tgt_mask is None:
            tgt_mask = self.create_causal_mask(tgt.size(1)).to(tgt.device)
        
        # Encode
        encoder_output = self.encoder(src_emb, src_mask)
        
        # Decode
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask)
        
        # Project to vocabulary
        output = self.output_proj(decoder_output)
        
        return output
    
    @staticmethod
    def create_causal_mask(seq_len: int):
        """Create causal mask for decoder self-attention"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask == 0  # Convert to boolean mask
    
    def generate(self, src, max_length=50, start_token=1, end_token=0):
        """
        Generate sequence using greedy decoding
        
        Args:
            src: Source sequence (batch_size, src_len)
            max_length: Maximum generation length
            start_token: Start token id
            end_token: End token id
        
        Returns:
            Generated sequence
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        with torch.no_grad():
            # Encode source
            src_emb = self.src_embedding(src) * np.sqrt(self.d_model)
            src_emb = src_emb + self.pos_encoding[:, :src.size(1), :]
            encoder_output = self.encoder(src_emb)
            
            # Start with start token
            tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
            
            for _ in range(max_length - 1):
                # Create mask
                tgt_mask = self.create_causal_mask(tgt.size(1)).to(device)
                
                # Embed target
                tgt_emb = self.tgt_embedding(tgt) * np.sqrt(self.d_model)
                tgt_emb = tgt_emb + self.pos_encoding[:, :tgt.size(1), :]
                
                # Decode
                decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask)
                
                # Get next token
                logits = self.output_proj(decoder_output[:, -1, :])
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Stop if all sequences hit end token
                if (next_token == end_token).all():
                    break
        
        return tgt
    
    def generate_simple(self, src):
        """
        Simple generation for copy task
        
        Args:
            src: Source sequence (batch_size, src_len)
        
        Returns:
            Generated sequence (same as input for copy task)
        """
        self.eval()
        batch_size = src.size(0)
        seq_len = src.size(1)
        device = src.device
        
        with torch.no_grad():
            # Encode source
            src_emb = self.src_embedding(src) * np.sqrt(self.d_model)
            src_emb = src_emb + self.pos_encoding[:, :src.size(1), :]
            encoder_output = self.encoder(src_emb)
            
            # Start with first token
            tgt = src[:, :1].clone()
            
            for i in range(seq_len - 1):
                # Create mask
                tgt_mask = self.create_causal_mask(tgt.size(1)).to(device)
                
                # Embed target
                tgt_emb = self.tgt_embedding(tgt) * np.sqrt(self.d_model)
                tgt_emb = tgt_emb + self.pos_encoding[:, :tgt.size(1), :]
                
                # Decode
                decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask)
                
                # Get next token
                logits = self.output_proj(decoder_output[:, -1, :])
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append
                tgt = torch.cat([tgt, next_token], dim=1)
        
        return tgt