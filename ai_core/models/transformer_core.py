"""
Transformer Core - Advanced Attention Mechanisms
Implements state-of-the-art transformer architecture with multi-head attention,
positional encoding, and advanced normalization techniques.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json


class AttentionMechanism:
    """
    Multi-head attention mechanism with various attention variants:
    - Scaled Dot-Product Attention
    - Multi-Head Self-Attention
    - Cross-Attention
    - Sparse Attention
    - Linear Attention
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        attention_type: str = "scaled_dot_product",
        causal_mask: bool = False
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_rate = dropout_rate
        self.attention_type = attention_type
        self.causal_mask = causal_mask
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Initialize projection matrices
        scale = np.sqrt(2.0 / (embed_dim + self.head_dim))
        self.W_query = np.random.randn(embed_dim, embed_dim) * scale
        self.W_key = np.random.randn(embed_dim, embed_dim) * scale
        self.W_value = np.random.randn(embed_dim, embed_dim) * scale
        self.W_output = np.random.randn(embed_dim, embed_dim) * scale
        
        # Attention weights cache
        self.attention_weights = None
        
    def _split_heads(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        """Split embedding dimension into multiple heads."""
        return x.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
    
    def _merge_heads(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        """Merge multiple heads back into single embedding."""
        return x.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.embed_dim)
    
    def _create_causal_mask(self, seq_length: int) -> np.ndarray:
        """Create causal mask for autoregressive attention."""
        mask = np.triu(np.ones((seq_length, seq_length)), k=1) * -1e9
        return mask.reshape(1, 1, seq_length, seq_length)
    
    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through attention mechanism.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim)
            key: Key tensor of shape (batch_size, seq_len, embed_dim)
            value: Value tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        
        # Linear projections
        Q = np.dot(query, self.W_query)
        K = np.dot(key, self.W_key)
        V = np.dot(value, self.W_value)
        
        # Split into heads
        Q = self._split_heads(Q, batch_size)
        K = self._split_heads(K, batch_size)
        V = self._split_heads(V, batch_size)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        # Apply masks
        if self.causal_mask:
            causal = self._create_causal_mask(seq_len_q)
            scores = scores + causal
        
        if mask is not None:
            scores = scores + mask * -1e9
        
        # Softmax attention weights
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)
        
        # Apply dropout during training
        if training and self.dropout_rate > 0:
            dropout_mask = (np.random.rand(*attention_weights.shape) > self.dropout_rate).astype(float)
            attention_weights = attention_weights * dropout_mask / (1 - self.dropout_rate)
        
        # Apply attention to values
        context = np.matmul(attention_weights, V)
        
        # Merge heads
        context = self._merge_heads(context, batch_size)
        
        # Output projection
        output = np.dot(context, self.W_output)
        
        self.attention_weights = attention_weights
        
        return output, attention_weights
    
    def compute_attention_entropy(self) -> float:
        """Compute entropy of attention weights for analysis."""
        if self.attention_weights is None:
            return 0.0
        
        # Avoid log(0)
        weights_clipped = np.clip(self.attention_weights, 1e-10, 1.0)
        entropy = -np.mean(np.sum(weights_clipped * np.log(weights_clipped), axis=-1))
        return float(entropy)


class TransformerCore:
    """
    Complete transformer architecture with:
    - Multi-layer encoder-decoder structure
    - Position-wise feed-forward networks
    - Layer normalization
    - Residual connections
    - Positional encodings
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 2048,
        dropout_rate: float = 0.1,
        max_seq_length: int = 512
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.max_seq_length = max_seq_length
        
        # Embedding layers
        scale = np.sqrt(2.0 / (vocab_size + embed_dim))
        self.token_embedding = np.random.randn(vocab_size, embed_dim) * scale
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(max_seq_length, embed_dim)
        
        # Encoder layers
        self.encoder_attention = []
        self.encoder_ffn = []
        self.encoder_norm1 = []
        self.encoder_norm2 = []
        
        for _ in range(num_layers):
            self.encoder_attention.append(
                AttentionMechanism(embed_dim, num_heads, dropout_rate)
            )
            
            # Feed-forward network weights
            scale_ff = np.sqrt(2.0 / (embed_dim + ff_dim))
            self.encoder_ffn.append({
                'W1': np.random.randn(embed_dim, ff_dim) * scale_ff,
                'b1': np.zeros(ff_dim),
                'W2': np.random.randn(ff_dim, embed_dim) * scale_ff,
                'b2': np.zeros(embed_dim)
            })
            
            # Layer normalization parameters
            self.encoder_norm1.append({
                'gamma': np.ones(embed_dim),
                'beta': np.zeros(embed_dim)
            })
            self.encoder_norm2.append({
                'gamma': np.ones(embed_dim),
                'beta': np.zeros(embed_dim)
            })
        
        # Decoder layers (for sequence-to-sequence tasks)
        self.decoder_attention = []
        self.decoder_cross_attention = []
        self.decoder_ffn = []
        self.decoder_norm1 = []
        self.decoder_norm2 = []
        self.decoder_norm3 = []
        
        for _ in range(num_layers):
            self.decoder_attention.append(
                AttentionMechanism(embed_dim, num_heads, dropout_rate, causal_mask=True)
            )
            self.decoder_cross_attention.append(
                AttentionMechanism(embed_dim, num_heads, dropout_rate)
            )
            
            scale_ff = np.sqrt(2.0 / (embed_dim + ff_dim))
            self.decoder_ffn.append({
                'W1': np.random.randn(embed_dim, ff_dim) * scale_ff,
                'b1': np.zeros(ff_dim),
                'W2': np.random.randn(ff_dim, embed_dim) * scale_ff,
                'b2': np.zeros(embed_dim)
            })
            
            self.decoder_norm1.append({'gamma': np.ones(embed_dim), 'beta': np.zeros(embed_dim)})
            self.decoder_norm2.append({'gamma': np.ones(embed_dim), 'beta': np.zeros(embed_dim)})
            self.decoder_norm3.append({'gamma': np.ones(embed_dim), 'beta': np.zeros(embed_dim)})
        
        # Output layer
        scale_out = np.sqrt(2.0 / (embed_dim + vocab_size))
        self.output_projection = np.random.randn(embed_dim, vocab_size) * scale_out
    
    def _create_positional_encoding(self, max_len: int, embed_dim: int) -> np.ndarray:
        """Create sinusoidal positional encodings."""
        position = np.arange(max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        
        pe = np.zeros((max_len, embed_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe.reshape(1, max_len, embed_dim)
    
    def _layer_norm(self, x: np.ndarray, norm_params: Dict, epsilon: float = 1e-6) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True) + epsilon
        
        normalized = (x - mean) / std
        return norm_params['gamma'] * normalized + norm_params['beta']
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation function."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def encode(self, input_ids: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Encode input sequence through transformer encoder.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_length)
            training: Whether in training mode
            
        Returns:
            Encoded representations of shape (batch_size, seq_length, embed_dim)
        """
        batch_size, seq_length = input_ids.shape
        
        # Token embedding
        embedded = self.token_embedding[input_ids] * np.sqrt(self.embed_dim)
        
        # Add positional encoding
        embedded = embedded + self.positional_encoding[:, :seq_length, :]
        
        # Apply embedding dropout
        if training and self.dropout_rate > 0:
            dropout_mask = (np.random.rand(*embedded.shape) > self.dropout_rate).astype(float)
            embedded = embedded * dropout_mask / (1 - self.dropout_rate)
        
        # Pass through encoder layers
        hidden = embedded
        for i in range(self.num_layers):
            # Self-attention with residual connection
            attention_out, _ = self.encoder_attention[i].forward(hidden, hidden, hidden, training=training)
            hidden = hidden + attention_out
            hidden = self._layer_norm(hidden, self.encoder_norm1[i])
            
            # Feed-forward network with residual connection
            ffn_input = hidden
            ffn_hidden = self._gelu(np.dot(hidden, self.encoder_ffn[i]['W1']) + self.encoder_ffn[i]['b1'])
            ffn_output = np.dot(ffn_hidden, self.encoder_ffn[i]['W2']) + self.encoder_ffn[i]['b2']
            hidden = ffn_input + ffn_output
            hidden = self._layer_norm(hidden, self.encoder_norm2[i])
        
        return hidden
    
    def decode(
        self,
        decoder_input: np.ndarray,
        encoder_output: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Decode through transformer decoder with cross-attention to encoder output.
        
        Args:
            decoder_input: Decoder input token IDs
            encoder_output: Output from encoder
            training: Whether in training mode
            
        Returns:
            Decoded representations
        """
        batch_size, seq_length = decoder_input.shape
        
        # Token embedding
        embedded = self.token_embedding[decoder_input] * np.sqrt(self.embed_dim)
        embedded = embedded + self.positional_encoding[:, :seq_length, :]
        
        if training and self.dropout_rate > 0:
            dropout_mask = (np.random.rand(*embedded.shape) > self.dropout_rate).astype(float)
            embedded = embedded * dropout_mask / (1 - self.dropout_rate)
        
        # Pass through decoder layers
        hidden = embedded
        for i in range(self.num_layers):
            # Self-attention (causal)
            self_attn_out, _ = self.decoder_attention[i].forward(hidden, hidden, hidden, training=training)
            hidden = hidden + self_attn_out
            hidden = self._layer_norm(hidden, self.decoder_norm1[i])
            
            # Cross-attention to encoder output
            cross_attn_out, _ = self.decoder_cross_attention[i].forward(
                hidden, encoder_output, encoder_output, training=training
            )
            hidden = hidden + cross_attn_out
            hidden = self._layer_norm(hidden, self.decoder_norm2[i])
            
            # Feed-forward network
            ffn_input = hidden
            ffn_hidden = self._gelu(np.dot(hidden, self.decoder_ffn[i]['W1']) + self.decoder_ffn[i]['b1'])
            ffn_output = np.dot(ffn_hidden, self.decoder_ffn[i]['W2']) + self.decoder_ffn[i]['b2']
            hidden = ffn_input + ffn_output
            hidden = self._layer_norm(hidden, self.decoder_norm3[i])
        
        return hidden
    
    def forward(self, input_ids: np.ndarray, target_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Full transformer forward pass (encoder-decoder or encoder-only).
        
        Args:
            input_ids: Input token IDs
            target_ids: Optional target IDs for decoder input
            
        Returns:
            Output logits
        """
        # Encode input
        encoder_output = self.encode(input_ids, training=(target_ids is not None))
        
        if target_ids is not None:
            # Decoder mode
            decoder_output = self.decode(target_ids, encoder_output, training=True)
            logits = np.dot(decoder_output, self.output_projection)
        else:
            # Encoder-only mode (e.g., for classification)
            # Use [CLS] token representation or mean pooling
            cls_representation = encoder_output[:, 0, :]  # First token
            logits = np.dot(cls_representation, self.output_projection)
        
        return logits
    
    def get_attention_maps(self) -> List[np.ndarray]:
        """Return attention weights from all layers for visualization."""
        attention_maps = []
        for attn_layer in self.encoder_attention:
            if attn_layer.attention_weights is not None:
                attention_maps.append(attn_layer.attention_weights.copy())
        return attention_maps
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        total = self.token_embedding.size
        total += self.positional_encoding.size
        
        for i in range(self.num_layers):
            # Encoder attention
            total += 3 * self.encoder_attention[i].W_query.size
            total += self.encoder_attention[i].W_output.size
            
            # Encoder FFN
            total += self.encoder_ffn[i]['W1'].size + self.encoder_ffn[i]['W2'].size
            total += self.encoder_ffn[i]['b1'].size + self.encoder_ffn[i]['b2'].size
            
            # Layer norms
            total += 2 * self.encoder_norm1[i]['gamma'].size
            total += 2 * self.encoder_norm2[i]['gamma'].size
        
        # Output projection
        total += self.output_projection.size
        
        return total
    
    def save_model(self, filepath: str):
        """Save model weights to file."""
        model_data = {
            'config': {
                'vocab_size': self.vocab_size,
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'ff_dim': self.ff_dim,
                'dropout_rate': self.dropout_rate,
                'max_seq_length': self.max_seq_length
            },
            'weights': {
                'token_embedding': self.token_embedding.tolist(),
                'positional_encoding': self.positional_encoding.tolist(),
                'output_projection': self.output_projection.tolist(),
                'encoder_ffn': [{
                    'W1': layer['W1'].tolist(),
                    'b1': layer['b1'].tolist(),
                    'W2': layer['W2'].tolist(),
                    'b2': layer['b2'].tolist()
                } for layer in self.encoder_ffn],
                'encoder_norm1': self.encoder_norm1,
                'encoder_norm2': self.encoder_norm2
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
