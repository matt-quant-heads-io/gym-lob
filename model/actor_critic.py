import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class ActorCritic(nn.Module):
    """
    Actor-Critic network with LSTM + Transformer architecture
    """
    
    def __init__(self, 
                 input_size: int = 122,
                 lstm_hidden_size: int = 128, 
                 transformer_heads: int = 4,
                 transformer_layers: int = 2,
                 action_dim: int = 7,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.action_dim = action_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_size, lstm_hidden_size)
        
        # LSTM feature extractor
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size, 
            num_layers=1,
            batch_first=True,
            dropout=dropout if transformer_layers > 1 else 0
        )
        
        # Positional encoding for transformer
        self.pos_encoding = PositionalEncoding(lstm_hidden_size)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden_size,
            nhead=transformer_heads,
            dim_feedforward=lstm_hidden_size * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=transformer_layers
        )
        
        # Output heads
        self.actor_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size // 2, action_dim)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                sequence_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size) or (batch_size, input_size)
            hidden: LSTM hidden state tuple (h, c)
            sequence_length: Optional sequence length for masking
            
        Returns:
            action_logits: (batch_size, action_dim) 
            value: (batch_size, 1)
            new_hidden: Updated LSTM hidden state
        """
        batch_size = x.size(0)
        
        # Handle single step vs sequence input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            single_step = True
        else:
            single_step = False
            
        seq_len = x.size(1)
        
        # Input projection
        x = F.relu(self.input_proj(x))  # (batch_size, seq_len, lstm_hidden_size)
        
        # LSTM processing
        lstm_out, new_hidden = self.lstm(x, hidden)  # (batch_size, seq_len, lstm_hidden_size)
        
        # Transformer processing
        # Add positional encoding
        lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch_size, lstm_hidden_size)
        lstm_out = self.pos_encoding(lstm_out)
        lstm_out = lstm_out.transpose(0, 1)  # (batch_size, seq_len, lstm_hidden_size)
        
        # Apply transformer
        transformer_out = self.transformer(lstm_out)  # (batch_size, seq_len, lstm_hidden_size)
        
        # Use last timestep for predictions
        if single_step:
            features = transformer_out[:, -1, :]  # (batch_size, lstm_hidden_size)
        else:
            features = transformer_out[:, -1, :]  # Use last timestep
        
        # Generate outputs
        action_logits = self.actor_head(features)  # (batch_size, action_dim)
        value = self.critic_head(features)  # (batch_size, 1)
        
        return action_logits, value, new_hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state"""
        h = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        return (h, c)
    
    def get_action_and_value(self, x: torch.Tensor, 
                           hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                           action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action and value for PPO training
        
        Returns:
            action: Selected action
            log_prob: Log probability of action  
            value: State value
            entropy: Action entropy
        """
        action_logits, value, new_hidden = self.forward(x, hidden)
        
        # Create action distribution
        dist = torch.distributions.Categorical(logits=action_logits)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, value, entropy