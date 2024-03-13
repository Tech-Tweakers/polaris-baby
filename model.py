import torch
from torch import nn
from config import CC

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim, 1)

    def forward(self, encoder_outputs):
        # Attention mechanism to generate attention weights
        attn_energies = self.attn(encoder_outputs).squeeze(2)    # [batch_size, seq_len]
        return torch.softmax(attn_energies, dim=1).unsqueeze(2)  # [batch_size, seq_len, 1]

class EnhancedRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout, num_layers, pretrained_embeddings=CC['weights_matrix']):
        super(EnhancedRNNModel, self).__init__()
        
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
        
        # Layer normalization for LSTM output
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output fully connected layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)  # [batch_size, seq_len, hidden_dim]
        
        # Apply layer normalization per time step
        normalized = self.layer_norm(rnn_out)  # [batch_size, seq_len, hidden_dim]

        # Fully connected layer for each time step
        out = self.fc(normalized)  # [batch_size, seq_len, vocab_size]
        return out
