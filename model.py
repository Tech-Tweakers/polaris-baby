import torch
from torch import nn
from config import CC, HP

class SwiGLU(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x): 
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.attn_heads = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.head_dim) for _ in range(num_heads)
        ])
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, encoder_outputs):
        batch_size, seq_len, _ = encoder_outputs.shape
        head_outputs = []

        for attn in self.attn_heads:
            head_outputs.append(attn(encoder_outputs).view(batch_size, seq_len, self.head_dim))

        concat = torch.cat(head_outputs, dim=2)
        
        combined = self.linear(concat)
        
        return combined

class EnhancedRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout, num_layers, num_heads, pretrained_embeddings=CC['weights_matrix']):
        super(EnhancedRNNModel, self).__init__()
        
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
        
        # SwiGLU layer after LSTM
        self.swiglu = SwiGLU(hidden_dim)

        # Multi-head attention
        self.multihead_attn = MultiHeadAttention(hidden_dim, num_heads)

        # Layer normalization for LSTM output
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output fully connected layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)

        swiglu_out = self.swiglu(rnn_out)

        # Apply multi-head attention
        attention_out = self.multihead_attn(swiglu_out)  # This now has the correct shape

        # Apply layer normalization per time step
        normalized = self.layer_norm(attention_out)

        out = self.fc(normalized)
        return out
