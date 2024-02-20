from torch import nn
from config import Colors
from config import HP

class SmallRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.5, num_layers=HP['num_layers']):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        print(f"{Colors.WARNING}SmallRNNModel initialized. Embedding dim: {embed_dim}, Hidden dim: {hidden_dim}, Vocab size: {vocab_size}{Colors.ENDC}")

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
