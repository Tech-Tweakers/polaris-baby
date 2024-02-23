from torch import nn
from config import CC

class SmallRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout, num_layers, pretrained_embeddings=CC['weights_matrix']):
        super(SmallRNNModel, self).__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)  # freeze=False allows the embeddings to be fine-tuned
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
