import torch
from torch.utils.data import Dataset
from config import Colors
from config import CC, HP
import numpy as np
from gensim.models import Word2Vec

class TextDataset(Dataset):
    def __init__(self, context_window=10):
        with open('input.txt', 'r') as file:
            text = file.read()
        print(f"{Colors.OKGREEN}Input text loaded. Length: {len(text)} characters.{Colors.ENDC}")

        self.vocab = sorted(list(set(text)))
        print(f"{Colors.OKGREEN}Vocabulary constructed. Size: {len(self.vocab)}. Vocabulary: {self.vocab[:10]}...{Colors.ENDC}")  # Show a sample of the vocabulary

        # Load the trained Word2Vec model
        print(f"{Colors.CYAN}Loading Word2Vec model...{Colors.ENDC}")
        word2vec_model = Word2Vec.load("word2vec.model")
        print(f"{Colors.CYAN}Word2Vec model loaded. Vocabulary size: {len(word2vec_model.wv)}{Colors.ENDC}")

        # Preparing the weights matrix
        vocab_size = len(word2vec_model.wv)
        embed_dim = word2vec_model.vector_size
        weights_matrix = np.zeros((vocab_size, embed_dim))
        print(f"{Colors.CYAN}Preparing weights matrix. Embedding dimension: {HP['embed_dim']}{Colors.ENDC}")

        for i, word in enumerate(word2vec_model.wv.index_to_key):
            weights_matrix[i] = word2vec_model.wv[word]

        # Converting to a tensor for PyTorch
        weights_matrix = torch.tensor(weights_matrix, dtype=torch.float)
        print(f"{Colors.OKGREEN}Weights matrix converted to tensor. Shape: {weights_matrix.shape}{Colors.ENDC}")

        CC.update({"weights_matrix": weights_matrix, "vocab_size": vocab_size, "vocab": self.vocab})

        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

        self.text = text
        self.encoded_text = [self.stoi[ch] for ch in text if ch in self.stoi]
        self.seq_length = context_window

        print(f"{Colors.OKGREEN}TextDataset initialized. Sequence length: {context_window}, Encoded text length: {len(self.encoded_text)}{Colors.ENDC}")

    def __len__(self):
        return max(0, len(self.encoded_text) - self.seq_length)

    def __getitem__(self, index):
        return (
            torch.tensor(self.encoded_text[index:index+self.seq_length], dtype=torch.long),
            torch.tensor(self.encoded_text[index+1:index+self.seq_length+1], dtype=torch.long)
        )


def load_embeddings(word2vec_model, vocab):
    print(f"{Colors.CYAN}Loading Word2Vec model for embeddings...{Colors.ENDC}")
    word2vec_model = Word2Vec.load("word2vec.model")  # Consider removing this if word2vec_model is already loaded and passed as an argument
    embedding_dim = word2vec_model.vector_size
    print(f"{Colors.CYAN}Embedding dimension: {embedding_dim}{Colors.ENDC}")

    weights_matrix = np.zeros((len(vocab) + 1, embedding_dim))  # +1 for padding token
    print(f"{Colors.CYAN}Preparing weights matrix for vocabulary. Size: {len(vocab) + 1}{Colors.ENDC}")

    print(f"{Colors.OKGREEN}Initializing Embeddings...{Colors.ENDC}")
    for i, word in enumerate(vocab):
        try:
            weights_matrix[i + 1] = word2vec_model.wv[word]  # i + 1 to account for padding token
            
        except KeyError:
            weights_matrix[i + 1] = np.random.normal(size=(embedding_dim,))

    return torch.tensor(weights_matrix, dtype=torch.float)
