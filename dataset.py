import torch
from torch.utils.data import Dataset
from config import Colors
from config import HP

class TextDataset(Dataset):
    def __init__(self, context_window=10):
        with open('input.txt', 'r') as file:
            text = file.read()
        print(f"{Colors.OKBLUE}Input text loaded. Length: {len(text)} characters.{Colors.ENDC}")

        self.vocab = sorted(list(set(text)))
        print(f"{Colors.OKBLUE}Vocabulary constructed. Size: {len(self.vocab)}{Colors.ENDC}")

        HP.update({"vocab_size": len(self.vocab)})
        HP.update({"vocab": self.vocab})

        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

        self.text = text
        self.encoded_text = [self.stoi[ch] for ch in text if ch in self.stoi]
        self.seq_length = context_window

        print(f"{Colors.OKBLUE}TextDataset initialized. Sequence length: {context_window}, Encoded text length: {len(self.encoded_text)}{Colors.ENDC}")

    def __len__(self):
        return max(0, len(self.encoded_text) - self.seq_length)

    def __getitem__(self, index):
        return (
            torch.tensor(self.encoded_text[index:index+self.seq_length], dtype=torch.long),
            torch.tensor(self.encoded_text[index+1:index+self.seq_length+1], dtype=torch.long)
        )