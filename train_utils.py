import numpy as np
import torch
from torch import nn
from config import HP, CC

def get_batches(encoded_text, split, batch_size, context_window, config=HP):
    data_length = len(encoded_text)
    train_end = int(.8 * data_length)
    val_end = int(.9 * data_length)

    if split == 'train':
        batch_data = encoded_text[:train_end]
    elif split == 'val':
        batch_data = encoded_text[train_end:val_end]
    else:  # split == 'test'
        batch_data = encoded_text[val_end:]
    
    ix = np.random.randint(0, len(batch_data) - context_window - 1, batch_size)
    x = torch.stack([torch.tensor(batch_data[i:i+context_window]) for i in ix])
    y = torch.stack([torch.tensor(batch_data[i+1:i+context_window+1]) for i in ix])
    return x.long(), y.long()

@torch.no_grad()
def evaluate_loss(model, criterion, dataset_instance, config=HP):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(10):
            xb, yb = get_batches(CC['encoded_text'], split, config['batch_size'], config['context_window'])
            outputs = model(xb)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs.transpose(1, 2), yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out
