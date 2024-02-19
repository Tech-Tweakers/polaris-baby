import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from config import HP
from config import Colors
from dataset import TextDataset
from train_utils import evaluate_loss
from model import SmallRNNModel
from engine import train
import os

print(f"{Colors.HEADER}Polaris Baby v0.1.0 - Training Start{Colors.ENDC}")
    
print(f"{Colors.OKBLUE}{HP}{Colors.ENDC}")

text_dataset = TextDataset(context_window=HP['context_window'])

HP.update({"encoded_text": text_dataset.encoded_text})

dataloader = DataLoader(text_dataset, batch_size=HP['batch_size'], shuffle=True, num_workers=8)

model = SmallRNNModel(HP['vocab_size'], HP['embed_dim'], HP['hidden_dim'])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=HP['learning_rate'])

print(f"{Colors.OKBLUE}Dataloader prepared. Batch size: {HP['batch_size']}{Colors.ENDC}")

# scheduler = OneCycleLR(optimizer, max_lr=HP['learning_rate'], total_steps=total_steps)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=1, verbose=True)

# Checkpoint path
model_path = "small_rnn_model_checkpoint.pth"

# Check if checkpoint exists
if os.path.isfile(model_path):
    print(f"{Colors.OKGREEN}Checkpoint found, loading...{Colors.ENDC}")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']  # Next epoch
    loss = checkpoint['loss']
else:
    start_epoch = 0  # Start from the beginning

total_steps = len(dataloader) * (HP['epochs'] - start_epoch)

print(f"{Colors.OKBLUE}Scheduler configured. Total steps: {total_steps}, Max LR: {HP['learning_rate']}{Colors.ENDC}")

dataset_instance = HP['encoded_text']
train(model, optimizer, scheduler, HP['encoded_text'])

print(f"{Colors.OKGREEN}Training Completed{Colors.ENDC}")

model_path_final = "small_rnn_model.pth"
torch.save(model.state_dict(), model_path_final)
print(f"{Colors.BOLD}{Colors.OKGREEN}Model saved successfully at {model_path_final}{Colors.ENDC}")
