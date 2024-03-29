import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from dataset import TextDataset
from model import EnhancedRNNModel
from config import HP, CC, Colors
from train_engine import train
from datetime import datetime
import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

print("")
print(f"{Colors.BLUE_BACKGROUND}{Colors.BOLD} - Tech Tweakers - Polaris Baby v0.1.0 - {Colors.ENDC}")

text_dataset = TextDataset(context_window=HP['context_window'])
dataloader = DataLoader(text_dataset, batch_size=HP['batch_size'], shuffle=True, num_workers=8)

model = EnhancedRNNModel(
    vocab_size=CC['vocab_size'],
    embed_dim=HP['embed_dim'],
    hidden_dim=HP['hidden_dim'],
    dropout=HP['dropout'],
    num_layers=HP['num_layers'],
    num_heads=HP['num_heads'],
    pretrained_embeddings=CC['weights_matrix']
)
optimizer = Adam(model.parameters(), lr=HP['learning_rate'])
scheduler = OneCycleLR(optimizer, max_lr=HP['learning_rate'], total_steps=len(dataloader) * HP['epochs'])

training_start_time = datetime.now()

print(f"{Colors.CYAN}{Colors.BOLD}Starting Training Session at: {training_start_time} {Colors.ENDC}")
train_results = train(model, dataloader, optimizer, scheduler, HP['epochs'], HP['log_interval'], 1)

training_finish_time = datetime.now()
print(f"{Colors.BLUE_BACKGROUND}{Colors.BOLD}Finished Training Session at: {training_finish_time} {Colors.ENDC}")

final_model_path = "small_rnn_model_final.pth"
torch.save(model.state_dict(), final_model_path)
print(f"{Colors.BLUE_BACKGROUND}{Colors.BOLD}Final model saved successfully at {final_model_path} {Colors.ENDC}")
