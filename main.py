import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from dataset import TextDataset
from model import SmallRNNModel
from config import HP, CC, Colors
from train_engine import train
from datetime import datetime

def colorful_print(message, color, end=Colors.ENDC):
    print(f"{color}{message}{end}")

print("")
colorful_print("Polaris Baby v0.1.0", Colors.HEADER + Colors.BOLD)
colorful_print("-------------------", Colors.HEADER + Colors.BOLD)

text_dataset = TextDataset(context_window=HP['context_window'])
dataloader = DataLoader(text_dataset, batch_size=HP['batch_size'], shuffle=True, num_workers=8)

model = SmallRNNModel(CC['vocab_size'], HP['embed_dim'], HP['hidden_dim'])
optimizer = Adam(model.parameters(), lr=HP['learning_rate'])
scheduler = OneCycleLR(optimizer, max_lr=HP['learning_rate'], total_steps=len(dataloader) * HP['epochs'])

training_start_time = datetime.now()

print(f"{Colors.BOLD}Starting Training Session at: {training_start_time} {Colors.ENDC}")
train_results = train(model, dataloader, optimizer, scheduler, HP['epochs'], HP['log_interval'], 1)

training_end_time = datetime.now()
print(f"{Colors.BOLD}Finished Training Session at: {training_end_time}{Colors.ENDC}")

final_model_path = "small_rnn_model_final.pth"
torch.save(model.state_dict(), final_model_path)
colorful_print("Final model saved successfully at " + final_model_path, Colors.OKGREEN + Colors.BOLD)
