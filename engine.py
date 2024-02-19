import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch import nn
from dataset import TextDataset
from model import SmallRNNModel
from config import HP, Colors
import pandas as pd
import os

def train(model, dataloader, optimizer, scheduler, epochs, log_interval, start_epoch=0):
    checkpoint_path = "model_checkpoint.pth"

    if os.path.isfile(checkpoint_path):
        print(f"{Colors.OKBLUE}Loading checkpoint '{checkpoint_path}'{Colors.ENDC}")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"{Colors.OKGREEN}Checkpoint loaded. Resuming training from epoch {start_epoch}{Colors.ENDC}")
    else:
        start_epoch = 0
        print(f"{Colors.WARNING}No checkpoint found at '{checkpoint_path}'. Starting training from scratch.{Colors.ENDC}")

    criterion = nn.CrossEntropyLoss()
    losses = []
    global_batch_count = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            global_batch_count += 1
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.transpose(1, 2), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
            if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
            
            if global_batch_count % log_interval == 0:
                print(f"{Colors.CYAN}Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {loss.item():.4f}{Colors.ENDC}")

            if global_batch_count >= HP['stop_batch']:
                print(f"{Colors.WARNING}Reached designated stopping global batch ({HP['stop_batch']}). Ending training.{Colors.ENDC}")
                checkpoint_path = "model_checkpoint.pth"
                torch.save({
                    'global_batch_count': global_batch_count,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"{Colors.BOLD}{Colors.OKGREEN}Checkpoint saved at {checkpoint_path}{Colors.ENDC}")
                return pd.DataFrame(losses)

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"{Colors.OKGREEN}Epoch [{epoch+1}/{epochs}] completed. Avg Loss: {avg_loss:.4f}{Colors.ENDC}")
        
        losses.append({'epoch': epoch+1, 'train_loss': avg_loss})

    return pd.DataFrame(losses)

if __name__ == "__main__":
    print(f"{Colors.HEADER}Training Start{Colors.ENDC}")

    text_dataset = TextDataset(context_window=HP['context_window'])
    dataloader = DataLoader(text_dataset, batch_size=HP['batch_size'], shuffle=True, num_workers=8)

    model = SmallRNNModel(HP['vocab_size'], HP['embed_dim'], HP['hidden_dim'])
    optimizer = Adam(model.parameters(), lr=HP['learning_rate'])
    scheduler = OneCycleLR(optimizer, max_lr=HP['learning_rate'], total_steps=len(dataloader) * HP['epochs'])

    train_results = train(model, dataloader, optimizer, scheduler, HP['epochs'], HP['log_interval'])

    final_model_path = "small_rnn_model_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"{Colors.BOLD}{Colors.OKGREEN}Final model saved successfully at {final_model_path}{Colors.ENDC}")
