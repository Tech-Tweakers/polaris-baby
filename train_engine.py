import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from config import HP, Colors
import pandas as pd
import os

def train(model, dataloader, optimizer, scheduler, epochs, log_interval, start_epoch=0):
    checkpoint_path = "model_checkpoint.pth"

    if os.path.isfile(checkpoint_path):
        print(f"{Colors.OKBLUE}Loading checkpoint '{checkpoint_path}'{Colors.ENDC}")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']  # Assuming 'epoch' is saved in the checkpoint.
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"{Colors.OKGREEN}Checkpoint loaded. Resuming training from epoch {start_epoch}{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}No checkpoint found at '{checkpoint_path}'. Starting training from scratch.{Colors.ENDC}")

    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.transpose(1, 2), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
            if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
            
            # Only print the log at intervals specified by log_interval.
            if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
                print(f"{Colors.CYAN}Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {loss.item():.4f}{Colors.ENDC}", end='\r', flush=True)

        # Print a new line to ensure checkpoint message is on a new line.
        print()

        # Checkpoint saving message.
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)
        print(f"{Colors.BOLD}{Colors.OKGREEN}Checkpoint saved at {checkpoint_path}{Colors.ENDC}")

        total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"{Colors.OKGREEN}Epoch [{epoch+1}/{epochs}] completed. Avg Loss: {avg_loss:.4f}{Colors.ENDC}")
        
        losses.append({'epoch': epoch+1, 'train_loss': avg_loss})

    return pd.DataFrame(losses)