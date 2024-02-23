import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from config import Colors
import pandas as pd
import os

def train(model, dataloader, optimizer, scheduler, epochs, log_interval, start_epoch=0):
    checkpoint_path = "model_checkpoint.pth"

    if os.path.isfile(checkpoint_path):
        print(f"{Colors.WARNING}Loading checkpoint '{checkpoint_path}'{Colors.ENDC}")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"{Colors.WARNING}Checkpoint loaded. Resuming training from epoch {start_epoch}{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}No checkpoint found at '{checkpoint_path}'. Starting training from scratch.{Colors.ENDC}")

    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0

        total_batches = len(dataloader)
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.transpose(1, 2), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)

            if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
            
            if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
                print(f"{Colors.FAIL}{Colors.BOLD}Epoch:{Colors.ENDC} {Colors.CYAN}Cur/Total - {epoch}/{epochs}{Colors.ENDC}, {Colors.FAIL}{Colors.BOLD}Batch:{Colors.ENDC} {Colors.CYAN}Cur/Total - {batch_idx+1}/{total_batches}{Colors.ENDC}, {Colors.FAIL}{Colors.BOLD}Loss:{Colors.ENDC} {Colors.CYAN}Cur/Avg - {loss.item():.4f}/{avg_loss:.4f}{Colors.ENDC}", end='\r', flush=True)
        print()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)
        print(f"{Colors.CYAN}Checkpoint saved at {checkpoint_path}{Colors.ENDC}")

        avg_loss = total_loss / len(dataloader)
        print(f"{Colors.OKGREEN}{Colors.BOLD}Epoch [{epoch}/{epochs}] completed. Avg Loss: {avg_loss:.4f}{Colors.ENDC}")
        
        losses.append({'epoch': epoch, 'train_loss': avg_loss})

    return pd.DataFrame(losses)
