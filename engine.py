import time
from datetime import datetime
import pandas as pd
from config import HP
from config import Colors
import torch
import torch.nn as nn
from train_utils import evaluate_loss, get_batches

def train(model, optimizer, scheduler, dataset_instance=HP['encoded_text'], print_logs=True):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    start_time = time.time()
    print(Colors.BOLD + "Training function started at:", datetime.now())
    print(Colors.ENDC)
    
    for epoch in range(HP['epochs']):
        model.train()  # Ensure model is in training mode
        optimizer.zero_grad()
        
        xs, ys = get_batches(dataset_instance, 'train', HP['batch_size'], HP['context_window'])
        
        # Start timer for forward and backward pass
        forward_start = time.time()
        logits = model(xs)  # Generate predictions
        
        # Reshape logits to [batch_size * seq_length, vocab_size]
        # and targets to [batch_size * seq_length]
        logits = logits.view(-1, logits.size(-1))  # Ensure logits are (N, C) where C is num_classes
        ys = ys.view(-1)  # Flatten targets to (N,)
        
        loss = criterion(logits, ys)  # Compute loss using model's output and targets
        loss.backward()
        optimizer.step()
        forward_end = time.time()

        # Evaluate loss here to use for scheduler
        if epoch % HP['log_interval'] == 0 or epoch == HP['epochs'] - 1:
            criterion = nn.CrossEntropyLoss()
            losses_dict = evaluate_loss(model, HP['encoded_text'], HP)
            # Ensure you're extracting just the validation loss float value
            val_loss = losses_dict['val']
            # Pass the validation loss to the scheduler
            scheduler.step(val_loss)

            if scheduler:
                scheduler.step(val_loss)  # Step scheduler with validation loss

            model.train()  # Switch back to training mode
            # Log training and validation progress
            if print_logs:
                batch_time = time.time() - start_time
                losses.append({'train': loss.item(), 'val': val_loss})
                print(Colors.OKBLUE + f"Epoch {epoch} | val loss {val_loss:.3f} | "
                      f"Time {batch_time:.3f} | "
                      f"Forward Time {forward_end - forward_start:.3f} | "
                      f"ETA in seconds {batch_time * (HP['epochs'] - epoch) / HP['log_interval']:.3f}" + Colors.ENDC)
            start_time = time.time()
    model_path = "small_rnn_model_checkpoint.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss.item(),
    }, model_path)
    print(f"{Colors.BOLD}{Colors.OKGREEN}Checkpoint saved at {model_path}{Colors.ENDC}")

    print(Colors.BOLD)
    print("Training function ended at:", datetime.now())
    print("validation loss: ", losses[-1]['val'])
    print(Colors.ENDC)
    return pd.DataFrame(losses).plot()
