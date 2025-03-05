import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_interactions import zoom_factory, panhandler
from sklearn.model_selection import train_test_split
import pickle
from ta import trend, momentum, volatility, volume
import math
from scipy.ndimage import gaussian_filter1d
from typing import Callable, Tuple
import shutil
import contextlib
import traceback
import gc, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



def train_and_validate(model, output_size, criterion, optimizer, 
                       X_train, y_train, X_val, y_val, scheduler, 
                       use_scheduler=None, num_epochs=10, batch_size=64, 
                       model_saving_folder=None, model_name=None, stop_signal_file=None):
    print("'train_and_validate' function started. \n")
    # Ensure model saving folder exists (deleting existing first if there is one)
    if model_saving_folder and os.path.exists(model_saving_folder):
        # os.rmdir(model_saving_folder) # Only works on empty folders 
        shutil.rmtree(model_saving_folder) # Safely remove all contents
        if not os.path.exists(model_saving_folder):
            print(f"Existing folder has been removed : {model_saving_folder}\n")
    if model_saving_folder and not os.path.exists(model_saving_folder):
        os.makedirs(model_saving_folder)
        
    if not model_saving_folder:
        model_saving_folder = './saved_models'
    if not model_name:
        model_name = 'model'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Convert data to tensors # Returns a copy, original is safe
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)  # (seqs, seq_len, features)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)    # (seqs, seq_len)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("y_train:")
    print(type(y_train))
    print(y_train.dtype)
    print(y_train.shape)
    print("X_train:")
    print(type(X_train))
    print(X_train.dtype)
    print(X_train.shape)
    print("\ny_val:")
    print(type(y_val))
    print(y_val.dtype)
    print(y_val.shape)
    print("X_val:")
    print(type(X_val))
    print(X_val.dtype)
    print(X_val.shape)

    # Debug prints for TensorDataset and DataLoader
    print("\nDataset Lengths:")
    print(f"Train Dataset Length: {len(train_dataset)}")
    print(f"Validation Dataset Length: {len(val_dataset)}")

    print("\nDataLoader Batch Sizes:")
    print(f"Number of Batches in Train DataLoader: {len(train_loader)}")
    print(f"Number of Batches in Validation DataLoader: {len(val_loader)}")

    # Additional details for y_train, y_val, and y_test
    print("\ny_train Unique Values and Stats:")
    print(f"Unique values in y_train: {y_train.unique()}")
    print(f"y_train Min: {y_train.min()}, Max: {y_train.max()}")

    print("\ny_val Unique Values and Stats:")
    print(f"Unique values in y_val: {y_val.unique()}")
    print(f"y_val Min: {y_val.min()}, Max: {y_val.max()}")

    # Device check
    print("\nDevice Info:")
    print(f"X_train Device: {X_train.device}")
    print(f"y_train Device: {y_train.device}")
    print(f"X_val Device: {X_val.device}")
    print(f"y_val Device: {y_val.device}\n")

    # Calculate number of batches
    # num_batches = (len(X_train) + batch_size - 1) // batch_size

    global best_results  # Ensure we can modify the external variable if defined outside.
    best_results = []    # Start empty each training run
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        if stop_signal_file and os.path.exists(stop_signal_file):
            print("\nStop signal detected. Exiting training loop safely.\n")
            break
        model.train()
        i=0
        for X_batch, y_batch in train_loader:
            # Reset gradients before forward pass
            optimizer.zero_grad()  # Best practice

            # Forward pass
            outputs = model(X_batch)
            outputs = outputs.view(-1, output_size)
            y_batch = y_batch.view(-1)

            if epoch == 1 and i < 3:
                i += 1
                print(f"\nUnique target values: {y_batch.unique()}")
                print(f"Target dtype: {y_batch.dtype}")
                print(f"Min target: {y_batch.min()}, Max target: {y_batch.max()}")
                print("Unique classes in y_train:", y_train.unique())
                print(f"Unique classes in y_val: {y_val.unique()}\n")
            
            # Compute loss
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            # No longer reset gradients here: optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0)  # Scale back to total loss
            
        train_loss = epoch_loss / len(train_loader.dataset)

        # Perform validation at the end of each epoch
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                val_outputs = model(X_val_batch).view(-1, output_size)
                val_labels = y_val_batch.view(-1)
                val_loss += criterion(val_outputs, val_labels).item() * X_val_batch.size(0)  # Scale to total loss
                val_predictions = torch.argmax(val_outputs, dim=-1)
                val_correct += (val_predictions == val_labels).sum().item()
                val_total += val_labels.size(0)
        val_loss /= len(val_loader.dataset)
        val_accuracy = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.9f}, "
              f"Val Loss: {val_loss:.9f}, "
              f"Val Accuracy: {val_accuracy * 100:.2f}%, "
              f"LR: {optimizer.param_groups[0]['lr']:.9f}")

        # Save current model and update best results if applicable
        current_epoch_info = {
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr'], # Optimizer state
            "model_path": os.path.join(model_saving_folder, f"{model_name}_epoch_{epoch+1}.pth")
        }

        # Insert this epoch if we have fewer than 5 results
        # or if it beats the lowest of the top 5
        if len(best_results) < 5 or val_accuracy > best_results[-1]["val_accuracy"]:
            if len(best_results) == 5:
                # Remove the worst model from the list, the last (lowest accuracy)
                worst = best_results.pop() 
                if os.path.exists(worst["model_path"]):
                    os.remove(worst["model_path"])
                    print(f"Removed old model with accuracy {worst['val_accuracy']*100:.2f}%, and file was at {worst['model_path']}")
            # Just insert and sort by val_accuracy descending
            best_results.append(current_epoch_info) 
            best_results.sort(key=lambda x: x["val_accuracy"], reverse=True)
            torch.save({ # Save this model
                'epoch': epoch+1,  # Save the current epoch
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_state_dict': model.state_dict(),  # Model weights
                'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
                'learning_rate': optimizer.param_groups[0]['lr'] # Optimizer state
            }, current_epoch_info["model_path"])
            print(f"Model saved after epoch {epoch+1} to {current_epoch_info['model_path']} \n")

        if use_scheduler == True:
            # Scheduler step should follow after considering the results (placed after otallher losses)
            scheduler.step(val_loss)


    # Save the final model
    if current_epoch_info:
        final_model_path = os.path.join(model_saving_folder, f"{model_name}_final.pth")
        torch.save({ # Save this model
            'epoch': epoch+1,  # Save the current epoch
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_state_dict': model.state_dict(),  # Model weights
            'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
            'learning_rate': optimizer.param_groups[0]['lr'] # Optimizer state
        }, final_model_path)
        print(f"\nFinal model saved to {final_model_path}")

    print("\nTraining complete. \n\nTop 5 Models Sorted by Validation Accuracy: ")
    for res in best_results:        
        print(f"Epoch {res['epoch']}/{num_epochs}, "
              f"Train Loss: {res['train_loss']:.9f}, " 
              f"Val Loss: {res['val_loss']:.9f}, "
              f"Val Accuracy: {res['val_accuracy'] * 100:.2f}%, "
              f"Model Path: {res['model_path']}")
    print('\n')
    
    del X_train, y_train, X_val, y_val, train_dataset, val_dataset, train_loader, val_loader
    torch.cuda.empty_cache()


def train_and_validate_lora(student_model, teacher_model, ce_criterion, optimizer, 
                            X_train, y_train, X_val, y_val, num_epochs=50, batch_size=64, alpha=0.5):
    """
    student_model: The new LoRA-based student model (with output size 3).
    teacher_model: Frozen teacher model from period 1 (with output size 2).
    ce_criterion: CrossEntropyLoss function.
    optimizer: Optimizer for student model.
    X_train, y_train, X_val, y_val: Training/validation data (as NumPy arrays or similar).
    num_epochs: Number of epochs to train.
    batch_size: Batch size for DataLoader.
    alpha: Weighting factor for distillation loss (alpha * distill_loss + (1-alpha) * ce_loss).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_model.to(device)
    teacher_model.to(device)
    
    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    
    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    teacher_model.eval()  # Ensure teacher is frozen
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            # Forward pass: student model produces logits for 3 classes.
            student_logits = student_model(X_batch)  # Shape: [batch, seq_len, 3]
            
            # Reshape for CE loss computation.
            student_logits_flat = student_logits.view(-1, student_logits.shape[-1])
            y_batch_flat = y_batch.view(-1)
            ce_loss = ce_criterion(student_logits_flat, y_batch_flat)
            
            # Knowledge Distillation:
            # Forward pass through teacher (pre-trained on period 1 data).
            with torch.no_grad():
                teacher_logits = teacher_model(X_batch)  # Shape: [batch, seq_len, 2]
            # We distill only the stable class (class 1).
            # Teacher's class index 1 corresponds to student's class index 1.
            teacher_stable = teacher_logits[:, :, 1]
            student_stable = student_logits[:, :, 1]
            distill_loss = F.mse_loss(student_stable, teacher_stable)
            
            # Total loss: balance between cross-entropy and distillation loss.
            total_loss = alpha * distill_loss + (1 - alpha) * ce_loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * X_batch.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation evaluation (only CE loss and accuracy)
        student_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                outputs = student_model(X_val_batch)
                outputs_flat = outputs.view(-1, outputs.shape[-1])
                y_val_flat = y_val_batch.view(-1)
                loss_val = ce_criterion(outputs_flat, y_val_flat)
                val_loss += loss_val.item() * X_val_batch.size(0)
                preds = outputs_flat.argmax(dim=1)
                correct += (preds == y_val_flat).sum().item()
                total += y_val_flat.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
