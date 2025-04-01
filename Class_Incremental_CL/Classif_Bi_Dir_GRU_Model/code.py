import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import shutil
import contextlib
import gc # For explicit garbage collection if needed

# Assume process_and_return_splits and ensure_folder are defined elsewhere
# from your_data_processing_module import process_and_return_splits, ensure_folder

# --- Utility Functions ---
def ensure_folder(folder_path):
    """Creates a folder if it doesn't exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")

def compute_classwise_accuracy(logits_flat, y_batch, class_correct, class_total):
    """
    Computes per-class accuracy by accumulating correct and total samples for each class.

    logits_flat: Model predictions (logits) in shape [batch_size * seq_len, output_size]
    y_batch: True labels in shape [batch_size * seq_len]
    class_correct: Dictionary to store correct predictions per class (passed by reference)
    class_total: Dictionary to store total samples per class (passed by reference)
    """
    predictions = torch.argmax(logits_flat, dim=-1)
    for label, pred in zip(y_batch.cpu().numpy(), predictions.cpu().numpy()):
        label_int = int(label) # Ensure keys are integers
        if label_int not in class_total:
            class_total[label_int] = 0
            class_correct[label_int] = 0
        class_total[label_int] += 1
        if label_int == pred:
            class_correct[label_int] += 1

# --- Model Definition ---
class BiGRUWithAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int, dropout: float = 0.0):
        super(BiGRUWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0) # Dropout only between layers
        self.attention_fc = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout_layer = nn.Dropout(dropout) # Apply dropout before the final FC layer
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name: # GRU weights
                nn.init.xavier_uniform_(param)
            elif 'weight' in name: # Other weights (attention, fc)
                 nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)

        attn_weights = torch.tanh(self.attention_fc(out))
        out = attn_weights * out # Element-wise attention application

        # Apply dropout before the final fully connected layer
        out = self.dropout_layer(out)

        # Apply final layer to each time step
        out = self.fc(out)
        return out

# --- EWC Class Definition (Revised) ---
class EWC:
    """
    Elastic Weight Consolidation (EWC).
    Stores Fisher information matrix and optimal parameters from a previous task.
    Provides a penalty term for the loss function.
    """
    def __init__(self, fisher: dict, params: dict):
        """
        Initializes EWC with pre-computed Fisher matrix and optimal parameters.
        Args:
            fisher (dict): Dictionary mapping parameter names to Fisher information tensors.
            params (dict): Dictionary mapping parameter names to optimal parameter tensors from the previous task.
        """
        self.fisher = fisher
        self.params = params

    @staticmethod
    def compute_fisher_and_params(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                                  device: torch.device, sample_size: int = None):
        """
        Computes the diagonal Fisher Information Matrix and copies optimal parameters.
        Should be called *after* training on a task is complete, using the data from that task.

        Args:
            model (nn.Module): The trained model (best model for the task).
            dataloader (DataLoader): DataLoader for the task's training data.
            criterion (nn.Module): The loss function used for training.
            device (torch.device): The device (CPU or CUDA).
            sample_size (int, optional): Max number of samples to use for Fisher computation. Defaults to None (use all).

        Returns:
            tuple(dict, dict): (fisher_dict, params_dict)
        """
        model.eval() # Ensure model is in eval mode for consistency, although grads are needed
        
        # Store optimal parameters
        params_dict = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        
        # Initialize Fisher
        fisher_dict = {n: torch.zeros_like(p) for n, p in params_dict.items()}

        num_samples_processed = 0
        num_batches = 0

        # We need gradients, so temporarily enable grad calculation, but don't take steps
        # It's often recommended to use model.eval() for consistency in dropout/batchnorm
        # but calculate gradients as if in training.
        original_mode = model.training
        model.train() # Need gradients enabled

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            current_batch_size = x.size(0)
            
            model.zero_grad()
            outputs = model(x)
            # Reshape outputs and targets as needed by the criterion
            # Assuming criterion expects [N*SeqLen, Classes] and [N*SeqLen]
            outputs_flat = outputs.view(-1, outputs.size(-1))
            y_flat = y.view(-1)
            
            # Ensure labels are within the expected range for the current model's output size
            valid_indices = (y_flat >= 0) & (y_flat < outputs_flat.size(1))
            if not valid_indices.all():
                print(f"Warning: Labels out of range detected during Fisher computation. Min: {y_flat.min()}, Max: {y_flat.max()}, Output size: {outputs_flat.size(1)}")
                # Filter data or handle appropriately, here we skip invalid labels for loss calculation
                outputs_flat = outputs_flat[valid_indices]
                y_flat = y_flat[valid_indices]
                if y_flat.numel() == 0:
                    continue # Skip batch if no valid labels remain

            loss = criterion(outputs_flat, y_flat)
            loss.backward()

            # Accumulate squared gradients
            for n, p in model.named_parameters():
                if p.grad is not None and n in fisher_dict:
                    # Accumulate fisher grads - note the use of batch size
                    # Averaging over dataset size is common
                    fisher_dict[n] += p.grad.data.clone().pow(2) * current_batch_size 

            num_samples_processed += current_batch_size
            num_batches += 1
            
            if sample_size is not None and num_samples_processed >= sample_size:
                print(f"Stopping Fisher computation early after {num_samples_processed} samples.")
                break
        
        # Normalize Fisher matrix by the total number of samples processed
        if num_samples_processed > 0:
            for n in fisher_dict:
                fisher_dict[n] /= num_samples_processed
        else:
             print("Warning: No samples processed for Fisher computation.")

        # Restore original model mode
        model.train(original_mode) # Set back to whatever it was

        return fisher_dict, params_dict

    def penalty(self, model: nn.Module):
        """
        Calculates the EWC penalty term.
        Only applies penalty to parameters present in the stored fisher/params
        and having the same shape as the current model's parameters.
        """
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher and n in self.params:
                # Check if shapes match before calculating penalty
                if p.shape == self.params[n].shape:
                    loss += (self.fisher[n] * (p - self.params[n].to(p.device))**2).sum()
                # else:
                #     print(f"Skipping EWC penalty for {n} due to shape mismatch: Current={p.shape}, Stored={self.params[n].shape}")
        return loss

# --- Unified Training and Validation Function (Handles EWC) ---
def train_validate_incremental(
    model: nn.Module,
    output_size: int,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    scheduler: optim.lr_scheduler._LRScheduler = None, # Type hint for scheduler
    use_scheduler: bool = False,
    ewc_object: EWC = None, # Pass the EWC object if applicable
    lambda_ewc: float = 0.0, # EWC regularization strength
    num_epochs: int = 10,
    batch_size: int = 64,
    model_saving_folder: str = './saved_models',
    model_name: str = 'model',
    stop_signal_file: str = None,
    device: torch.device = None
    ):
    """
    Trains and validates the model for one period, with optional EWC penalty.

    Returns:
        list: List of dictionaries containing info about the top 5 best models based on validation accuracy.
              Returns empty list if training is interrupted early or fails.
    """
    print(f"--- Starting Training Period ---")
    print(f"Model: {model_name}, Output Size: {output_size}, EWC Active: {ewc_object is not None and lambda_ewc > 0}, Lambda EWC: {lambda_ewc if ewc_object else 'N/A'}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Ensure model saving folder exists
    ensure_folder(model_saving_folder)
    # Optionally clear previous contents (Be careful with this in production)
    # if os.path.exists(model_saving_folder):
    #     print(f"Clearing contents of existing folder: {model_saving_folder}")
    #     shutil.rmtree(model_saving_folder)
    #     os.makedirs(model_saving_folder)

    # Convert data to tensors
    try:
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    except Exception as e:
        print(f"Error converting data to tensors: {e}")
        # Print shapes and types for debugging
        print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
        print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
        print(f"X_val shape: {X_val.shape}, dtype: {X_val.dtype}")
        print(f"y_val shape: {y_val.shape}, dtype: {y_val.dtype}")
        raise # Re-raise the exception

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True if device.type == 'cuda' else False)

    print("\nData Shapes & Unique Labels:")
    print(f"X_train shape: {X_train_t.shape}, y_train shape: {y_train_t.shape}, Unique y_train: {torch.unique(y_train_t)}")
    print(f"X_val shape: {X_val_t.shape},   y_val shape: {y_val_t.shape},   Unique y_val: {torch.unique(y_val_t)}")
    print(f"Train DataLoader: {len(train_loader)} batches, Val DataLoader: {len(val_loader)} batches")
    print(f"Device: {device}\n")

    best_results_list = [] # Local list to store best results for this run

    for epoch in range(num_epochs):
        if stop_signal_file and os.path.exists(stop_signal_file):
            print("\nStop signal detected. Exiting training loop safely.")
            break

        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0.0
        train_class_correct = {}
        train_class_total = {}

        for i, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(X_batch) # Shape: [batch, seq_len, output_size]
            outputs_flat = outputs.view(-1, output_size) # Shape: [batch * seq_len, output_size]
            y_batch_flat = y_batch.view(-1) # Shape: [batch * seq_len]

            # Filter out potential padding or invalid labels if necessary
            # Example: Ignore labels like -1 if used for padding
            # valid_indices = y_batch_flat >= 0
            # outputs_flat = outputs_flat[valid_indices]
            # y_batch_flat = y_batch_flat[valid_indices]
            # if y_batch_flat.numel() == 0: continue # Skip if batch becomes empty

            if y_batch_flat.min() < 0 or y_batch_flat.max() >= output_size:
                 print(f"Warning: Batch labels out of range! Min: {y_batch_flat.min()}, Max: {y_batch_flat.max()}, Output Size: {output_size}")
                 # Option: Skip batch, filter labels, or crash depending on severity
                 continue # Skip this batch

            # Standard Cross-Entropy Loss
            loss = criterion(outputs_flat, y_batch_flat)

            # Add EWC Penalty (if applicable)
            if ewc_object is not None and lambda_ewc > 0:
                ewc_penalty = ewc_object.penalty(model)
                loss += (lambda_ewc / 2.0) * ewc_penalty # Factor of 1/2 is common

            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item() * X_batch.size(0) # Accumulate total loss for the epoch
            compute_classwise_accuracy(outputs_flat, y_batch_flat, train_class_correct, train_class_total)

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_classwise_acc_str = {c: f"{(train_class_correct[c] / train_class_total[c]) * 100:.2f}% ({train_class_correct[c]}/{train_class_total[c]})"
                                   if train_class_total[c] > 0 else "0.00% (0/0)"
                                   for c in sorted(train_class_total.keys())}


        # --- Validation Phase ---
        model.eval()
        epoch_val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_class_correct = {}
        val_class_total = {}

        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                val_outputs = model(X_val_batch) # [batch, seq_len, output_size]
                val_outputs_flat = val_outputs.view(-1, output_size)
                val_labels_flat = y_val_batch.view(-1)

                # Filter labels if needed
                # valid_indices_val = val_labels_flat >= 0
                # val_outputs_flat = val_outputs_flat[valid_indices_val]
                # val_labels_flat = val_labels_flat[valid_indices_val]
                # if val_labels_flat.numel() == 0: continue

                if val_labels_flat.min() < 0 or val_labels_flat.max() >= output_size:
                    print(f"Warning: Validation Batch labels out of range! Min: {val_labels_flat.min()}, Max: {val_labels_flat.max()}, Output Size: {output_size}")
                    continue # Skip this batch

                val_loss_batch = criterion(val_outputs_flat, val_labels_flat)
                epoch_val_loss += val_loss_batch.item() * X_val_batch.size(0)

                val_predictions = torch.argmax(val_outputs_flat, dim=-1)
                val_correct += (val_predictions == val_labels_flat).sum().item()
                val_total += val_labels_flat.size(0)
                compute_classwise_accuracy(val_outputs_flat, val_labels_flat, val_class_correct, val_class_total)

        avg_val_loss = epoch_val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        val_classwise_acc_str = {c: f"{(val_class_correct[c] / val_class_total[c]) * 100:.2f}% ({val_class_correct[c]}/{val_class_total[c]})"
                                 if val_class_total[c] > 0 else "0.00% (0/0)"
                                 for c in sorted(val_class_total.keys())}

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              # f"Train Acc (Classwise): {train_classwise_acc_str} | " # Can be verbose
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Val Acc: {val_accuracy * 100:.2f}% | "
              f"Val Acc (Classwise): {val_classwise_acc_str}")

        # --- Model Saving Logic ---
        current_epoch_info = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_classwise_accuracy_dict": train_class_correct, # Store raw counts for potential later aggregation
            "train_classwise_total_dict": train_class_total,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_classwise_accuracy_dict": val_class_correct,
            "val_classwise_total_dict": val_class_total,
            'learning_rate': current_lr,
            "model_path": os.path.join(model_saving_folder, f"{model_name}_epoch_{epoch+1}.pth")
        }

        # Keep top 5 models based on validation accuracy
        if len(best_results_list) < 5 or val_accuracy > best_results_list[-1]["val_accuracy"]:
            # Remove worst model if list is full
            if len(best_results_list) == 5:
                worst = best_results_list.pop()
                if os.path.exists(worst["model_path"]):
                    try:
                        os.remove(worst["model_path"])
                        # print(f"Removed old model: {os.path.basename(worst['model_path'])} (Val Acc: {worst['val_accuracy']*100:.2f}%)")
                    except OSError as e:
                        print(f"Error removing old model file {worst['model_path']}: {e}")

            # Add current model info and sort
            best_results_list.append(current_epoch_info)
            best_results_list.sort(key=lambda x: x["val_accuracy"], reverse=True)

            # Save the current best model's state
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'output_size': output_size, # Save the output size configuration
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_accuracy': val_accuracy,
                    'val_loss': avg_val_loss,
                }, current_epoch_info["model_path"])
                # print(f"Saved model to {current_epoch_info['model_path']}")
            except Exception as e:
                 print(f"Error saving model checkpoint at epoch {epoch+1}: {e}")


        # --- Scheduler Step ---
        if use_scheduler and scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step() # For schedulers like StepLR

    # --- End of Training ---
    print("\nTraining complete for this period.")
    if not best_results_list:
         print("No models were saved (potentially due to interruption or no improvement).")
         return []

    print("\nTop 5 Models (Sorted by Validation Accuracy):")
    for i, res in enumerate(best_results_list):
        print(f"Rank {i+1}: Epoch {res['epoch']}, Val Acc: {res['val_accuracy'] * 100:.2f}%, Val Loss: {res['val_loss']:.6f}, Path: {res['model_path']}")

    # Save the final model state (last epoch) regardless of performance
    final_model_path = os.path.join(model_saving_folder, f"{model_name}_final_epoch_{epoch+1}.pth")
    try:
        torch.save({
            'epoch': epoch + 1,
            'output_size': output_size,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_accuracy': val_accuracy, # Last epoch's accuracy
            'val_loss': avg_val_loss,     # Last epoch's loss
        }, final_model_path)
        print(f"\nFinal model state saved to {final_model_path}")
    except Exception as e:
        print(f"Error saving final model state: {e}")

    # Clean up GPU memory
    del X_train_t, y_train_t, X_val_t, y_val_t, train_dataset, val_dataset, train_loader, val_loader
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return best_results_list # Return the list of best results

# --- Main Incremental Learning Loop ---

# --- General Configuration ---
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    print(f"CUDA Available: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, using CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Simulation/Loading Setup ---
# Replace with your actual data loading/processing logic
# Simulating file paths and parameters for demonstration
BASE_FOLDER_PATH = "Data/Simulated_BTCUSD_1min" # Example path
ensure_folder(BASE_FOLDER_PATH)

# Define list of simulated data files and settings per period
# In reality, these would come from your `list_period_files_full_path`
# and `process_and_return_splits` calls
period_configs = [
    { # Period 1
        "data_file": os.path.join(BASE_FOLDER_PATH, "period_1_data.csv"),
        "trends_to_keep": {0, 1},
        "output_size": 2,
        "model_save_dir": "Class_Incremental_CL/Trained_Models/Period_1",
        "ewc_lambda": 0.0, # No EWC for the first period
        "num_epochs": 5, # Reduced for demonstration
        "input_size": 7, # Assume constant features
    },
    { # Period 2
        "data_file": os.path.join(BASE_FOLDER_PATH, "period_2_data.csv"),
        "trends_to_keep": {0, 1, 2},
        "output_size": 3,
        "model_save_dir": "Class_Incremental_CL/Trained_Models/Period_2",
        "ewc_lambda": 4000.0, # Example EWC strength (needs tuning)
        "num_epochs": 5,
        "input_size": 7,
    },
    { # Period 3
        "data_file": os.path.join(BASE_FOLDER_PATH, "period_3_data.csv"),
        "trends_to_keep": {0, 1, 2, 3},
        "output_size": 4,
        "model_save_dir": "Class_Incremental_CL/Trained_Models/Period_3",
        "ewc_lambda": 4000.0,
        "num_epochs": 5,
        "input_size": 7,
    },
    { # Period 4
        "data_file": os.path.join(BASE_FOLDER_PATH, "period_4_data.csv"),
        "trends_to_keep": {0, 1, 2, 3, 4},
        "output_size": 5,
        "model_save_dir": "Class_Incremental_CL/Trained_Models/Period_4",
        "ewc_lambda": 4000.0,
        "num_epochs": 5,
         "input_size": 7,
    },
    # Add more periods as needed
]

# --- Dummy Data Generation (Replace with your actual data loading) ---
def generate_dummy_data(file_path, trends_to_keep, seq_len=100, n_features=7, n_samples=500):
    """Generates dummy CSV data for demonstration."""
    ensure_folder(os.path.dirname(file_path))
    # This is highly simplified. Your actual data processing is complex.
    # We just need dummy arrays X, y for train/val/test.
    num_classes = len(trends_to_keep)
    # Generate random data and labels within the allowed trend range
    X = np.random.rand(n_samples, seq_len, n_features).astype(np.float32)
    # Ensure labels are drawn only from trends_to_keep
    possible_labels = list(trends_to_keep)
    y = np.random.choice(possible_labels, size=(n_samples, seq_len)).astype(np.int64)

    # Split data (simple split for demo)
    split_idx1 = int(0.8 * n_samples)
    split_idx2 = int(0.9 * n_samples)
    X_tr, y_tr = X[:split_idx1], y[:split_idx1]
    X_v, y_v = X[split_idx1:split_idx2], y[split_idx1:split_idx2]
    X_te, y_te = X[split_idx2:], y[split_idx2:]
    print(f"Generated dummy data for {file_path} with labels {np.unique(y)}")
    return X_tr, y_tr, X_v, y_v, X_te, y_te, n_features

# Generate dummy files if they don't exist
for p_cfg in period_configs:
    if not os.path.exists(p_cfg["data_file"]):
         generate_dummy_data(p_cfg["data_file"], p_cfg["trends_to_keep"], n_features=p_cfg["input_size"])

# --- Model Hyperparameters ---
hidden_size = 64
num_layers = 2 # Reduced layers for faster demo
dropout = 0.1
learning_rate = 0.001
batch_size = 32
use_scheduler = True
ewc_fisher_sample_size = 256 # Number of samples to estimate Fisher (None for all)

# --- Incremental Training Loop ---
previous_best_model_path = None
ewc_fisher_dict = None
ewc_params_dict = None
all_period_results = [] # Store best results from each period

for period_idx, config in enumerate(period_configs):
    current_period = period_idx + 1
    print(f"\n{'='*20} Period {current_period} {'='*20}")
    print(f"Config: {config}")

    # 1. Load Data for the current period
    # Replace dummy generation with your actual data loading function call
    # X_train, y_train, X_val, y_val, X_test, y_test, n_features = process_and_return_splits(...)
    X_train, y_train, X_val, y_val, _, _, n_features = generate_dummy_data(
        config["data_file"], config["trends_to_keep"], n_features=config["input_size"]
    )
    current_output_size = config["output_size"]
    model_save_dir = config["model_save_dir"]
    model_name = f"Period_{current_period}_BiGRU_EWC"
    ensure_folder(model_save_dir)

    # 2. Instantiate Model
    current_model = BiGRUWithAttention(
        input_size=n_features,
        hidden_size=hidden_size,
        output_size=current_output_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    # 3. Load Weights from Previous Period (if applicable)
    ewc_object_for_training = None
    if current_period > 1 and previous_best_model_path:
        print(f"\nLoading state from previous period: {previous_best_model_path}")
        try:
            checkpoint = torch.load(previous_best_model_path, map_location=device)
            prev_state_dict = checkpoint['model_state_dict']
            
             # Load EWC state if saved in the checkpoint
            if 'ewc_fisher' in checkpoint and 'ewc_params' in checkpoint:
                ewc_fisher_dict = checkpoint['ewc_fisher']
                ewc_params_dict = checkpoint['ewc_params']
                print("Loaded EWC Fisher and Params from checkpoint.")
                 # Create EWC object for the current training period using loaded state
                ewc_object_for_training = EWC(fisher=ewc_fisher_dict, params=ewc_params_dict)
            else:
                print("Warning: EWC state not found in previous checkpoint.")
                ewc_object_for_training = None # Ensure it's reset if not found

            # Filter state dict to load only compatible layers
            current_model_dict = current_model.state_dict()
            filtered_prev_state_dict = {
                k: v for k, v in prev_state_dict.items()
                if k in current_model_dict and v.size() == current_model_dict[k].size()
            }

            missing_keys, unexpected_keys = current_model.load_state_dict(filtered_prev_state_dict, strict=False)
            print(f"Loaded compatible weights. Missing keys: {missing_keys}, Unexpected keys (normal if output layer changed): {unexpected_keys}")

            # Optionally load optimizer/scheduler state if desired (ensure compatibility)
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        except FileNotFoundError:
            print(f"Warning: Previous best model path not found: {previous_best_model_path}. Starting from scratch.")
        except Exception as e:
            print(f"Error loading checkpoint from {previous_best_model_path}: {e}. Starting from scratch.")
            previous_best_model_path = None # Reset path if loading fails
            ewc_object_for_training = None

    # 4. Define Loss, Optimizer, Scheduler for the current period
    criterion = nn.CrossEntropyLoss() # Consider ignore_index if you have padding labels
    optimizer = optim.Adam(current_model.parameters(), lr=learning_rate)
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True) # Example scheduler

    # 5. Train the Model for the current period
    period_best_results = train_validate_incremental(
        model=current_model,
        output_size=current_output_size,
        criterion=criterion,
        optimizer=optimizer,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        scheduler=scheduler,
        use_scheduler=use_scheduler,
        ewc_object=ewc_object_for_training, # Pass loaded EWC state
        lambda_ewc=config["ewc_lambda"],
        num_epochs=config["num_epochs"],
        batch_size=batch_size,
        model_saving_folder=model_save_dir,
        model_name=model_name,
        stop_signal_file=None, # Add path if needed
        device=device
    )

    # 6. Post-Training: Process results and Prepare for Next Period
    if not period_best_results:
        print(f"Training failed or was interrupted for Period {current_period}. Stopping.")
        break # Stop the incremental learning process

    # Store the best result of this period
    best_model_info = period_best_results[0]
    all_period_results.append({f"Period_{current_period}": best_model_info})
    previous_best_model_path = best_model_info["model_path"] # Path to the best performing model of this period

    print(f"\nBest model for Period {current_period} saved at: {previous_best_model_path}")
    print(f"Achieved Val Accuracy: {best_model_info['val_accuracy']*100:.2f}%")

    # 7. Compute and Save EWC State for the *Next* Period
    # Use the *best* model from the current period and the *current* period's training data
    print(f"\nComputing EWC Fisher/Params based on Period {current_period} data...")
    # Reload the best model state to ensure we compute Fisher on the actual best weights
    best_checkpoint = torch.load(previous_best_model_path, map_location=device)
    current_model.load_state_dict(best_checkpoint['model_state_dict'])

    # Recreate DataLoader for Fisher computation (or reuse if still available)
    # Important: Use the training data of the *current* period
    train_dataset_ewc = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.long))
    # Use a potentially smaller batch size for Fisher calculation if memory is an issue
    ewc_batch_size = batch_size // 2 if batch_size > 1 else 1
    ewc_dataloader = DataLoader(train_dataset_ewc, batch_size=ewc_batch_size, shuffle=True)

    new_ewc_fisher, new_ewc_params = EWC.compute_fisher_and_params(
        model=current_model,
        dataloader=ewc_dataloader,
        criterion=criterion,
        device=device,
        sample_size=ewc_fisher_sample_size
    )
    print("EWC Fisher/Params computation complete.")

    # Save the computed EWC state *into* the best model checkpoint file
    # This makes the checkpoint self-contained for the next period
    best_checkpoint['ewc_fisher'] = {k: v.cpu() for k, v in new_ewc_fisher.items()} # Save to CPU
    best_checkpoint['ewc_params'] = {k: v.cpu() for k, v in new_ewc_params.items()} # Save to CPU
    try:
        torch.save(best_checkpoint, previous_best_model_path)
        print(f"Saved EWC state into checkpoint: {previous_best_model_path}")
    except Exception as e:
         print(f"Error saving checkpoint with EWC state: {e}")


    # Clean up before next period
    del current_model, optimizer, criterion, scheduler, period_best_results
    del X_train, y_train, X_val, y_val, train_dataset_ewc, ewc_dataloader
    if 'ewc_object_for_training' in locals(): del ewc_object_for_training
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()


print("\n--- Incremental Learning Finished ---")
print("Summary of best results per period:")
for result in all_period_results:
    print(result)