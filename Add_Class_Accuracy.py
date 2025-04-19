#
#
#
#
# Changes for all notebooks.
#
#
#
#
"""
🛠️ Changes to apply inside the training and validation functions (including continual learning versions):

1. Use a cleaner `current_epoch_info` structure to track training progress.
2. Save the model using unpacked `current_epoch_info` (excluding 'model_path').
3. Scheduler call's comment was corrected.
4. Save a final model after training is complete was updated.
5. Print the top 5 models by validation accuracy was updated.

        # -----------------------------------------------------------
        # Save current model and update best results if applicable
        current_epoch_info = {
            "epoch": epoch+1,
            "train_loss": train_loss,
            "Train-Class-Acc": train_classwise_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "Val-Class-Acc": val_classwise_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr'], # Optimizer state
            "model_path": os.path.join(model_saving_folder, f"{model_name}_epoch_{epoch+1}.pth")
        }


            # -----------------------------------------------------------
            torch.save({
                **{k: v for k, v in current_epoch_info.items() if k != "model_path"},
                "model_state_dict": model.state_dict(), # <<<<<<<<<< Sometimes this should be student_model instead of model >>>>>>>>>
                "optimizer_state_dict": optimizer.state_dict()
            }, current_epoch_info["model_path"])
            print(f"Model saved after epoch {epoch+1} to {current_epoch_info['model_path']} \n")


        # -----------------------------------------------------------
        if use_scheduler == True:
            # Scheduler step should follow after considering the results (placed after other losses)
            scheduler.step(val_loss)


    # -----------------------------------------------------------
    # Save the final model
    if current_epoch_info:
        final_model_path = os.path.join(model_saving_folder, f"{model_name}_final.pth")
        torch.save({ # Save this model
            **{k: v for k, v in current_epoch_info.items() if k != "model_path"},
            "model_state_dict": model.state_dict(), # <<<<<<<<<< Sometimes this should be student_model instead of model >>>>>>>>>
            "optimizer_state_dict": optimizer.state_dict()
        }, final_model_path)
        print(f"\nFinal model saved to {final_model_path}")
            

    # -----------------------------------------------------------
    print("\nTraining complete. \n\nTop 5 Models Sorted by Validation Accuracy: ")
    for res in best_results:        
        print(f"Epoch {res['epoch']}/{num_epochs}, "
              f"Train Loss: {res['train_loss']:.9f}, "
              f"Train-Class-Acc: {res['Train-Class-Acc']}, " 
              f"Val Loss: {res['val_loss']:.9f}, "
              f"Val Accuracy: {res['val_accuracy'] * 100:.2f}%, "
              f"Val-Class-Acc: {res['Val-Class-Acc']}, "
              f"Model Path: {res['model_path']}")
    print('\n')
"""




# -----------------------------------------------------------
# -----------------------------------------------------------
"""
📈 Add back classwise accuracy & overall val accuracy to checkpoints
"""
# -----------------------------------------------------------
# -----------------------------------------------------------




# -----------------------------------------------------------
# Same compute_classwise_accuracy() as before
def compute_classwise_accuracy(student_logits_flat, y_batch, class_correct, class_total):
    """
    Computes per-class accuracy by accumulating correct and total samples for each class using vectorized operations.
    
    Args:
        student_logits_flat (torch.Tensor): Model predictions (logits) in shape [batch_size * seq_len, output_size]
        y_batch (torch.Tensor): True labels in shape [batch_size * seq_len]
        class_correct (dict): Dictionary to store correct predictions per class
        class_total (dict): Dictionary to store total samples per class
    """
    # Ensure inputs are on the same device
    if student_logits_flat.device != y_batch.device:
        raise ValueError("student_logits_flat and y_batch must be on the same device")

    # Convert logits to predicted class indices
    predictions = torch.argmax(student_logits_flat, dim=-1)  # Shape: [batch_size * seq_len]

    # Compute correct predictions mask
    correct_mask = (predictions == y_batch)  # Shape: [batch_size * seq_len], boolean

    # Get unique labels in this batch
    unique_labels = torch.unique(y_batch)

    # Update class_total and class_correct using vectorized operations
    for label in unique_labels:
        label = label.item()  # Convert tensor to scalar
        if label not in class_total:
            class_total[label] = 0
            class_correct[label] = 0
        
        # Count total samples for this label
        label_mask = (y_batch == label)
        class_total[label] += label_mask.sum().item()
        
        # Count correct predictions for this label
        class_correct[label] += (label_mask & correct_mask).sum().item()


# -----------------------------------------------------------
"""
New function needed to add.
    This function will not be able to add all elements as the changes above  to the train and validation functions, 
    but it can add the elements we have been printing and missing (val accuracy and Val-Class-Acc).
"""
def add_val_class_acc(model_copy, output_size, criterion, X_val, y_val, 
                      file_path, model_checkpoint, batch_size=64):
    """
    Updates a model checkpoint with computed validation accuracy and classwise accuracy.

    Args:
        model_copy: A deepcopy of the model to evaluate
        output_size (int): Number of output classes
        criterion: Loss function used
        X_val, y_val: Validation features and labels (NumPy arrays or compatible)
        file_path (str): Path where the updated checkpoint should be saved
        model_checkpoint (dict): The checkpoint dictionary loaded from the file
        batch_size (int): Batch size used for validation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_copy.to(device)
    # Convert data to tensors # Returns a copy, original is safe
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    # Create TensorDatasets
    val_dataset = TensorDataset(X_val, y_val)
    # Create DataLoaders
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Perform validation at the end of each epoch
    model_copy.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_class_correct = {}
    val_class_total = {}
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            val_outputs = model_copy(X_val_batch).view(-1, output_size)
            val_labels = y_val_batch.view(-1)
            val_loss += criterion(val_outputs, val_labels).item() * X_val_batch.size(0)  # Scale to total loss
            val_predictions = torch.argmax(val_outputs, dim=-1)
            val_correct += (val_predictions == val_labels).sum().item()
            val_total += val_labels.size(0)
            # Compute per-class validation accuracy
            compute_classwise_accuracy(val_outputs, val_labels, val_class_correct, val_class_total)
    val_loss /= len(val_loader.dataset)
    val_accuracy = val_correct / val_total
    val_classwise_accuracy = {int(c): f"{(val_class_correct[c] / val_class_total[c]) * 100:.2f}%" if val_class_total[c] > 0 else "0.00%" 
                                for c in sorted(val_class_total.keys())}

    # -----------------------------------------------------------
    model_checkpoint.update({
        "val_accuracy": val_accuracy,
        "Val-Class-Acc": val_classwise_accuracy
    })
    torch.save(model_checkpoint, file_path)
    print(f"Model updated and saved to {file_path} \n")
      
    for var in [
        "X_val", "y_val", "model_copy", "model_checkpoint"
    ]:
        if var in locals():
            del locals()[var]
    # --- Force garbage collection ---
    torch.cuda.empty_cache()
    gc.collect()
    # -----------------------------------------------------------


"""
Optional helper to bulk update checkpoints from a folder
"""
def update_checkpoints_with_val_acc(model_template, ckpt_folder, output_size, criterion, X_val, y_val, batch_size=64):
    """
    Automatically loads and updates all .pth checkpoints in a folder
    using provided model template, val set, and loss function.
    """
    ckpt_paths = sorted([os.path.normpath(p) for p in glob.glob(os.path.join(ckpt_folder, "*.pth"))])
    print(f"\nFound {len(ckpt_paths)} checkpoints in {ckpt_folder}\n")

    for file_path in ckpt_paths:
        # --- make an isolated copy of the template model ---
        model_copy = copy.deepcopy(model_template)
        # --- load checkpoint weights into the copy ---
        checkpoint = torch.load(file_path, map_location="cpu", weights_only=True)
        model_copy.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded pretrained model from: {file_path}")
        # --- compute val / class‑wise accuracy & update checkpoint ---
        add_val_class_acc(model_copy, output_size, criterion, X_val, y_val, file_path, checkpoint, batch_size=batch_size)
        del model_copy, checkpoint
        torch.cuda.empty_cache()
    gc.collect()




# Centralize imports to avoid double imports
import os
import glob
import copy
import torch


"""
Load required & appropriate files
"""
"""
- 'trend': Categorized trend values based on the detected phases:
    - 0: No trend
    - 1: Moderate negative trend
    - 2: Very strong negative trend
    - 3: Moderate positive trend
    - 4: Very strong positive trend
"""
with contextlib.redirect_stdout(open(os.devnull, 'w')):
    X_train, y_train, X_val, y_val, X_test, y_test, Number_features = process_and_return_splits(
        with_indicators_file_path = list_period_files_full_path[1], # Change 
        downsampled_data_minutes = downsampled_data_minutes,
        exclude_columns = exclude_columns,
        lower_threshold = lower_threshold,
        upper_threshold = upper_threshold,
        reverse_steps = reverse_steps,
        sequence_length = sequence_length,
        sliding_interval = sliding_interval,
        trends_to_keep = {0, 1, 2}  # Default keeps all trends : {0, 1, 2, 3, 4}
        # trends_to_keep = {0, 1, 2, 3, 4}  # Default keeps all trends
    )
print(f"\nNumber_features = {Number_features}")
unique_classes = np.unique(y_val)
num_classes = len(unique_classes)
print(f"unique_classes = {unique_classes}")
print(f"num_classes = {num_classes}")
print_class_distribution(y_val, "y_val")
del X_val, y_val



"""
Model parameters needed
"""
# -----------------------------------------------------------
input_size = Number_features  # Number of features
hidden_size = 64  # Number of GRU units
output_size = num_classes # Must be dynamic, up to 5  # Number of trend classes (0, 15, 25, -15, -25)
num_layers = 4  # Number of GRU layers
dropout = 0.0
lora_r = 4 # Rank of the low-rank update matrices
batch_size= 64 # How many sequences passed at once to the model


"""
model_saving_folder needed
"""
model_saving_folder = os.path.normpath(os.path.join('Class_Incremental_CL', "Classif_Bi_Dir_GRU_Model/Trained_models/LoRA/Period_2/1st_try"))


# 📌 NOTE: Sometimes you should use `student_model` for the `model` — check your context.
"""
Instantiate the model object as model (create object from classes or combo of classes, depends on the context)
"""
#--------------------------------
# Instantiate the previous model on cpu, since you will add the pretrained weights later
#-------------------------------------------------------------------------

# rebuild architecture
base_stub = BiGRUWithAttention(input_size, hidden_size, output_size-1,
                               num_layers, dropout)
model = BiGRUWithLoRA(base_stub,
                      old_num_classes=output_size-1,
                      new_total_classes=output_size,
                      lora_rank=lora_r)
#-------------------------------------------------------------------------


"""
Get the criterion from the cell where the training function was called
"""
# Define the loss function, optimizer and scheduler
criterion = nn.CrossEntropyLoss()



"""
Add the missing elements to each checkpoints through this for loop
"""
update_checkpoints_with_val_acc(model, model_saving_folder, output_size, criterion, X_val, y_val, batch_size=64)


"""
Delete all that is left and needs to be deleted
"""
for var in [
    "X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
    "Number_features", "unique_classes", "num_classes",
    "model_saving_folder", "model", "criterion", 
]:
    if var in locals():
        del locals()[var]
# --- Force garbage collection ---
torch.cuda.empty_cache()
gc.collect()
