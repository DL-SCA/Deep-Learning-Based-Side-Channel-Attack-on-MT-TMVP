"""Deep learning side-channel attack on MT-TMVP.

Trains one DeepResidualCNN per target coefficient, evaluates cross-coefficient
transferability, and computes the all-correct rate (ACR).
"""

import os
import sys
import gc
import time
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set memory allocation strategy before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

SEED = 42
FULL_TRACE_LENGTH = 20000
NUM_CLASSES = 3
NUM_COEFFICIENTS = 10

# ============================================================================
# LOGGING
# ============================================================================

class TeeLogger:
    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    def close(self):
        self.log_file.close()

# ============================================================================
# DATA AUGMENTATION (Vectorized for speed)
# ============================================================================

def augment_batch(traces, noise_std=0.02, time_shift=10):
    """Vectorized augmentation on GPU tensors."""
    if noise_std > 0:
        noise = torch.randn_like(traces) * noise_std * traces.std()
        traces = traces + noise
    return traces

# ============================================================================
# MODEL - DeepResidualCNN
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)

class DeepResidualCNN(nn.Module):
    def __init__(self, input_length=20000, num_classes=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(64), nn.MaxPool1d(4),
            ResidualBlock(64), nn.MaxPool1d(4),
            ResidualBlock(64), nn.MaxPool1d(4),
            ResidualBlock(64), nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes),
        )
        
    def forward(self, x):
        return self.head(self.res_blocks(self.stem(x)))

# ============================================================================
# DATA LOADING - Optimized
# ============================================================================

def load_all_traces_fast(dataset_path: Path):
    """Fast parallel loading of traces."""
    traces_dir = dataset_path / "traces"
    trace_files = sorted(traces_dir.glob("traces_*.npz"), key=lambda x: int(x.stem.split('_')[1]))
    
    # First pass: count total and get shapes
    n_files = len(trace_files)
    
    # Load first file to get shape
    first_data = np.load(trace_files[0])
    trace_len = first_data['wave'].flatten().shape[0]
    label_len = first_data['labels'].flatten().shape[0]
    
    # Pre-allocate arrays
    all_traces = np.zeros((n_files, trace_len), dtype=np.float32)
    all_labels = np.zeros((n_files, label_len), dtype=np.int64)
    
    # Load in parallel-friendly way
    for i, tf in enumerate(tqdm(trace_files, desc="Loading traces", leave=False)):
        data = np.load(tf)
        all_traces[i] = data['wave'].flatten()
        all_labels[i] = data['labels'].flatten()
    
    return all_traces, all_labels

def create_split(num_samples, train_ratio=0.70, val_ratio=0.15, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(num_samples)
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    return indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def get_gpu_memory_info():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU Mem: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "No GPU"

# ============================================================================
# CROSS-COEFFICIENT EVALUATION - Optimized
# ============================================================================

def evaluate_cross_coefficient_fast(model, traces_tensor, all_labels, indices, 
                                     device, batch_size=1024):
    """
    Fast cross-coefficient evaluation using pre-normalized tensor.
    """
    model.eval()
    n_samples = len(indices)
    
    # Get predictions in batches
    all_preds = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_indices = indices[i:end_idx]
            batch = traces_tensor[batch_indices].to(device, non_blocking=True)
            with autocast():
                outputs = model(batch)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.append(preds)
            del batch, outputs
    
    all_preds = torch.cat(all_preds).numpy()
    
    # Evaluate against each coefficient
    cross_acc = {}
    labels_subset = all_labels[indices]
    for target_coeff in range(NUM_COEFFICIENTS):
        true_labels = labels_subset[:, target_coeff] + 1
        cross_acc[target_coeff] = accuracy_score(true_labels, all_preds)
    
    return cross_acc

# ============================================================================
# TRAINING - Optimized
# ============================================================================

def train_model_optimized(traces_tensor, labels_tensor, train_indices, val_indices,
                          all_labels_np, coeff_idx, epochs, patience, use_mixup,
                          batch_size, save_dir, device, num_gpus):
    """
    Optimized training with proper memory management.
    """
    # Create model fresh
    torch.manual_seed(SEED + coeff_idx * 100)
    model = DeepResidualCNN().to(device)
    
    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    
    # Get labels for this coefficient
    train_labels = torch.LongTensor(all_labels_np[train_indices, coeff_idx] + 1)
    val_labels = torch.LongTensor(all_labels_np[val_indices, coeff_idx] + 1)
    
    # Create simple tensor datasets (data already normalized)
    train_dataset = TensorDataset(
        traces_tensor[train_indices],
        train_labels
    )
    val_dataset = TensorDataset(
        traces_tensor[val_indices],
        val_labels
    )
    
    eff_batch = batch_size * num_gpus
    train_loader = DataLoader(train_dataset, batch_size=eff_batch, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=eff_batch * 2, shuffle=False,
                            num_workers=0, pin_memory=False)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=epochs, 
                          steps_per_epoch=len(train_loader), pct_start=0.1)
    scaler = GradScaler()
    
    best_val_acc, best_state, patience_cnt = 0, None, 0
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [], 'epoch_time': [], 'lr': [],
        'cross_coeff_acc': []
    }
    
    total_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_preds, train_labels_list = [], []
        
        pbar = tqdm(train_loader, desc=f"C{coeff_idx} E{epoch+1}/{epochs}", 
                    leave=False, dynamic_ncols=True)
        for traces_batch, labels in pbar:
            traces_batch = traces_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(traces_batch)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            _, pred = outputs.max(1)
            train_total += labels.size(0)
            train_correct += pred.eq(labels).sum().item()
            train_preds.extend(pred.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*train_correct/train_total:.1f}%'})
            
            # Clear intermediate tensors
            del traces_batch, labels, outputs, loss, pred
        pbar.close()
        
        train_acc = train_correct / train_total
        train_f1 = f1_score(train_labels_list, train_preds, average='macro', zero_division=0)
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for traces_batch, labels in val_loader:
                traces_batch = traces_batch.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast():
                    outputs = model(traces_batch)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, pred = outputs.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()
                val_preds.extend(pred.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
                del traces_batch, labels, outputs, loss, pred
        
        val_acc = val_correct / val_total
        val_f1 = f1_score(val_labels_list, val_preds, average='macro', zero_division=0)
        
        # Cross-coefficient evaluation (use base model)
        eval_model = model.module if hasattr(model, 'module') else model
        cross_acc = evaluate_cross_coefficient_fast(
            eval_model, traces_tensor, all_labels_np, val_indices, device
        )
        
        epoch_time = time.time() - epoch_start
        
        # Update history
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['epoch_time'].append(epoch_time)
        history['lr'].append(scheduler.get_last_lr()[0])
        history['cross_coeff_acc'].append(cross_acc)
        
        # Print epoch summary
        cross_str = " | ".join([f"C{c}:{cross_acc[c]*100:.1f}%" for c in range(NUM_COEFFICIENTS)])
        print(f"  Epoch {epoch+1:3d}: train={train_acc*100:.1f}%, val={val_acc*100:.1f}%, "
              f"f1={val_f1:.3f}, time={epoch_time:.1f}s")
        print(f"    Cross-coeff: {cross_str}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            best_state = {k: v.cpu().clone() for k, v in state_dict.items()}
            patience_cnt = 0
            if save_dir:
                torch.save({'model_state_dict': best_state, 'val_acc': best_val_acc, 'epoch': epoch}, 
                          save_dir / f"coeff_{coeff_idx}_best.pth")
        else:
            patience_cnt += 1
        
        if patience_cnt >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
        
        # Clear memory at end of epoch
        torch.cuda.empty_cache()
    
    history['total_train_time'] = time.time() - total_start
    history['best_val_acc'] = best_val_acc
    
    # Cleanup
    del model, optimizer, scheduler, scaler, criterion
    del train_loader, val_loader, train_dataset, val_dataset
    del train_labels, val_labels
    clear_gpu_memory()
    
    return best_val_acc, best_state, history

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_training_curves(history, coeff_idx, save_dir):
    """Plot comprehensive training curves for a single coefficient."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r--', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Coefficient {coeff_idx}: Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [a*100 for a in history['train_acc']], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, [a*100 for a in history['val_acc']], 'r--', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Coefficient {coeff_idx}: Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # F1 Score
    ax = axes[0, 2]
    ax.plot(epochs, history['train_f1'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_f1'], 'r--', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score (macro)')
    ax.set_title(f'Coefficient {coeff_idx}: F1 Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning Rate
    ax = axes[1, 0]
    ax.plot(epochs, history['lr'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title(f'Coefficient {coeff_idx}: Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Epoch Time
    ax = axes[1, 1]
    ax.bar(epochs, history['epoch_time'], color='steelblue', alpha=0.7)
    ax.axhline(np.mean(history['epoch_time']), color='red', linestyle='--', 
               label=f"Mean: {np.mean(history['epoch_time']):.1f}s")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time (s)')
    ax.set_title(f'Coefficient {coeff_idx}: Epoch Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cross-coefficient accuracy over epochs (for trained coeff)
    ax = axes[1, 2]
    trained_coeff_acc = [h[coeff_idx]*100 for h in history['cross_coeff_acc']]
    ax.plot(epochs, trained_coeff_acc, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Coefficient {coeff_idx}: Own-Coefficient Val Accuracy')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'training_curves_coeff_{coeff_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_cross_coefficient_evolution(history, coeff_idx, save_dir):
    """Plot how cross-coefficient accuracy evolves during training."""
    epochs = range(1, len(history['cross_coeff_acc']) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_COEFFICIENTS))
    
    for target_c in range(NUM_COEFFICIENTS):
        accs = [h[target_c]*100 for h in history['cross_coeff_acc']]
        linewidth = 3 if target_c == coeff_idx else 1.5
        linestyle = '-' if target_c == coeff_idx else '--'
        label = f'C{target_c}' + (' (trained)' if target_c == coeff_idx else '')
        ax.plot(epochs, accs, color=colors[target_c], linewidth=linewidth, 
                linestyle=linestyle, label=label, marker='o' if target_c == coeff_idx else None,
                markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Model Trained on Coeff {coeff_idx}: Cross-Coefficient Accuracy Evolution', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    ax.axhline(33.33, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'cross_coeff_evolution_coeff_{coeff_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, coeff_idx, save_dir):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['-1', '0', '1'], yticklabels=['-1', '0', '1'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix - Coefficient {coeff_idx}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / f'confusion_matrix_coeff_{coeff_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_overall_summary(all_results, save_dir):
    """Plot overall summary across all coefficients."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    coeffs = [r['coeff_idx'] for r in all_results]
    test_accs = [r['test_metrics']['accuracy'] * 100 for r in all_results]
    f1_scores = [r['test_metrics']['f1_macro'] for r in all_results]
    precisions = [r['test_metrics']['precision_macro'] for r in all_results]
    recalls = [r['test_metrics']['recall_macro'] for r in all_results]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(coeffs)))
    
    # Test accuracy
    ax = axes[0, 0]
    bars = ax.bar(coeffs, test_accs, color=colors)
    ax.axhline(np.mean(test_accs), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(test_accs):.2f}%')
    ax.set_xlabel('Coefficient Index')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy per Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # F1 Score
    ax = axes[0, 1]
    bars = ax.bar(coeffs, f1_scores, color=colors)
    ax.axhline(np.mean(f1_scores), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(f1_scores):.3f}')
    ax.set_xlabel('Coefficient Index')
    ax.set_ylabel('F1 Score (macro)')
    ax.set_title('F1 Score per Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Precision vs Recall
    ax = axes[1, 0]
    x = np.arange(len(coeffs))
    width = 0.35
    ax.bar(x - width/2, precisions, width, label='Precision', color='forestgreen')
    ax.bar(x + width/2, recalls, width, label='Recall', color='darkorange')
    ax.set_xlabel('Coefficient Index')
    ax.set_ylabel('Score')
    ax.set_title('Precision vs Recall per Coefficient')
    ax.set_xticks(x)
    ax.set_xticklabels(coeffs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Training time
    ax = axes[1, 1]
    times = [r['training_time'] for r in all_results]
    ax.bar(coeffs, times, color='steelblue')
    ax.axhline(np.mean(times), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(times):.1f}s')
    ax.set_xlabel('Coefficient Index')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Training Time per Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'overall_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_cross_coefficient_heatmap(all_results, save_dir):
    """Plot heatmap of cross-coefficient prediction accuracy."""
    n_coeffs = NUM_COEFFICIENTS
    
    acc_matrix = np.zeros((n_coeffs, n_coeffs))
    for result in all_results:
        trained_coeff = result['coeff_idx']
        final_cross_acc = result['history']['cross_coeff_acc'][-1]
        for target_c in range(n_coeffs):
            acc_matrix[trained_coeff, target_c] = final_cross_acc[target_c] * 100
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(acc_matrix, annot=True, fmt='.1f', cmap='RdYlGn', 
                vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
    
    ax.set_xlabel('Target Coefficient (to predict)', fontsize=12)
    ax.set_ylabel('Model Trained on Coefficient', fontsize=12)
    ax.set_title('Cross-Coefficient Prediction Accuracy (%)', fontsize=14)
    ax.set_xticklabels([f'C{i}' for i in range(n_coeffs)])
    ax.set_yticklabels([f'C{i}' for i in range(n_coeffs)])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'cross_coefficient_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_all_training_curves_combined(all_results, save_dir):
    """Plot all coefficients' training curves in one figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_COEFFICIENTS))
    
    # Train accuracy
    ax = axes[0, 0]
    for result in all_results:
        coeff = result['coeff_idx']
        epochs = range(1, len(result['history']['train_acc']) + 1)
        ax.plot(epochs, [a*100 for a in result['history']['train_acc']], 
                color=colors[coeff], label=f'C{coeff}', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Accuracy (%)')
    ax.set_title('Training Accuracy - All Coefficients')
    ax.legend(loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Val accuracy
    ax = axes[0, 1]
    for result in all_results:
        coeff = result['coeff_idx']
        epochs = range(1, len(result['history']['val_acc']) + 1)
        ax.plot(epochs, [a*100 for a in result['history']['val_acc']], 
                color=colors[coeff], label=f'C{coeff}', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Accuracy (%)')
    ax.set_title('Validation Accuracy - All Coefficients')
    ax.legend(loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Train loss
    ax = axes[1, 0]
    for result in all_results:
        coeff = result['coeff_idx']
        epochs = range(1, len(result['history']['train_loss']) + 1)
        ax.plot(epochs, result['history']['train_loss'], 
                color=colors[coeff], label=f'C{coeff}', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss')
    ax.set_title('Training Loss - All Coefficients')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Val F1
    ax = axes[1, 1]
    for result in all_results:
        coeff = result['coeff_idx']
        epochs = range(1, len(result['history']['val_f1']) + 1)
        ax.plot(epochs, result['history']['val_f1'], 
                color=colors[coeff], label=f'C{coeff}', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val F1 (macro)')
    ax.set_title('Validation F1 Score - All Coefficients')
    ax.legend(loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'all_training_curves_combined.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model_fast(best_state, traces_tensor, all_labels_np, test_indices, 
                        coeff_idx, batch_size, device):
    """Fast evaluation using pre-loaded tensors."""
    # Create fresh model and load state
    model = DeepResidualCNN().to(device)
    model.load_state_dict(best_state)
    model.eval()
    
    test_labels = all_labels_np[test_indices, coeff_idx] + 1
    n_test = len(test_indices)
    
    all_preds = []
    start = time.time()
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            end_idx = min(i + batch_size, n_test)
            batch_indices = test_indices[i:end_idx]
            batch = traces_tensor[batch_indices].to(device, non_blocking=True)
            with autocast():
                outputs = model(batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            del batch, outputs
    inference_time = time.time() - start
    
    all_preds = np.array(all_preds)
    
    metrics = compute_metrics(test_labels, all_preds)
    metrics['inference_time_total'] = inference_time
    metrics['inference_latency_per_sample_ms'] = inference_time / n_test * 1000
    metrics['confusion_matrix'] = confusion_matrix(test_labels, all_preds).tolist()
    
    # Get final cross-coeff on test
    final_cross_acc = evaluate_cross_coefficient_fast(
        model, traces_tensor, all_labels_np, test_indices, device
    )
    
    del model
    clear_gpu_memory()
    
    return metrics, all_preds, test_labels, final_cross_acc

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DL-SCA Attack on MT-TMVP")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size per GPU (reduced for memory)")
    parser.add_argument("--coefficients", type=int, nargs='+', default=None)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-mixup", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=None)
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    models_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    log_file = output_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    tee = TeeLogger(log_file)
    sys.stdout = tee
    
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    num_gpus = args.num_gpus or torch.cuda.device_count()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    target_coefficients = args.coefficients or list(range(NUM_COEFFICIENTS))
    
    print(f"\n{'='*70}")
    print("DL-SCA ATTACK V8 - OPTIMIZED SINGLE MODEL")
    print(f"{'='*70}")
    print(f"Model: DeepResidualCNN (single model per coefficient)")
    print(f"GPUs: {num_gpus}, Batch size per GPU: {args.batch_size}")
    print(f"Effective batch size: {args.batch_size * num_gpus}")
    print(f"Epochs: {args.epochs}, Patience: {args.patience}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Target coefficients: {target_coefficients}")
    print(f"Augmentation: {not args.no_augment}, Mixup: {not args.no_mixup}")
    
    # Load data
    print(f"\n{'='*70}\nLOADING DATA\n{'='*70}")
    t0 = time.time()
    traces_np, all_labels_np = load_all_traces_fast(dataset_path)
    print(f"Loaded {len(traces_np)} traces in {time.time()-t0:.1f}s")
    print(f"Trace shape: {traces_np.shape}, Labels shape: {all_labels_np.shape}")
    
    # Split
    train_idx, val_idx, test_idx = create_split(len(traces_np), 0.70, 0.15, SEED)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Compute normalization from training data
    print("Computing normalization statistics...")
    t0 = time.time()
    global_mean = traces_np[train_idx].mean()
    global_std = traces_np[train_idx].std() + 1e-8
    print(f"Normalization: mean={global_mean:.6f}, std={global_std:.6f} ({time.time()-t0:.1f}s)")
    
    # Normalize ALL data once and convert to tensor (stays on CPU)
    print("Normalizing and converting to tensor...")
    t0 = time.time()
    traces_np = (traces_np - global_mean) / global_std
    traces_tensor = torch.FloatTensor(traces_np).unsqueeze(1)  # Add channel dim
    print(f"Tensor shape: {traces_tensor.shape} ({time.time()-t0:.1f}s)")
    print(f"Tensor memory: {traces_tensor.element_size() * traces_tensor.nelement() / 1e9:.2f} GB")
    
    # Free numpy array
    del traces_np
    gc.collect()
    
    all_results = []
    all_test_predictions = {}
    total_start = time.time()
    
    for coeff_idx in target_coefficients:
        print(f"\n{'='*70}")
        print(f"TRAINING COEFFICIENT {coeff_idx}")
        print(f"{'='*70}")
        print(f"  {get_gpu_memory_info()}")
        
        coeff_start = time.time()
        
        # Train
        best_val_acc, best_state, history = train_model_optimized(
            traces_tensor, None, train_idx, val_idx, all_labels_np,
            coeff_idx, args.epochs, args.patience, not args.no_mixup,
            args.batch_size, models_dir, device, num_gpus
        )
        
        # Evaluate
        print(f"\n  Evaluating on test set...")
        eff_batch = args.batch_size * num_gpus * 2
        metrics, preds, true_labels, final_cross_acc = evaluate_model_fast(
            best_state, traces_tensor, all_labels_np, test_idx, coeff_idx, 
            eff_batch, device
        )
        all_test_predictions[coeff_idx] = preds
        
        coeff_time = time.time() - coeff_start
        
        print(f"\n  Coeff {coeff_idx} RESULTS:")
        print(f"    Test accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"    F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"    Training time: {coeff_time:.1f}s")
        print(f"    Final cross-coeff (test): " + 
              " | ".join([f"C{c}:{final_cross_acc[c]*100:.1f}%" for c in range(NUM_COEFFICIENTS)]))
        
        result = {
            'coeff_idx': coeff_idx,
            'test_metrics': metrics,
            'training_time': coeff_time,
            'history': history,
            'final_cross_coeff_test': final_cross_acc,
        }
        all_results.append(result)
        
        # Plot for this coefficient
        plot_training_curves(history, coeff_idx, plots_dir)
        plot_cross_coefficient_evolution(history, coeff_idx, plots_dir)
        plot_confusion_matrix(np.array(metrics['confusion_matrix']), coeff_idx, plots_dir)
        
        # Cleanup between coefficients
        del best_state
        clear_gpu_memory()
        print(f"  {get_gpu_memory_info()}")
    
    # Final summary
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    
    # All-correct
    n_test = len(test_idx)
    all_correct_mask = np.ones(n_test, dtype=bool)
    test_labels_all = all_labels_np[test_idx]
    for coeff_idx in target_coefficients:
        true_labels_coeff = test_labels_all[:, coeff_idx] + 1
        all_correct_mask &= (all_test_predictions[coeff_idx] == true_labels_coeff)
    num_all_correct = all_correct_mask.sum()
    
    print(f"\nPer-Coefficient Results:")
    print(f"{'Coeff':<8} {'Test Acc':<12} {'F1':<10} {'Precision':<12} {'Recall':<10}")
    print("-" * 55)
    for r in all_results:
        m = r['test_metrics']
        print(f"{r['coeff_idx']:<8} {m['accuracy']*100:>6.2f}%     {m['f1_macro']:.4f}    "
              f"{m['precision_macro']:.4f}      {m['recall_macro']:.4f}")
    
    avg_acc = np.mean([r['test_metrics']['accuracy'] for r in all_results])
    avg_f1 = np.mean([r['test_metrics']['f1_macro'] for r in all_results])
    print("-" * 55)
    print(f"{'Average':<8} {avg_acc*100:>6.2f}%     {avg_f1:.4f}")
    
    print(f"\nAll-correct: {num_all_correct}/{n_test} ({100*num_all_correct/n_test:.2f}%)")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Generate summary plots
    print(f"\nGenerating summary plots...")
    plot_overall_summary(all_results, plots_dir)
    plot_cross_coefficient_heatmap(all_results, plots_dir)
    plot_all_training_curves_combined(all_results, plots_dir)
    
    # Save results
    results_to_save = []
    for r in all_results:
        r_copy = r.copy()
        r_copy['history'] = {
            k: v if not isinstance(v, np.ndarray) else v.tolist() 
            for k, v in r['history'].items()
        }
        r_copy['final_cross_coeff_test'] = {str(k): v for k, v in r['final_cross_coeff_test'].items()}
        results_to_save.append(r_copy)
    
    with open(output_dir / "attack_results.json", 'w') as f:
        json.dump({
            'avg_test_accuracy': float(avg_acc),
            'avg_f1_macro': float(avg_f1),
            'all_correct_rate': float(num_all_correct / n_test),
            'total_time_s': total_time,
            'per_coefficient': results_to_save,
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Plots saved to: {plots_dir}")
    
    print(f"\n{'='*70}\nCOMPLETE\n{'='*70}")
    sys.stdout = tee.terminal
    tee.close()

if __name__ == "__main__":
    main()
