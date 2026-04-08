import os
import math
import random
import wfdb
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import tqdm
import warnings
import scipy.io as sio

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.signal import butter, filtfilt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.benchmark = True

def bandpass_eeg_signal(data, fs=250.0, lowcut=2.0, highcut=30.0):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

class BCI2aDataset(Dataset):
    def __init__(self, data_dir, subjects, is_train=True):
        self.samples = []
        self.labels = []
        self.is_train = is_train
        suffix = 'T' if is_train else 'E'
        
        for subj in subjects:
            filepath = os.path.join(data_dir, f"A{subj:02d}{suffix}.mat")
            if not os.path.exists(filepath):
                print(f"Warning: {filepath} not found. Skipping.")
                continue
                
            mat_dict = sio.loadmat(filepath)
            data_array = mat_dict['data']
            
            for run_idx in range(data_array.shape[1]):
                run_struct = data_array[0, run_idx]
                
                if isinstance(run_struct, np.ndarray) and run_struct.shape == (1, 1):
                    run_struct = run_struct[0, 0]
                    
                if 'y' not in run_struct.dtype.names or len(run_struct['y']) == 0:
                    continue
                    
                X_cont = run_struct['X'][:, :22].T.astype(np.float32)
                X_cont = bandpass_eeg_signal(X_cont, fs=250.0)
                
                run_mean = X_cont.mean(axis=-1, keepdims=True)
                run_std = X_cont.std(axis=-1, keepdims=True) + 1e-8
                X_cont = (X_cont - run_mean) / run_std
                
                trials = run_struct['trial'].flatten()
                y_run = run_struct['y'].flatten()
                
                offset = 475 
                max_window_size = 1050 
                
                for i, start_idx in enumerate(trials):
                    actual_start = start_idx + offset
                    if actual_start + max_window_size > X_cont.shape[1]:
                        continue
                        
                    raw_label = y_run[i]
                    if np.isnan(raw_label):
                        continue
                    
                    # BCI2a classes: 1=Left, 2=Right, 3=Foot, 4=Tongue
                    if int(raw_label) not in [1, 2]:
                        continue
                        
                    trial_data = X_cont[:, actual_start : actual_start + max_window_size]
                    
                    label = int(raw_label) - 1 
                    self.samples.append(trial_data)
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.samples[idx]).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
        x = x[:, :500]
            
        return x, y


def get_eeg_dataloaders(data_dir="./ml", batch_size=64):
    all_subjects = list(range(1, 10))
    
    train_dataset = BCI2aDataset(data_dir, subjects=all_subjects, is_train=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    eval_dataset = BCI2aDataset(data_dir, subjects=all_subjects, is_train=False)
    val_loader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader


#AGFL
def sparsity_schedule(l, L, smax=0.2, smin=0.8, alpha=3.0):
    return smin + (smax - smin) * math.exp(-alpha * l / L)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss) 
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class GraphConstructor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.scale = math.sqrt(dim) 

    def forward(self, X):
        tau = self.temperature.clamp(0.1, 5.0)
        scores = (X @ X.transpose(-1, -2)) / (self.scale * tau)
        return scores

class GraphFilter(nn.Module):
    def __init__(self, dim, K, separate_W=True):
        super().__init__()
        self.K = K
        self.separate_W = separate_W
        self.alpha_logits = nn.Parameter(torch.zeros(K + 1))
        
        if separate_W:
            self.W = nn.ModuleList([
                nn.Linear(dim, dim, bias=False)
                for _ in range(K + 1)
            ])
        else:
            self.W = nn.Linear(dim, dim, bias=False)

    def forward(self, A, X):
        alpha = self.alpha_logits
        
        P_k = X 
        if self.separate_W:
            H = alpha[0] * self.W[0](P_k)
        else:
            H = alpha[0] * self.W(P_k)
            
        for k in range(1, self.K + 1):
            P_k = A @ P_k # A^k X
            
            # Apply W_{l,k}
            if self.separate_W:
                proj = self.W[k](P_k)
            else:
                proj = self.W(P_k)
                
            H = H + alpha[k] * proj
            
        return H

class AGFL(nn.Module):
    def __init__(self, dim, heads, K, separate_W=True):
        super().__init__()
        self.heads = heads
        self.dim_h = dim // heads
        
        self.builders = nn.ModuleList([
            GraphConstructor(self.dim_h)
            for _ in range(heads)
        ])
        self.filters = nn.ModuleList([
            GraphFilter(self.dim_h, K, separate_W)
            for _ in range(heads)
        ])
        self.proj = nn.Linear(dim, dim)
        self.last_adj = None

    def forward(self, X, layer_idx, L):
        B, N, D = X.shape

        X = X.view(B, N, self.heads, self.dim_h).transpose(1, 2)
        

        sparsity = sparsity_schedule(layer_idx, L) 
        k_val = max(1, int((1 - sparsity) * N))
        
        outs = []
        adjs = []
        
        for h in range(self.heads):
            Xh = X[:, h]
            S = self.builders[h](Xh)
            
            if k_val < N:
                threshold = torch.topk(S, k_val, dim=-1).values[..., -1:]
                mask = S < threshold
                S = S.masked_fill(mask, float('-inf'))
            
            A_sparse = torch.softmax(S, dim=-1)
            
            outs.append(self.filters[h](A_sparse, Xh))
            adjs.append(A_sparse.detach())

        self.last_adj = torch.stack(adjs)
        
        out = torch.cat(outs, dim=-1)

        return self.proj(out)


class StandardAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.5):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=heads, 
            dropout=dropout,
            batch_first=True
        )
        self.last_attn = None

    def forward(self, x):
        out, attn_weights = self.attn(x, x, x, need_weights=True)
        self.last_attn = attn_weights.detach() 
        return out




#model (EEGnet)
class EEGNet(nn.Module):
    def __init__(self, num_channels=22, num_classes=2, samples=500, 
                 F1=8, D=2, F2=16, attention_type='standard', dropout_rate=0.5):
        super(EEGNet, self).__init__()
        self.attention_type = attention_type
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        if self.attention_type == 'agfl':
            self.attn_blocks = nn.ModuleList([AGFL(dim=F2, heads=4, K=2, separate_W=True)])
        elif self.attention_type == 'standard':
            self.attn_blocks = nn.ModuleList([StandardAttention(dim=F2, heads=4)])
        else:
            self.attn_blocks = None


        out_features = self._calculate_out_features(samples, num_channels)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_features, num_classes)
        )

    def _calculate_out_features(self, samples, channels):
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, channels, samples)
            x = self.block1(dummy_x)
            x = self.block2(x)
            x = self.block3(x)
            return x.numel()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        if self.attn_blocks is not None:
            x = x.squeeze(2).transpose(1, 2)
            
            L = len(self.attn_blocks)
            for i, block in enumerate(self.attn_blocks):
                if self.attention_type == 'agfl':
                    x = block(x, layer_idx=i, L=L)
                else:
                    x = block(x)
                        
        out = self.classifier(x)
        return out




def compute_class_weights(train_loader, device):
    all_labels = []
    for _, y in train_loader:
        all_labels.extend(y.numpy())
    
    all_labels = np.array(all_labels)
    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    num_classes = len(class_counts)
    
    weights = total_samples / (num_classes * class_counts)
    
    print(f"Class counts (0: Left, 1: Right, 2: Foot, 3: Tongue): {class_counts}")
    print(f"Applied weights: {weights}")
    
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_eval(model, train_loader, val_loader, device, epochs=100, lr=3e-4):
    model.to(device)
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3) 
    class_weights = compute_class_weights(train_loader, device)
    
    loss_fn = FocalLoss(weight=class_weights, gamma=3.0)
    

    warmup_epochs = 25
    warmup_scheduler = LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(opt, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(opt, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == 'cuda'))

    history = {'loss': [], 'val_acc': [], 'lr': []}
    
    pbar = tqdm.tqdm(range(epochs), desc="Training Model", unit="epoch")
    
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                    logits_val = model(x_val)
                    
                val_preds.extend(logits_val.argmax(-1).cpu().numpy())
                val_targets.extend(y_val.numpy())
                
        current_acc = accuracy_score(val_targets, val_preds)
        current_lr = opt.param_groups[0]['lr']
        
        history['loss'].append(avg_loss)
        history['val_acc'].append(current_acc)
        history['lr'].append(current_lr)

        pbar.set_postfix({
            "Loss": f"{avg_loss:.4f}", 
            "Val Acc": f"{current_acc:.4f}",
            "LR": f"{current_lr:.2e}"
        })

    model.eval()
    preds, targets, probs = [], [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                logits = model(x)
                
            preds.extend(logits.argmax(-1).cpu().numpy())
            targets.extend(y.numpy())
            
            probabilities = F.softmax(logits.float(), dim=-1)
            probs.extend(probabilities.cpu().numpy())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro') 
    
    probs = np.array(probs)
    if probs.shape[1] == 2:
        auc = roc_auc_score(targets, probs[:, 1])
    else:
        auc = roc_auc_score(targets, probs, multi_class='ovr')
    
    print(f"Final Results -> Accuracy: {acc:.4f} | F1 (Macro): {f1:.4f} | ROC-AUC: {auc:.4f}")

    return acc, f1, auc, history



def plot_training_curves(history, name, save_path):
    epochs = range(1, len(history['loss']) + 1)
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
    axs[0].plot(epochs, history['loss'], marker='o', color='#d62728', linewidth=2, label='Train Loss')
    axs[0].set_ylabel('Loss', fontsize=12)
    axs[0].set_title(f'{name} Training Trajectory', fontsize=14, fontweight='bold')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend()

    axs[1].plot(epochs, history['val_acc'], marker='o', color='#1f77b4', linewidth=2, label='Val Accuracy')
    axs[1].set_ylabel('Accuracy', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend()
    
    axs[2].plot(epochs, history['lr'], marker='o', color='#2ca02c', linewidth=2, label='Learning Rate')
    axs[2].set_xlabel('Epochs', fontsize=12)
    axs[2].set_ylabel('Learning Rate', fontsize=12)
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_curves_{name}.png", dpi=300)
    plt.close()


def plot_layer_heatmaps(models, x, save_path, save_prefix="heatmap"):
    for name, model in models.items():
        if model.attn_blocks is None:
            continue
            
        with torch.no_grad():
            _ = model(x)

        num_layers = len(model.attn_blocks)
        fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4))
        if num_layers == 1:
            axes = [axes]

        for i, layer in enumerate(model.attn_blocks):
            A = None
            if hasattr(layer, "last_adj") and layer.last_adj is not None:
                A = layer.last_adj
                if A.ndim == 4:
                    A = A.mean(0).mean(0).cpu().numpy()
                elif A.ndim == 3:
                    A = A.mean(0).cpu().numpy()
            elif hasattr(layer, "last_attn") and layer.last_attn is not None:
                A = layer.last_attn
                if A.ndim == 4:
                    A = A.mean(0).mean(0).cpu().numpy()
                elif A.ndim == 3:
                    A = A.mean(0).cpu().numpy()

            if A is not None and A.ndim == 2:
                im = axes[i].imshow(A, aspect='auto', cmap='magma') 
                axes[i].set_title(f"Window / Block {i}")
                axes[i].set_xlabel("Time Step (Pooled)")
                axes[i].set_ylabel("Time Step (Pooled)")
                fig.colorbar(im, ax=axes[i])

        plt.suptitle(f"{name} Temporal Connectivity Heatmaps")
        plt.tight_layout()
        plt.savefig(f"{save_path}/{save_prefix}_{name}.png")
        plt.close()


def plot_eeg_spatial_attention(x, models, save_path, save_prefix="temporal_attention"):
    x_sample = x[0].unsqueeze(0) 
    
    for name, model in models.items():
        if model.attn_blocks is None:
            continue
            
        device = next(model.parameters()).device
        x_dev = x_sample.to(device)
        
        with torch.no_grad():
            _ = model(x_dev)

        for i, layer in enumerate(model.attn_blocks):
            A = None
            
            if hasattr(layer, "last_adj") and layer.last_adj is not None:
                A = layer.last_adj
            elif hasattr(layer, "last_attn") and layer.last_attn is not None:
                A = layer.last_attn

            if A is not None:
                if A.ndim == 4:
                    A = A.mean(0).mean(0).cpu().numpy()
                elif A.ndim == 3:
                    A = A.mean(0).cpu().numpy()

            if A is not None and A.ndim == 2:
                fig, ax1 = plt.subplots(figsize=(8, 6))
                
                im = ax1.imshow(A, aspect='equal', cmap='plasma', alpha=0.9)
                
                seq_len = A.shape[0]
                ax1.set_xticks(np.arange(seq_len))
                ax1.set_yticks(np.arange(seq_len))
                
                ax1.set_xlabel("Target Time Step", fontsize=11)
                ax1.set_ylabel("Source Time Step", fontsize=11)
                
                cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
                cbar.set_label("Attention Weight", rotation=270, labelpad=15)
                
                plt.title(f"{name} - Block {i} Temporal Attention", fontsize=13, fontweight='bold')
                
                fig.tight_layout()
                plt.savefig(f"{save_path}/{save_prefix}_{name}_block{i}.png", dpi=300)
                plt.close(fig)


def plot_eeg_epoch(x, save_path, y=None, fs=250.0, channel_names=None):
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if torch.is_tensor(y):
        y = y.item()
        
    channels, time_steps = x.shape
    time = np.arange(time_steps) / fs
    fig, ax = plt.subplots(figsize=(12, 8))
    offset = 4.0 
    
    for i in range(channels):
        ax.plot(time, x[i, :] - (i * offset), color='#1f77b4', linewidth=0.8)
        
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Channels', fontsize=12)
    
    y_ticks = [-i * offset for i in range(channels)]
    ax.set_yticks(y_ticks)
    
    if channel_names and len(channel_names) == channels:
        ax.set_yticklabels(channel_names)
    else:
        ax.set_yticklabels([f'Ch {i+1}' for i in range(channels)])
        
    title = 'EEG Epoch Visualization'
    if y is not None:
        label_map = {0: 'Left Hand (Class 1)', 1: 'Right Hand (Class 2)'}
        title += f" - {label_map.get(y, f'Class {y}')}"
        
    ax.set_title(title, fontsize=14, pad=15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/epochs.png", dpi=300)
    plt.close(fig)



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "./out_attention6"
    
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_dir = "./ml"

    train_loader, val_loader = get_eeg_dataloaders(data_dir=data_dir, batch_size=64)

    x_batch, y_batch = next(iter(train_loader))
    x_single = x_batch[0]
    y_single = y_batch[0]
    plot_eeg_epoch(x_single, save_path, y_single, fs=250.0)

    configs = [
        ("AGFL", "agfl"),
        #("Standard", "standard"),
        #("None", "none")
    ]

    results = {}
    models = {}

    for name, mode in configs:
        print(f"--- Training {name} ---")
        
        model = EEGNet(
            attention_type=mode,
            num_classes=2,
            num_channels=22
        ).to(device)
        
        acc, f1, auc, history = train_eval(model, train_loader, val_loader, device, epochs=100)
        
        print(f"\n{name} Results:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 (Macro): {f1:.4f}")
        print(f"  ROC-AUC (OVR): {auc:.4f}\n")

        results[name] = {
            'Accuracy': acc,
            'F1_Score': f1,
            'ROC_AUC': auc
        }
        models[name] = model
        plot_training_curves(history, name, save_path)

    x_vis, y_vis = next(iter(val_loader))
    x_vis = x_vis.to(device)

    plot_layer_heatmaps(models, x_vis, save_path)
    plot_eeg_spatial_attention(x_vis, models, save_path) 

if __name__ == "__main__":
    main()



# AGFL Results:                                                                                  
#   Accuracy : 0.7438                                                                            
#   F1 (Macro): 0.7433                                                                           
#   ROC-AUC (OVR): 0.8407                                                                        
                                                                                               
                                                                                               
# Standard Results:                                                                              
#   Accuracy : 0.7307                                                                            
#   F1 (Macro): 0.7291                                                                           
#   ROC-AUC (OVR): 0.8276                                                                        
                                                                                                                  
                                                                                               
# None Results:                                                                                  
#   Accuracy : 0.7253                                                                            
#   F1 (Macro): 0.7229                                                                           
#   ROC-AUC (OVR): 0.8265 



#2 seconds epoch
#topographic / topomat - need power of one channel for the epoch - and I need channel locations - spectum map



#explain temporall attention heatmaps
#attention between different channels