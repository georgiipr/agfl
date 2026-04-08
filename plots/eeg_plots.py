import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import accuracy_score
import mne
import os

def get_mne_info(ch_names, fs=250.0):
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1005')
    info.set_montage(montage)
    return info

def plot_spectrum_map_mne(epoch_data, ch_names, save_path, fs=250.0):
    info = get_mne_info(ch_names, fs)
    epochs = mne.EpochsArray(epoch_data[np.newaxis, :, :], info, verbose=False)
    
    spectrum = epochs.compute_psd(method='welch', fmin=1.0, fmax=40.0, verbose=False)
    fig = spectrum.plot(show=False, spatial_colors=True) 
    
    fig.savefig(os.path.join(save_path, "spectrum_map_mne.png"), dpi=300)
    plt.close(fig)

def plot_topographic_map_mne(epoch_data, ch_names, save_path, fs=250.0, title="Epoch Power Topomap"):
    info = get_mne_info(ch_names, fs)
    epochs = mne.EpochsArray(epoch_data[np.newaxis, :, :], info, verbose=False)
    
    spectrum = epochs.compute_psd(method='welch', fmin=1.0, fmax=40.0, verbose=False)
    
    power_per_channel = np.mean(spectrum.get_data()[0], axis=-1)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    mne.viz.plot_topomap(
        power_per_channel, 
        info, 
        axes=ax, 
        show=False, 
        cmap='jet', 
        contours=6,
        extrapolate='box'
    )
    
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "topographic_map_mne.png"), dpi=300)
    plt.close(fig)

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


def plot_layer_heatmaps_u(models, x, save_path, save_prefix="heatmap"):
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