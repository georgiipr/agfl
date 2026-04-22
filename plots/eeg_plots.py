import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import accuracy_score
import mne
import os
import scipy.signal as signal


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
        fig, axes = plt.subplots(1, num_layers, figsize=(4*num_layers, 4))
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
                im = axes[i].imshow(A, aspect='auto', cmap='viridis') 
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
                fig, ax1 = plt.subplots(figsize=(6, 5))
                
                im = ax1.imshow(A, aspect='auto', cmap='viridis')
                
                ax1.set_xlabel("Target Time Step", fontsize=11)
                ax1.set_ylabel("Source Time Step", fontsize=11)
                
                cbar = fig.colorbar(im, ax=ax1)
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
        # UPDATED: 4 Classes
        label_map = {0: 'Left Hand', 1: 'Right Hand', 2: 'Foot', 3: 'Tongue'}
        title += f" - {label_map.get(y, f'Class {y}')}"
        
    ax.set_title(title, fontsize=14, pad=15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/epochs.png", dpi=300)
    plt.close(fig)


def plot_csp_patterns(X, y, ch_names, save_path, fs=250.0):
    if hasattr(X, 'cpu'):
        X = X.cpu().numpy()
    if hasattr(y, 'cpu'):
        y = y.cpu().numpy()
        
    target_chs = ['C3', 'Cz', 'C4', 'Pz']
    ch_indices = [ch_names.index(ch) for ch in target_chs if ch in ch_names]
    
    if len(ch_indices) == 4:
        X = X[:, ch_indices, :]
        ch_names = [ch_names[i] for i in ch_indices]
    else:
        print(f"Warning: Could not isolate {target_chs}. Ensure they are in ch_names.")
        
    n_ch = len(ch_names)

    # 1. MASK FOR BINARY CLASSIFICATION (Left Hand vs Right Hand)
    # MNE's plot_patterns cannot handle 4 classes natively.
    binary_mask = (y == 0) | (y == 1)
    X_bin = X[binary_mask]
    y_bin = y[binary_mask]
    
    # Safety check: Ensure the current batch has both classes
    if len(np.unique(y_bin)) < 2:
        print("Warning: Not enough classes in this batch to plot CSP (need 0 and 1). Skipping.")
        return

    if X_bin.ndim == 4:
        print(f"Error: CSP failed. Your data 'X' has shape {X_bin.shape}.")
        return
    elif X_bin.ndim == 3:
        if X_bin.shape[1] != n_ch:
            if X_bin.shape[2] == n_ch:
                X_bin = np.swapaxes(X_bin, 1, 2)
            else:
                print(f"Error: Channel names mismatch.")
                return
    else:
        return

    info = get_mne_info(ch_names, fs)
    epochs = mne.EpochsArray(X_bin, info, verbose=False)
    
    try:
        epochs.set_montage('standard_1020')
    except ValueError as e:
        print(f"Warning: Could not set standard_1020 montage. Error: {e}")
    
    # 2. Fit the binary CSP
    csp = mne.decoding.CSP(n_components=4, reg=None, log=True, norm_trace=False)
    
    try:
        csp.fit(epochs.get_data(copy=False), y_bin)
        
        # 3. Plot patterns (Will now work because it's a binary fit)
        fig = csp.plot_patterns(epochs.info, ch_type='eeg', show=False, size=1.5)
        
        fig.suptitle('Common Spatial Patterns (Left vs Right Hand)', fontsize=14, y=1.05)
        fig.savefig(os.path.join(save_path, "csp_patterns.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    except ValueError as e:
        print(f"Failed to compute or plot CSP: {e}")

def plot_c3_c4_stft(X, y, ch_names, save_path, fs=250.0):
    if 'C3' not in ch_names or 'C4' not in ch_names:
        print("C3 or C4 not found in channels. Skipping STFT plot.")
        return

    c3_idx = ch_names.index('C3')
    c4_idx = ch_names.index('C4')
    
    class_labels = ['Left Hand', 'Right Hand', 'Foot', 'Tongue']

    def get_avg_stft(data_indices, ch_idx):
        if len(data_indices) == 0:
            return None, None, None
            
        Sxx_list = []
        for idx in data_indices:
            f, t, Sxx = signal.spectrogram(X[idx, ch_idx, :], fs=fs, nperseg=128, noverlap=112)
            Sxx_list.append(Sxx)
        return f, t, np.mean(Sxx_list, axis=0)

    fig, axs = plt.subplots(4, 2, figsize=(10, 12), sharex=True, sharey=True)

    all_sxx = []
    plot_data = []

    for i in range(4):
        class_idx = np.where(y == i)[0]
        f, t, sxx_c3 = get_avg_stft(class_idx, c3_idx)
        f, t, sxx_c4 = get_avg_stft(class_idx, c4_idx)
        
        if f is not None:
            s3_db = 10 * np.log10(sxx_c3 + 1e-10)
            s4_db = 10 * np.log10(sxx_c4 + 1e-10)
            all_sxx.extend([s3_db, s4_db])
            plot_data.append((s3_db, s4_db))
        else:
            plot_data.append((None, None))

    if not all_sxx:
        return

    freq_mask = f <= 40
    f_plot = f[freq_mask]
    vmax = max([s[freq_mask, :].max() for s in all_sxx])
    vmin = vmax - 30

    im = None
    for i in range(4):
        s3, s4 = plot_data[i]
        
        if s3 is not None:
            im = axs[i, 0].pcolormesh(t, f_plot, s3[freq_mask, :], shading='gouraud', cmap='viridis', vmax=vmax, vmin=vmin)
            axs[i, 0].set_ylabel('Freq (Hz)')
            if i == 0: axs[i, 0].set_title('C3 (Left Motor Cortex)')
            
            axs[i, 1].pcolormesh(t, f_plot, s4[freq_mask, :], shading='gouraud', cmap='viridis', vmax=vmax, vmin=vmin)
            if i == 0: axs[i, 1].set_title('C4 (Right Motor Cortex)')
            axs[i, 1].text(1.05, 0.5, class_labels[i], transform=axs[i, 1].transAxes, fontsize=12, va='center', rotation=-90, fontweight='bold')
        else:
            axs[i, 0].text(0.5, 0.5, 'No Data', ha='center', va='center')
            axs[i, 1].text(0.5, 0.5, 'No Data', ha='center', va='center')

    axs[3, 0].set_xlabel('Time (s)')
    axs[3, 1].set_xlabel('Time (s)')

    if im:
        cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.03, pad=0.08)
        cbar.set_label('Power Spectral Density (dB)') 
    
    plt.suptitle('Time-Frequency Response (STFT) - Watch for ERD (8-30 Hz)', fontsize=14, fontweight='bold', y=0.92)
    plt.savefig(os.path.join(save_path, "stft_c3_c4.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)