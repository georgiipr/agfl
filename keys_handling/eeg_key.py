import os
from datetime import datetime
import numpy as np

from data_loaders.BCIDataset import get_eeg_dataloaders
from models.eegnet import EEGNet
from models.eegencoder import EEGEncoder
from models.dsts_eeg_encoder import DSTSEEGEncoder
from trainers.eeg_trainer import train_eval_eeg
from plots.eeg_plots import (
    plot_training_curves, plot_layer_heatmaps_u,
    plot_eeg_spatial_attention, plot_eeg_epoch, get_mne_info, 
    plot_spectrum_map_mne, plot_topographic_map_mne,
    plot_csp_patterns, plot_c3_c4_stft
)

BCI2A_CH_NAMES = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 
    'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
]

MODEL_REGISTRY = {
    "eegnet": lambda mode, cls, ch: EEGNet(
        attention_type=mode, num_classes=cls, num_channels=ch
    ),
    "eegencoder": lambda mode, cls, ch: EEGEncoder(
        attention_type=mode, n_classes=cls, in_chans=ch
    ),
    "dstseegencoder": lambda mode, cls, ch: DSTSEEGEncoder(
        attention_type=mode, num_classes=cls, num_channels=ch, K=3
    )
}

def run_eeg(device, PROJECT_ROOT, model_name, agfl_status, num_classes_global):
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "ml")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(current_dir, "agfl", "out_agfl_eeg", timestamp)
    os.makedirs(save_path, exist_ok=True)

    train_loader, val_loader = get_eeg_dataloaders(data_dir=data_dir, batch_size=64)

    x_batch, y_batch = next(iter(train_loader))
    x_single = x_batch[0]
    y_single = y_batch[0]
    plot_eeg_epoch(x_single, save_path, y_single, fs=250.0)

    mode = "agfl" if agfl_status == "on" else "standard"
    display_name = f"{model_name.capitalize()} ({mode.upper()})"

    results = {}
    models = {}
    
    model_key = model_name.lower().replace("_", "").replace("-", "")
    
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_factory = MODEL_REGISTRY[model_key]
    model = model_factory(mode, num_classes_global, 22).to(device)
    
    acc, f1, auc, history = train_eval_eeg(model, train_loader, val_loader, device, epochs=100)
    
    print(f"\n{display_name} Results:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 (Macro): {f1:.4f}")
    print(f"  ROC-AUC (OVR): {auc:.4f}\n")

    results[display_name] = {
        'Accuracy': acc,
        'F1_Score': f1,
        'ROC_AUC': auc
    }
    models[display_name] = model
    plot_training_curves(history, display_name, save_path)

    x_vis, y_vis = next(iter(val_loader))
    x_vis = x_vis.to(device)

    plot_layer_heatmaps_u(models, x_vis, save_path)
    plot_eeg_spatial_attention(x_vis, models, save_path) 

    numpy_epoch = x_single.cpu().numpy()
    
    plot_spectrum_map_mne(
        epoch_data=numpy_epoch, 
        ch_names=BCI2A_CH_NAMES, 
        save_path=save_path, 
        fs=250.0
    )
    
    plot_topographic_map_mne(
        epoch_data=numpy_epoch, 
        ch_names=BCI2A_CH_NAMES, 
        save_path=save_path, 
        fs=250.0, 
        title=f"Epoch Power Topomap (Class {y_single.item()})"
    )

    all_x, all_y = [], []
    for i, (xb, yb) in enumerate(val_loader):
        all_x.append(xb.cpu().numpy())
        all_y.append(yb.cpu().numpy())
        if i >= 3:
            break
            
    X_multitrial = np.concatenate(all_x, axis=0)
    Y_multitrial = np.concatenate(all_y, axis=0)

    plot_csp_patterns(
        X=X_multitrial, 
        y=Y_multitrial, 
        ch_names=BCI2A_CH_NAMES, 
        save_path=save_path, 
        fs=250.0
    )

    plot_c3_c4_stft(
        X=X_multitrial, 
        y=Y_multitrial, 
        ch_names=BCI2A_CH_NAMES, 
        save_path=save_path, 
        fs=250.0
    )