import os
import torch
import mne

from data_loaders.BCIDataset import BCI2aDataset, get_eeg_dataloaders
from models.eegnet import EEGNet
from models.eegencoder import EEGEncoder
from models.dsts_eeg_encoder import DSTSEEGEncoder
from trainers.eeg_trainer import train_eval_eeg
from plots.eeg_plots import (
    plot_training_curves, plot_layer_heatmaps_u,
    plot_eeg_spatial_attention, plot_eeg_epoch, get_mne_info, 
    plot_spectrum_map_mne, plot_topographic_map_mne
)

BCI2A_CH_NAMES = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 
    'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
]

def run_eeg(device, PROJECT_ROOT, model_name, agfl_status, num_classes_global):
    print("Starting EEG Runner...")
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "ml")
    save_path = os.path.join(current_dir, "agfl", "out_agfl_eeg")

    os.makedirs(save_path, exist_ok=True)

    train_loader, val_loader = get_eeg_dataloaders(data_dir=data_dir, batch_size=64)

    x_batch, y_batch = next(iter(train_loader))
    x_single = x_batch[0]
    y_single = y_batch[0]
    plot_eeg_epoch(x_single, save_path, y_single, fs=250.0)

    # Determine mode based on AGFL flag
    mode = "agfl" if agfl_status == "on" else "standard"
    display_name = f"{model_name.capitalize()} ({mode.upper()})"

    results = {}
    models = {}

    print(f"--- Training {display_name} ---")
    
    # Expand here if `--model` calls for EEGEncoder vs EEGNet
    model = EEGNet(
        attention_type=mode,
        num_classes=num_classes_global,
        num_channels=22
    ).to(device)
    
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