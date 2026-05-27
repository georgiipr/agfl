import os
from datetime import datetime
import numpy as np
import torch

from data_loaders.BCIDataset import get_eeg_dataloaders
from models.eegnet import EEGNet
from models.snn import SpikingEEGNet
from trainers.eeg_trainer import train_eval_eeg
from plots.eeg_plots import (
    plot_training_curves, plot_layer_heatmaps_u,
    plot_eeg_spatial_attention, plot_eeg_epoch, get_mne_info, 
    plot_spectrum_map_mne, plot_topographic_map_mne,
    plot_csp_patterns, plot_c3_c4_stft, plot_raw_vs_spikes
)

BCI2A_CH_NAMES = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 
    'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
]

MODEL_REGISTRY = {
    "eegnet": lambda mode, cls, ch: EEGNet(
        attention_type=mode, num_classes=cls, num_channels=ch
    ),
    "snn": lambda mode, cls, ch: SpikingEEGNet(
        attention_type=mode, num_classes=cls, num_channels=ch
    )
}

def run_eeg(device, PROJECT_ROOT, model_name, agfl_status, num_classes_global, task_name="eeg"):
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "ml")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_save_path = os.path.join(current_dir, "agfl", "out_agfl_eeg", timestamp)
    os.makedirs(base_save_path, exist_ok=True)

    mode = "agfl" if agfl_status == "on" else "standard" if agfl_status == "off" else "none"
    display_name = f"{model_name.capitalize()} ({mode.upper()})"
    
    model_key = model_name.lower().replace("_", "").replace("-", "")
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    
    is_snn = (task_name == "nc" or model_key == "spikingeegnet" or model_key == "snn")
    
    model_factory = MODEL_REGISTRY[model_key]

    all_acc, all_f1, all_auc = [], [], []

    print(f"  Starting Subject-Dependent Pipeline ")
    print(f"  Task : {task_name.upper()} | SNN Mode: {is_snn}")
    print(f"  Model: {display_name}")

    for subject_id in range(1, 10):
        print(f"\n Processing {subject_id}/9")
        
        subj_save_path = os.path.join(base_save_path, f"Subject_{subject_id}")
        os.makedirs(subj_save_path, exist_ok=True)

        train_loader, val_loader = get_eeg_dataloaders(
            data_dir=data_dir, 
            subject_id=subject_id, 
            batch_size=64,
            is_snn=is_snn
        )

        in_channels = 44 if is_snn else 22
        model = model_factory(mode, num_classes_global, in_channels).to(device)
        
        acc, f1, auc, history = train_eval_eeg(
            model, train_loader, val_loader, device, epochs=250, is_snn=is_snn
        )
        
        all_acc.append(acc)
        all_f1.append(f1)
        all_auc.append(auc)

        print(f"Subject {subject_id} Results:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 (Macro): {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")

        models_dict = {display_name: model}
        plot_training_curves(history, display_name, subj_save_path)

        batch_vis = next(iter(val_loader))
        if len(batch_vis) == 3:
            x_vis, y_vis, x_raw_vis = batch_vis
        else:
            x_vis, y_vis = batch_vis
            x_raw_vis = x_vis

        x_vis_dev = x_vis.to(device)
        plot_layer_heatmaps_u(models_dict, x_vis_dev, subj_save_path)
        plot_eeg_spatial_attention(x_vis_dev, models_dict, subj_save_path) 

        x_single = x_vis[0]
        y_single = y_vis[0]
        x_single_raw = x_raw_vis[0]

        plot_eeg_epoch(x_single_raw, subj_save_path, y_single, fs=250.0, channel_names=BCI2A_CH_NAMES)

        numpy_raw_epoch = x_single_raw.cpu().numpy()
        
        plot_spectrum_map_mne(
            epoch_data=numpy_raw_epoch, 
            ch_names=BCI2A_CH_NAMES, 
            save_path=subj_save_path, 
            fs=250.0
        )
        
        plot_topographic_map_mne(
            epoch_data=numpy_raw_epoch, 
            ch_names=BCI2A_CH_NAMES, 
            save_path=subj_save_path, 
            fs=250.0, 
            title=f"Epoch Power Topomap (Class {y_single.item()})"
        )

        all_x_raw, all_y = [], []
        for i, batch_b in enumerate(val_loader):
            if len(batch_b) == 3:
                _, yb, xrb = batch_b
                all_x_raw.append(xrb.cpu().numpy())
            else:
                xb, yb = batch_b
                all_x_raw.append(xb.cpu().numpy())
            all_y.append(yb.cpu().numpy())
            if i >= 3:
                break
                
        X_multitrial_raw = np.concatenate(all_x_raw, axis=0)
        Y_multitrial = np.concatenate(all_y, axis=0)

        plot_csp_patterns(
            X=X_multitrial_raw, 
            y=Y_multitrial, 
            ch_names=BCI2A_CH_NAMES, 
            save_path=subj_save_path, 
            fs=250.0
        )

        plot_c3_c4_stft(
            X=X_multitrial_raw, 
            y=Y_multitrial, 
            ch_names=BCI2A_CH_NAMES, 
            save_path=subj_save_path, 
            fs=250.0
        )

        if is_snn:
            plot_raw_vs_spikes(x_single_raw, x_single, BCI2A_CH_NAMES, 'C3', subj_save_path, fs=250.0)
            plot_raw_vs_spikes(x_single_raw, x_single, BCI2A_CH_NAMES, 'C4', subj_save_path, fs=250.0)

    avg_acc = np.mean(all_acc)
    avg_f1 = np.mean(all_f1)
    avg_auc = np.mean(all_auc)

    print(f"\nFinal averaged results:")
    print(f"  Average Accuracy : {avg_acc:.4f}")
    print(f"  Average F1 Score : {avg_f1:.4f}")
    print(f"  Average ROC-AUC  : {avg_auc:.4f}")