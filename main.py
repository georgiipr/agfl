import os
import argparse
import torch
import numpy as np
import random
import warnings
from torch.utils.data import DataLoader
import mne
import os


from data_loaders.ECGDataset import ECGDataset
from models.conformer import ConformerModel
from trainers.ecg_trainer import train_eval_ecg
from plots.ecg_plots import (
    plot_training_curves, plot_comparison, plot_layer_heatmaps,
    plot_embedding_space, plot_ecg_attention, plot_sparsity_vs_accuracy,
    plot_entropy, plot_sparsity
)

from data_loaders.BCIDataset import BCI2aDataset, get_eeg_dataloaders
from models.eegnet import EEGNet
from models.eegencoder import EEGEncoder
from trainers.eeg_trainer import train_eval_eeg
from plots.eeg_plots import (
    plot_training_curves, plot_layer_heatmaps_u,
    plot_eeg_spatial_attention, plot_eeg_epoch, get_mne_info, plot_spectrum_map_mne, plot_topographic_map_mne
)

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.benchmark = True

BCI2A_CH_NAMES = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 
    'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
]

def run_ecg(device, PROJECT_ROOT):
    print("starting")
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "mit-bih-arrhythmia-database-1.0.0")
    save_path = os.path.join(current_dir, "agfl", "out_agfl_ecg")
    os.makedirs(save_path, exist_ok=True)

    train_records = [
        '101', '106', '108', '109', '112', '114', '115', '116', 
        '118', '119', '122', '124', '201', '203', '205', '207', 
        '208', '209', '215', '220', '223', '230'
    ] 
    
    val_records = [
        '100', '103', '104', '105', '111', '113', '117', '121', 
        '200', '202', '210', '212', '213', '214', '219', '221', 
        '222', '228', '231', '232', '233', '234'
    ]

    train_loader = DataLoader(ECGDataset(train_records, data_dir, is_train=True), batch_size=64, shuffle=True)
    val_loader = DataLoader(ECGDataset(val_records, data_dir), batch_size=64)

    configs = [
        ("AGFL", "agfl"),
        #("None", "none"),
        ("Standard", "standard"),
    ]

    results = {}
    models = {}

    for name, mode in configs:
        print(f"{name} is on\n")
        model = ConformerModel(mode=mode).to(device)
        
        acc, f1, auc, history = train_eval_ecg(model, train_loader, val_loader, device)
        
        print(f"{name}:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 (Macro): {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}\n")

        results[name] = {
            'Accuracy': acc,
            'F1_Score': f1,
            'ROC_AUC': auc
        }
        models[name] = model
        plot_training_curves(history, name, save_path)

    x_vis, y_vis = next(iter(val_loader))
    x_vis = x_vis.to(device)
    y_vis = y_vis.cpu().numpy()

    plot_comparison(results, save_path)
    plot_layer_heatmaps(models, x_vis, save_path)
    plot_embedding_space(models, x_vis, y_vis, save_path, method="tsne")
    plot_embedding_space(models, x_vis, y_vis, save_path, method="umap")
    plot_ecg_attention(x_vis, models, save_path)
    plot_sparsity_vs_accuracy(models, val_loader, device, save_path)
    plot_entropy(models, x_vis, save_path)
    plot_sparsity(models, x_vis, save_path)


def run_eeg(device, PROJECT_ROOT, num_classes_global):
    print("starting")
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "ml")
    save_path = os.path.join(current_dir, "agfl", "out_agfl_eeg")

    os.makedirs(save_path, exist_ok=True)

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
            num_classes=num_classes_global,
            num_channels=22
        ).to(device)
        
        acc, f1, auc, history = train_eval_eeg(model, train_loader, val_loader, device, epochs=100)
        
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


def main():
    parser = argparse.ArgumentParser(description="Run biological signal models.")
    parser.add_argument(
        '--task', 
        type=str, 
        choices=['ecg', 'eeg'], 
        required=True, 
        help="Specify which experiment to run: 'ecg' or 'eeg'"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    if args.task == 'ecg':
        run_ecg(device, PROJECT_ROOT)
    elif args.task == 'eeg':
        run_eeg(device, PROJECT_ROOT, num_classes_global=2)

if __name__ == "__main__":
    main()


#PYTHONPATH=. python3 main.py --task ecg