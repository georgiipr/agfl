import os
from torch.utils.data import DataLoader
from datetime import datetime

from data_loaders.ECGDataset import ECGDataset
from models.conformer import ConformerModel
from trainers.ecg_trainer import train_eval_ecg
from plots.ecg_plots import (
    plot_training_curves, plot_comparison, plot_layer_heatmaps,
    plot_embedding_space, plot_ecg_attention, plot_sparsity_vs_accuracy,
    plot_entropy, plot_sparsity
)

def run_ecg(device, PROJECT_ROOT, model_name, agfl_status):
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "mit-bih-arrhythmia-database-1.0.0")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(current_dir, "agfl", "out_agfl_ecg", timestamp)
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

    mode = "agfl" if agfl_status == "on" else "standard"
    display_name = f"{model_name.capitalize()} ({mode.upper()})"

    results = {}
    models = {}
    
    model = ConformerModel(mode=mode).to(device)
    
    acc, f1, auc, history = train_eval_ecg(model, train_loader, val_loader, device)
    
    print(f"{display_name} Results:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 (Macro): {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}\n")

    results[display_name] = {
        'Accuracy': acc,
        'F1_Score': f1,
        'ROC_AUC': auc
    }
    models[display_name] = model
    plot_training_curves(history, display_name, save_path)

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