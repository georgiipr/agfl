import os
import copy
from datetime import datetime
import numpy as np
import torch

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
        attention_type=mode, num_classes=cls, num_channels=ch, K=1
    )
}

def run_eeg(device, PROJECT_ROOT, model_name, agfl_status, num_classes_global):
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
    
    model_factory = MODEL_REGISTRY[model_key]

    # --- MEMORY STORAGE FOR MULTI-PHASE TRAINING ---
    dataloaders = {}       # Maps subject_id -> (train_loader, val_loader)
    models = {}            # Maps subject_id -> trained_model
    initial_metrics = {}   # Maps subject_id -> dict of acc, f1, auc, history

    print(f"  Starting Two-Phase Subject-Dependent Pipeline ")
    print(f"  Model: {display_name}")

    # ========================================================
    # PHASE 1: INITIAL TRAINING FOR ALL SUBJECTS
    # ========================================================
    print("\n" + "="*50)
    print(" PHASE 1: INITIAL TRAINING (ALL SUBJECTS)")
    print("="*50)
    
    for subject_id in range(1, 10):
        print(f"\n>>> Initial Training Subject {subject_id}/9")
        
        # Load and store data
        train_loader, val_loader = get_eeg_dataloaders(data_dir=data_dir, subject_id=subject_id, batch_size=64)
        dataloaders[subject_id] = (train_loader, val_loader)

        # Initialize and train model
        model = model_factory(mode, num_classes_global, 22).to(device)
        acc, f1, auc, history = train_eval_eeg(model, train_loader, val_loader, device, epochs=250)
        
        # Save states
        models[subject_id] = model
        initial_metrics[subject_id] = {'acc': acc, 'f1': f1, 'auc': auc, 'history': history}

        print(f"Subject {subject_id} Initial Acc: {acc:.4f} | F1: {f1:.4f}")

    # ========================================================
    # PHASE 2: TRIAGE (IDENTIFY DONORS AND RECIPIENTS)
    # ========================================================
    print("\n" + "="*50)
    print(" PHASE 2: EVALUATION & TRIAGE")
    print("="*50)

    # Calculate median accuracy to separate Good and Bad models
    accuracies = [initial_metrics[sid]['acc'] for sid in range(1, 10)]
    threshold = np.median(accuracies)

    good_subjects = [sid for sid in range(1, 10) if initial_metrics[sid]['acc'] >= threshold]
    bad_subjects = [sid for sid in range(1, 10) if initial_metrics[sid]['acc'] < threshold]

    print(f"Median Accuracy Threshold: {threshold:.4f}")
    print(f"Donor Subjects (Good): {good_subjects}")
    print(f"Recipient Subjects (Bad): {bad_subjects}")

    final_metrics = copy.deepcopy(initial_metrics)

    # ========================================================
    # PHASE 3 & 4: KNOWLEDGE TRANSFER & FINE-TUNING
    # ========================================================
    if len(good_subjects) > 0 and len(bad_subjects) > 0:
        print("\n" + "="*50)
        print(" PHASE 3 & 4: KNOWLEDGE TRANSFER & FINE-TUNING")
        print("="*50)
        
        # Create Average Donor State Dict
        donor_state = {}
        for key in models[good_subjects[0]].state_dict():
            stacked = torch.stack([models[sid].state_dict()[key] for sid in good_subjects])
            if stacked.is_floating_point():
                donor_state[key] = stacked.mean(dim=0)
            else:
                donor_state[key] = stacked.float().mean(dim=0).long()

        # We will try different blend ratios to find the best starting point
        alpha_options = [0.25, 0.50, 0.75] 
        
        for bad_sid in bad_subjects:
            print(f"\n>>> Fine-Tuning Subject {bad_sid} with Knowledge Transfer...")
            
            original_model = models[bad_sid]
            original_state = copy.deepcopy(original_model.state_dict())
            initial_acc = initial_metrics[bad_sid]['acc']
            
            # Track the best model across attempts. Initialize with Phase 1 baseline.
            best_state = copy.deepcopy(original_state)
            best_acc = initial_acc
            best_metrics = copy.deepcopy(initial_metrics[bad_sid])
            
            train_loader, val_loader = dataloaders[bad_sid]

            for attempt, alpha in enumerate(alpha_options, 1):
                print(f"  -> Attempt {attempt}/{len(alpha_options)} (Blend Alpha: {alpha})")
                
                new_state = {}
                for key in original_state:
                    if "running" in key or "tracked" in key or "num_batches" in key:
                        new_state[key] = original_state[key]
                    elif original_state[key].is_floating_point():
                        new_state[key] = (alpha * donor_state[key]) + ((1 - alpha) * original_state[key])
                    else:
                        new_state[key] = donor_state[key]
                        
                # Load the blended weights into the model
                original_model.load_state_dict(new_state)

                # Fine-tune
                acc, f1, auc, history = train_eval_eeg(
                    original_model, train_loader, val_loader, device, epochs=50, lr=1e-4
                )

                print(f"     Attempt {attempt} Acc: {acc:.4f} (Baseline: {initial_acc:.4f})")

                # If this attempt is the best we've seen, save it
                if acc > best_acc:
                    print(f"     *** New Best for Subject {bad_sid}! ({best_acc:.4f} -> {acc:.4f}) ***")
                    best_acc = acc
                    best_state = copy.deepcopy(original_model.state_dict())
                    best_metrics = {'acc': acc, 'f1': f1, 'auc': auc, 'history': history}

            # --- Final Fallback Check ---
            if best_acc > initial_acc:
                print(f"  [SUCCESS] Subject {bad_sid} improved from {initial_acc:.4f} to {best_acc:.4f}")
            else:
                print(f"  [REVERTED] Transfer hurt Subject {bad_sid}. Reverting to original {initial_acc:.4f}")
                
            # Apply the absolute best state we found (even if it's the original)
            original_model.load_state_dict(best_state)
            final_metrics[bad_sid] = best_metrics
            models[bad_sid] = original_model



    all_final_acc, all_final_f1, all_final_auc = [], [], []

    for subject_id in range(1, 10):
        print(f"Generating artifacts for Subject {subject_id}...")
        subj_save_path = os.path.join(base_save_path, f"Subject_{subject_id}")
        os.makedirs(subj_save_path, exist_ok=True)

        model = models[subject_id]
        train_loader, val_loader = dataloaders[subject_id]
        metrics = final_metrics[subject_id]

        all_final_acc.append(metrics['acc'])
        all_final_f1.append(metrics['f1'])
        all_final_auc.append(metrics['auc'])

        models_dict = {display_name: model}
        
        # Extract evaluation logic exactly as you had it initially
        plot_training_curves(metrics['history'], display_name, subj_save_path)

        x_vis, y_vis = next(iter(val_loader))
        x_vis_dev = x_vis.to(device)

        plot_layer_heatmaps_u(models_dict, x_vis_dev, subj_save_path)
        plot_eeg_spatial_attention(x_vis_dev, models_dict, subj_save_path) 

        x_single = x_vis[0]
        y_single = y_vis[0]
        plot_eeg_epoch(x_single, subj_save_path, y_single, fs=250.0)

        numpy_epoch = x_single.cpu().numpy()
        
        plot_spectrum_map_mne(
            epoch_data=numpy_epoch, 
            ch_names=BCI2A_CH_NAMES, 
            save_path=subj_save_path, 
            fs=250.0
        )
        
        plot_topographic_map_mne(
            epoch_data=numpy_epoch, 
            ch_names=BCI2A_CH_NAMES, 
            save_path=subj_save_path, 
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
            save_path=subj_save_path, 
            fs=250.0
        )

        plot_c3_c4_stft(
            X=X_multitrial, 
            y=Y_multitrial, 
            ch_names=BCI2A_CH_NAMES, 
            save_path=subj_save_path, 
            fs=250.0
        )

    avg_acc = np.mean(all_final_acc)
    avg_f1 = np.mean(all_final_f1)
    avg_auc = np.mean(all_final_auc)

    print(f"\nFinal averaged results (Post-Transfer Phase):")
    print(f"  Average Accuracy : {avg_acc:.4f}")
    print(f"  Average F1 Score : {avg_f1:.4f}")
    print(f"  Average ROC-AUC  : {avg_auc:.4f}")