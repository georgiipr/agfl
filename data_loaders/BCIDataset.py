import os
import torch
import numpy as np
import mne
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, random_split, DataLoader
import warnings

def bandpass_eeg_signal(data, fs=250.0, lowcut=2.0, highcut=30.0):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def standard_delta_modulation(signal, threshold=0.2):
    channels, time_steps = signal.shape
    spikes_up = np.zeros_like(signal)
    spikes_down = np.zeros_like(signal)
    
    ref = signal[:, 0].copy()
    
    for t in range(1, time_steps):
        diff = signal[:, t] - ref
        
        up_mask = diff >= threshold
        spikes_up[up_mask, t] = 1.0
        ref[up_mask] += threshold
        
        down_mask = diff <= -threshold
        spikes_down[down_mask, t] = 1.0
        ref[down_mask] -= threshold
        
    return np.concatenate([spikes_up, spikes_down], axis=0)

class BCI2aDataset(Dataset):
    def __init__(self, data_dir, subjects, is_train=True, is_snn=False, threshold=0.5):
        self.samples = []
        self.raw_samples = []  # Added to track unspiked signals for visualization
        self.labels = []
        self.is_train = is_train
        self.is_snn = is_snn
        self.threshold = threshold
        suffix = 'T' if is_train else 'E'
        
        for subj in subjects:
            filepath = os.path.join(data_dir, f"A{subj:02d}{suffix}.gdf")
            if not os.path.exists(filepath):
                print(f"Warning: {filepath} not found. Skipping.")
                continue
                
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                raw = mne.io.read_raw_gdf(filepath, preload=True, verbose='ERROR')
            
            X_cont = raw.get_data()[:22, :].astype(np.float32)
            X_cont = bandpass_eeg_signal(X_cont, fs=250.0)
            session_mean = X_cont.mean(axis=-1, keepdims=True)
            session_std = X_cont.std(axis=-1, keepdims=True) + 1e-8
            X_cont = (X_cont - session_mean) / session_std
            
            events, event_dict = mne.events_from_annotations(raw, verbose=False)
            inv_event_dict = {v: k for k, v in event_dict.items()}
            
            offset = int(0.5 * 250)
            max_window_size = int(3.5 * 250)
            
            for event in events:
                start_idx = event[0]
                event_id = event[2]
                event_code = inv_event_dict.get(event_id, '')
                
                if event_code == '769':
                    label = 0
                elif event_code == '770':
                    label = 1
                elif event_code == '771':
                    label = 2
                elif event_code == '772':
                    label = 3
                else:
                    continue 
                    
                actual_start = start_idx + offset
                
                if actual_start + max_window_size > X_cont.shape[1]:
                    continue
                    
                trial_data = X_cont[:, actual_start : actual_start + max_window_size]
                
                if self.is_snn:
                    spiked_data = standard_delta_modulation(trial_data, threshold=self.threshold)
                    self.samples.append(spiked_data)
                    self.raw_samples.append(trial_data)
                else:
                    self.samples.append(trial_data)
                
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.samples[idx]).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        x = x[:, :1000]
            
        if self.is_snn:
            x_raw = torch.from_numpy(self.raw_samples[idx]).float()
            x_raw = x_raw[:, :1000]
            return x, y, x_raw  # Returns the raw tuple pack
            
        return x, y

def get_eeg_dataloaders(data_dir="./ml", subject_id=1, batch_size=64, is_snn=False, threshold=0.5):
    full_dataset = BCI2aDataset(
        data_dir, 
        subjects=[subject_id], 
        is_train=True, 
        is_snn=is_snn,
        threshold=threshold
    )
    
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader