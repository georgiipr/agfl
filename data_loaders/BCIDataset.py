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

def delta_sigma_encode(signal, delta=0.3):
    """
    Converts continuous EEG into a dual-coded spike tensor using 
    Hybrid Delta-Sigma (change-based) and Stochastic Rate Modulation.
    Inputs:
        signal: numpy array of shape (Channels, Time)
        delta: float, step parameter for the Delta-Sigma modulation (default: 0.3 as per paper)
    Outputs:
        spikes: numpy array of shape (Channels * 2, Time)
    """
    channels, time_steps = signal.shape
    
    # 1. Delta-Sigma modulation
    s_ds = np.zeros((channels, time_steps), dtype=np.float32)
    r = np.zeros(channels, dtype=np.float32)  # Initialize adaptive reference
    
    for t in range(time_steps):
        x_t = signal[:, t]
        # Spike occurs if current signal > adaptive reference
        s_t = (x_t > r).astype(np.float32)
        s_ds[:, t] = s_t
        # Update adaptive reference: r(t+1) = r(t) + s(t) - delta
        r = r + s_t - delta
        
    # 2. Stochastic Rate Coding
    # Calculate probability: p(t) = 1 / (1 + exp(-x(t)))
    p_t = 1.0 / (1.0 + np.exp(-signal))
    
    # Generate spikes based on Bernoulli distribution: b(t) ~ Bernoulli(p(t))
    s_rate = (np.random.rand(channels, time_steps) < p_t).astype(np.float32)
    
    # 3. Concatenate both representations along the channel axis
    spikes = np.concatenate([s_ds, s_rate], axis=0)
    
    return spikes

class BCI2aDataset(Dataset):
    def __init__(self, data_dir, subjects, is_train=True, is_snn=False, delta=0.3):
        self.samples = []
        self.labels = []
        self.is_train = is_train
        self.is_snn = is_snn
        self.delta = delta
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
            
            events, event_dict = mne.events_from_annotations(raw, verbose=False)
            inv_event_dict = {v: k for k, v in event_dict.items()}
            
            offset = int(0.5 * 250)
            max_window_size = int(4.0 * 250)
            
            for event in events:
                start_idx = event[0]
                event_id = event[2]
                event_code = inv_event_dict[event_id]
                
                if event_code == '769':
                    label = 0  # Left Hand
                elif event_code == '770':
                    label = 1  # Right Hand
                elif event_code == '771':
                    label = 2  # Foot
                elif event_code == '772':
                    label = 3  # Tongue
                else:
                    continue 
                    
                actual_start = start_idx + offset
                
                if actual_start + max_window_size > X_cont.shape[1]:
                    continue
                    
                trial_data = X_cont[:, actual_start : actual_start + max_window_size]
                
                t_mean = trial_data.mean(axis=-1, keepdims=True)
                t_std = trial_data.std(axis=-1, keepdims=True) + 1e-8
                trial_data = (trial_data - t_mean) / t_std
                
                if self.is_snn:
                    trial_data = delta_sigma_encode(trial_data, delta=self.delta)
                
                self.samples.append(trial_data)
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.samples[idx]).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
        x = x[:, :1000]
            
        return x, y

def get_eeg_dataloaders(data_dir="./ml", subject_id=1, batch_size=64, is_snn=False, delta=0.3):
    full_dataset = BCI2aDataset(
        data_dir, 
        subjects=[subject_id], 
        is_train=True, 
        is_snn=is_snn,
        delta=delta
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