import os
import torch
import torch.nn.functional as F
import numpy as np
import mne
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import warnings

def bandpass_eeg_signal(data, fs=250.0, lowcut=2.0, highcut=30.0):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

class BCI2aDataset(Dataset):
    def __init__(self, data_dir, subjects, is_train=True):
        self.samples = []
        self.labels = []
        self.is_train = is_train
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
            
            run_mean = X_cont.mean(axis=-1, keepdims=True)
            run_std = X_cont.std(axis=-1, keepdims=True) + 1e-8
            X_cont = (X_cont - run_mean) / run_std
            
            events, event_dict = mne.events_from_annotations(raw, verbose=False)
            inv_event_dict = {v: k for k, v in event_dict.items()}
            
            offset = int(0.5 * 250)
            max_window_size = int(4.0 * 250)
            
            for event in events:
                start_idx = event[0]
                event_id = event[2]
                event_code = inv_event_dict[event_id]
                if event_code == '769':
                    raw_label = 1
                elif event_code == '770':
                    raw_label = 2
                else:
                    continue 
                    
                actual_start = start_idx + offset
                if actual_start + max_window_size > X_cont.shape[1]:
                    continue
                    
                trial_data = X_cont[:, actual_start : actual_start + max_window_size]
                
                label = int(raw_label) - 1 
                self.samples.append(trial_data)
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.samples[idx]).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        x = x[:, :1000] 
        
        n_fft = 500 
        hop_length = 6
        
        window = torch.hann_window(n_fft)
        
        stft_res = torch.stft(
            x, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=n_fft, 
            window=window, 
            return_complex=True
        )
        
        mag = torch.abs(stft_res)
        
        mag = mag[:, 4:60, :]
        
        if mag.shape[2] >= 80:
            mag = mag[:, :, :80]
        else:
            pad_size = 80 - mag.shape[2]
            mag = F.pad(mag, (0, pad_size))
            
        return mag, y
    
# def get_eeg_dataloaders(data_dir="./ml", batch_size=64):
#     all_subjects = list(range(1, 10))
    
#     full_dataset = BCI2aDataset(data_dir, subjects=all_subjects, is_train=True)
    
#     train_size = int(0.8 * len(full_dataset))
#     val_size = len(full_dataset) - train_size
    
#     train_dataset, eval_dataset = random_split(
#         full_dataset, 
#         [train_size, val_size],
#         generator=torch.Generator().manual_seed(42)
#     )
    
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         shuffle=True, 
#         num_workers=4, 
#         pin_memory=True
#     )
    
#     val_loader = DataLoader(
#         eval_dataset, 
#         batch_size=batch_size, 
#         shuffle=False, 
#         num_workers=4, 
#         pin_memory=True
#     )
    
#     return train_loader, val_loader