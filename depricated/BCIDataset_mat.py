import os
import scipy.io as sio
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, DataLoader


def bandpass_eeg_signal(data, fs=250.0, lowcut=2.0, highcut=30.0):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

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
            filepath = os.path.join(data_dir, f"A{subj:02d}{suffix}.mat")
            if not os.path.exists(filepath):
                print(f"Warning: {filepath} not found. Skipping.")
                continue
                
            mat_dict = sio.loadmat(filepath)
            data_array = mat_dict['data']
            
            for run_idx in range(data_array.shape[1]):
                run_struct = data_array[0, run_idx]
                
                if isinstance(run_struct, np.ndarray) and run_struct.shape == (1, 1):
                    run_struct = run_struct[0, 0]
                    
                if 'y' not in run_struct.dtype.names or len(run_struct['y']) == 0:
                    continue
                    
                X_cont = run_struct['X'][:, :22].T.astype(np.float32)
                X_cont = bandpass_eeg_signal(X_cont, fs=250.0)
                
                run_mean = X_cont.mean(axis=-1, keepdims=True)
                run_std = X_cont.std(axis=-1, keepdims=True) + 1e-8
                X_cont = (X_cont - run_mean) / run_std
                
                trials = run_struct['trial'].flatten()
                y_run = run_struct['y'].flatten()
                
                offset = 475 
                max_window_size = 1050 
                
                for i, start_idx in enumerate(trials):
                    actual_start = start_idx + offset
                    if actual_start + max_window_size > X_cont.shape[1]:
                        continue
                        
                    raw_label = y_run[i]
                    if np.isnan(raw_label):
                        continue
                    
                    # BCI2a classes: 1=Left, 2=Right, 3=Foot, 4=Tongue
                    if int(raw_label) not in [1, 2]:
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
            
        return x, y


def get_eeg_dataloaders(data_dir="./ml", batch_size=64):
    all_subjects = list(range(1, 10))
    
    train_dataset = BCI2aDataset(data_dir, subjects=all_subjects, is_train=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    eval_dataset = BCI2aDataset(data_dir, subjects=all_subjects, is_train=False)
    val_loader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader