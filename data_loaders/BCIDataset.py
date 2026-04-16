import os
import torch
import torch.nn.functional as F
import numpy as np
import mne
from torch.utils.data import Dataset, DataLoader
import warnings

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
            
            raw.pick(picks=range(22))
            raw.notch_filter(freqs=50.0, verbose='ERROR')
            raw.filter(l_freq=2.0, h_freq=30.0, fir_design='firwin', verbose='ERROR')
            
            raw.set_eeg_reference('average', projection=False, verbose='ERROR')
            
            events, event_dict = mne.events_from_annotations(raw, verbose='ERROR')
            target_event_id = {k: v for k, v in event_dict.items() if k in ['769', '770']}
            inv_event_dict = {v: k for k, v in event_dict.items()}
            
            epochs = mne.Epochs(
                raw, 
                events, 
                event_id=target_event_id, 
                tmin=-0.5, 
                tmax=4.5, 
                baseline=(-0.5, 0.0), 
                preload=True, 
                verbose='ERROR'
            )
            
            X_epoched = epochs.get_data()
            y_events = epochs.events[:, 2]
            
            start_idx = int(1.0 * 250) 
            end_idx = start_idx + int(4.0 * 250)
            
            for i in range(len(X_epoched)):
                event_code = inv_event_dict[y_events[i]]
                label = 0 if event_code == '769' else 1
                
                trial_data = X_epoched[i, :, start_idx:end_idx]
                
                trial_data = trial_data * 1e6
                
                trial_std = trial_data.std(axis=-1, keepdims=True) + 1e-8
                trial_data = trial_data / trial_std 
                
                self.samples.append(trial_data.astype(np.float32))
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.samples[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
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