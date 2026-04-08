import os
import random
import wfdb
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt

def clean_ecg_signal(data, fs=360.0):
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 45.0 / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data)

class ECGDataset(Dataset):
    def __init__(self, records, data_dir, window=256, is_train=False):
        self.samples = []
        self.is_train = is_train
        half = window // 2
        for rec in records:
            sig, _ = wfdb.rdsamp(os.path.join(data_dir, rec))
            ann = wfdb.rdann(os.path.join(data_dir, rec), "atr")
            raw_ecg = sig[:, 0]
            
            clean_ecg = clean_ecg_signal(raw_ecg, fs=360.0)
            
            
            for pos, sym in zip(ann.sample, ann.symbol):
                if pos < half or pos + half >= len(clean_ecg):
                    continue

                segment = clean_ecg[pos-half : pos+half]
                segment = (segment - segment.mean()) / (segment.std() + 1e-8)
                
                if sym == "N":
                    label = 0
                elif sym in ["V","A","L","R","F"]:
                    label = 1
                else:
                    continue

                self.samples.append((segment.astype(np.float32), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if self.is_train:
            shift = random.randint(-10, 10)
            if shift != 0:
                x = np.roll(x, shift)
                if shift > 0:
                    x[:shift] = 0
                else:
                    x[shift:] = 0
                    
            scale = random.uniform(0.9, 1.1)
            x = x * scale
            noise = np.random.normal(0, 0.05, size=x.shape).astype(np.float32)
            x = x + noise

        return torch.from_numpy(x), torch.tensor(y)