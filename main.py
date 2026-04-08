import os
import argparse
import torch
import numpy as np
import random
import warnings

from keys_handling.ecg_key import run_ecg
from keys_handling.eeg_key import run_eeg

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser(description="Run biological signal models.")
    
    parser.add_argument(
        '--task', 
        type=str, 
        choices=['ecg', 'eeg'], 
        required=True, 
        help="Specify which experiment to run: 'ecg' or 'eeg'"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='default',
        help="Specify the model name to run (e.g., 'conformer', 'eegnet')"
    )
    
    parser.add_argument(
        '--agfl',
        type=str,
        choices=['on', 'off'],
        default='off',
        help="Toggle custom AGFL attention: 'on' or 'off'"
    )
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    print(f"Initializing {args.task.upper()} task...")
    print(f"Device: {device} | Model: {args.model} | AGFL: {args.agfl}")

    if args.task == 'ecg':
        run_ecg(device, PROJECT_ROOT, model_name=args.model, agfl_status=args.agfl)
    elif args.task == 'eeg':
        run_eeg(device, PROJECT_ROOT, model_name=args.model, agfl_status=args.agfl, num_classes_global=2)

if __name__ == "__main__":
    main()