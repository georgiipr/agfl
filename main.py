import os
import argparse
import torch
import numpy as np
import random
import warnings

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
        choices=['eeg', 'nc'], 
        required=True, 
        help="Specify which experiment to run: 'eeg', or 'nc'"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='default',
        help="Specify the model name to run (e.g., 'conformer', 'eegnet', 'snn')"
    )
    
    parser.add_argument(
        '--agfl',
        type=str,
        choices=['on', 'off', 'none'],
        default='off',
        help="Toggle custom AGFL attention: 'on', 'off', or 'none'"
    )
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    print(f"Initializing {args.task.upper()} task")
    print(f"Device: {device} | Model: {args.model} | AGFL: {args.agfl}")

    if args.task in ['eeg', 'nc']:
        run_eeg(
            device, 
            PROJECT_ROOT, 
            model_name=args.model, 
            agfl_status=args.agfl, 
            num_classes_global=4,
            task_name=args.task
        )

if __name__ == "__main__":
    main()