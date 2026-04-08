import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss) 
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def compute_class_weights(train_loader, device):
    all_labels = []
    for _, y in train_loader:
        all_labels.extend(y.numpy())
    
    all_labels = np.array(all_labels)
    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    num_classes = len(class_counts)
    
    weights = total_samples / (num_classes * class_counts)
    
    print(f"Class counts: {class_counts}")
    print(f"Applied weights: {weights}")
    
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_eval_eeg(model, train_loader, val_loader, device, epochs=100, lr=3e-4):
    model.to(device)
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3) 
    class_weights = compute_class_weights(train_loader, device)
    
    loss_fn = FocalLoss(weight=class_weights, gamma=3.0)
    

    warmup_epochs = 25
    warmup_scheduler = LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(opt, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(opt, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == 'cuda'))

    history = {'loss': [], 'val_acc': [], 'lr': []}
    
    pbar = tqdm.tqdm(range(epochs), desc="Training Model", unit="epoch")
    
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                    logits_val = model(x_val)
                    
                val_preds.extend(logits_val.argmax(-1).cpu().numpy())
                val_targets.extend(y_val.numpy())
                
        current_acc = accuracy_score(val_targets, val_preds)
        current_lr = opt.param_groups[0]['lr']
        
        history['loss'].append(avg_loss)
        history['val_acc'].append(current_acc)
        history['lr'].append(current_lr)

        pbar.set_postfix({
            "Loss": f"{avg_loss:.4f}", 
            "Val Acc": f"{current_acc:.4f}",
            "LR": f"{current_lr:.2e}"
        })

    model.eval()
    preds, targets, probs = [], [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                logits = model(x)
                
            preds.extend(logits.argmax(-1).cpu().numpy())
            targets.extend(y.numpy())
            
            probabilities = F.softmax(logits.float(), dim=-1)
            probs.extend(probabilities.cpu().numpy())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro') 
    
    probs = np.array(probs)
    if probs.shape[1] == 2:
        auc = roc_auc_score(targets, probs[:, 1])
    else:
        auc = roc_auc_score(targets, probs, multi_class='ovr')
    
    print(f"Final Results -> Accuracy: {acc:.4f} | F1 (Macro): {f1:.4f} | ROC-AUC: {auc:.4f}")

    return acc, f1, auc, history