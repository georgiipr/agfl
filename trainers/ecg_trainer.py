import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tqdm
import copy
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
    
    print(f"Class counts (0: Normal, 1: Abnormal): {class_counts}")
    print(f"Applied weights: {weights}")
    
    return torch.tensor(weights, dtype=torch.float32).to(device)

def train_eval_ecg(model, train_loader, val_loader, device, epochs=50):
    model.to(device)
    lr = 3e-4
    
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3) 
    class_weights = compute_class_weights(train_loader, device)
    loss_fn = FocalLoss(weight=class_weights, gamma=1.0)
    
    warmup_epochs = 10
    warmup_scheduler = LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(opt, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(opt, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    pbar = tqdm.trange(epochs, desc="Training Model", unit="epoch")
    history = {'loss': [], 'val_acc': [], 'lr': []}
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for x_val, y_val in val_loader:
                logits_val = model(x_val.to(device))
                val_preds.extend(logits_val.argmax(-1).cpu().numpy())
                val_targets.extend(y_val.numpy())
                
        current_acc = accuracy_score(val_targets, val_preds)
        
        if current_acc > best_acc:
            best_acc = current_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        history['loss'].append(avg_loss)
        history['val_acc'].append(current_acc)
        history['lr'].append(opt.param_groups[0]['lr'])

        pbar.set_postfix({
            "Loss": f"{avg_loss:.4f}", 
            "Val_Acc": f"{current_acc:.4f}",
            "Best_Acc": f"{best_acc:.4f}",
            "LR": f"{opt.param_groups[0]['lr']:.2e}"
        })

    model.load_state_dict(best_model_wts)
    model.eval()
    
    preds, targets, probs = [], [], []

    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x.to(device))
            preds.extend(logits.argmax(-1).cpu().numpy())
            probabilities = F.softmax(logits, dim=-1)
            probs.extend(probabilities[:, 1].cpu().numpy())
            targets.extend(y.numpy())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro') 
    auc = roc_auc_score(targets, probs)

    return acc, f1, auc, history