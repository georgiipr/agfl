import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import accuracy_score

def plot_training_curves(history, name, save_path):
    epochs = range(1, len(history['loss']) + 1)
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
    axs[0].plot(epochs, history['loss'], marker='o', color='#d62728', linewidth=2, label='Train Loss')
    axs[0].set_ylabel('Loss', fontsize=12)
    axs[0].set_title(f'{name} Training Trajectory', fontsize=14, fontweight='bold')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend()

    axs[1].plot(epochs, history['val_acc'], marker='o', color='#1f77b4', linewidth=2, label='Val Accuracy')
    axs[1].set_ylabel('Accuracy', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend()
    
    axs[2].plot(epochs, history['lr'], marker='o', color='#2ca02c', linewidth=2, label='Learning Rate')
    axs[2].set_xlabel('Epochs', fontsize=12)
    axs[2].set_ylabel('Learning Rate', fontsize=12)
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_curves_{name}.png", dpi=300)
    plt.close()

def attention_entropy(A):
    eps = 1e-8

    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A)

    return -(A * torch.log(A + eps)).sum(dim=-1).mean().item()

def plot_comparison(results, save_path):
    plt.figure(figsize=(8, 5))
    names = list(results.keys())
    vals = [metrics['Accuracy'] for metrics in results.values()]
    bars = plt.bar(names, vals, color='#4C72B0')
    
    plt.title("Model Comparison (Accuracy)", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0.0, 1.0)
    plt.bar_label(bars, fmt='%.4f', padding=3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/comparison.png", dpi=300)
    plt.close()

def plot_entropy(models, x, save_path):
    plt.figure()
    for name, model in models.items():
        with torch.no_grad():
            _ = model(x)
        ent = []
        for layer in model.layers:
            if hasattr(layer.attn, "last_adj") and layer.attn.last_adj is not None:
                A = layer.attn.last_adj
                if A.ndim == 4:
                    A = A.mean(0).mean(0).cpu().numpy()
                elif A.ndim == 3:
                    A = A.mean(0).cpu().numpy()
                ent.append(attention_entropy(A))
            elif hasattr(layer.attn, "last_attn") and layer.attn.last_attn is not None:
                A = layer.attn.last_attn
                if A.ndim == 4:
                    A = A.mean(0).mean(0).cpu().numpy()
                elif A.ndim == 3:
                    A = A.mean(0).cpu().numpy()
                ent.append(attention_entropy(A))
        if ent:
            plt.plot(ent, marker='o', label=name)

    plt.legend()
    plt.title("Attention Entropy")
    plt.savefig(f"{save_path}/entropy.png")
    plt.close()


def plot_sparsity(models, x, save_path):
    plt.figure()
    for name, model in models.items():
        with torch.no_grad():
            _ = model(x)
        sp = []
        for layer in model.layers:
            A = None
            if hasattr(layer.attn, "last_adj") and layer.attn.last_adj is not None:
                A = layer.attn.last_adj
                if A.ndim == 4:
                    A = A.mean(0).mean(0)
                elif A.ndim == 3:
                    A = A.mean(0)
            elif hasattr(layer.attn, "last_attn") and layer.attn.last_attn is not None:
                A = layer.attn.last_attn
                if A.ndim == 4:
                    A = A.mean(0).mean(0)
                elif A.ndim == 3:
                    A = A.mean(0)

            if A is not None:
                sp.append((A > 1e-3).float().mean().item())

        if sp:
            plt.plot(sp, marker='o', label=name)

    plt.legend()
    plt.title("Sparsity")
    plt.savefig(f"{save_path}/sparsity.png")
    plt.close()

def plot_layer_heatmaps(models, x, save_path, save_prefix="heatmap"):
    for name, model in models.items():
        with torch.no_grad():
            _ = model(x)

        fig, axes = plt.subplots(1, len(model.layers), figsize=(4*len(model.layers), 4))
        if len(model.layers) == 1:
            axes = [axes]

        for i, layer in enumerate(model.layers):
            A = None
            if hasattr(layer.attn, "last_adj") and layer.attn.last_adj is not None:
                A = layer.attn.last_adj
                if A.ndim == 4:
                    A = A.mean(0).mean(0).cpu().numpy()
                elif A.ndim == 3:
                    A = A.mean(0).cpu().numpy()
            elif hasattr(layer.attn, "last_attn") and layer.attn.last_attn is not None:
                A = layer.attn.last_attn
                if A.ndim == 4:
                    A = A.mean(0).mean(0).cpu().numpy()
                elif A.ndim == 3:
                    A = A.mean(0).cpu().numpy()

            if A is not None and A.ndim == 2:
                im = axes[i].imshow(A, aspect='auto', cmap='viridis')
                axes[i].set_title(f"Layer {i}")
                axes[i].set_xlabel("Node")
                axes[i].set_ylabel("Node")
                fig.colorbar(im, ax=axes[i])

        plt.suptitle(f"{name} Layer-wise Attention/Adjacency Heatmaps")
        plt.tight_layout()
        plt.savefig(f"{save_path}/{save_prefix}_{name}.png")
        plt.close()

def plot_embedding_space(models, x, y, save_path, method="tsne", save_prefix="embedding"):
    y_np = y.cpu().numpy() if torch.is_tensor(y) else np.array(y)
    
    for name, model in models.items():
        with torch.no_grad():
            x_dev = x.to(next(model.parameters()).device)
            out = model.input(x_dev.unsqueeze(-1))
            
            out = out + 0.1 * model.pos_emb[:, :out.size(1)]
            
            for i, layer in enumerate(model.layers):
                if model.mode == "agfl":
                    out = layer(out, i, len(model.layers))
                else:
                    out = layer(out)
                    
            out = model.norm(out)
            embeddings = out.mean(dim=1).cpu().numpy()

        if method.lower() == "tsne":
            reducer = TSNE(n_components=2, random_state=0)
        else:
            reducer = umap.UMAP(n_components=2, random_state=0)

        emb2d = reducer.fit_transform(embeddings)

        plt.figure(figsize=(6,6))
        
        for c in np.unique(y_np):
            plt.scatter(emb2d[y_np==c, 0], emb2d[y_np==c, 1], label=f"Class {c}", alpha=0.7)
            
        plt.legend()
        plt.title(f"{name} Embedding {method.upper()}")
        plt.savefig(f"{save_path}/{save_prefix}_{name}_{method}.png")
        plt.close()

def plot_ecg_attention(x, models, save_path, save_prefix="ecg_attention"):
    x_sample = x[0].unsqueeze(0) 
    
    for name, model in models.items():
        device = next(model.parameters()).device
        x_dev = x_sample.to(device)
        
        with torch.no_grad():
            _ = model(x_dev)

        for i, layer in enumerate(model.layers):
            A = None
            
            if hasattr(layer.attn, "last_adj") and layer.attn.last_adj is not None:
                A = layer.attn.last_adj
            elif hasattr(layer.attn, "last_attn") and layer.attn.last_attn is not None:
                A = layer.attn.last_attn

            if A is not None:
                if A.ndim == 4:
                    A = A.mean(0).mean(0).cpu().numpy()
                elif A.ndim == 3:
                    A = A.mean(0).cpu().numpy()

            if A is not None and A.ndim == 2:
                fig, ax1 = plt.subplots(figsize=(10, 4))
                
                im = ax1.imshow(A, aspect='auto', cmap='viridis', alpha=0.85)
                ax1.set_xlabel("Time (Samples)", fontsize=11)
                ax1.set_ylabel("Attention Nodes (Queries)", fontsize=11)
                
                cbar = fig.colorbar(im, ax=ax1, pad=0.1)
                cbar.set_label("Attention Weight", rotation=270, labelpad=15)

                ax2 = ax1.twinx()
                ecg_signal = x_sample[0].cpu().numpy()
                
                ax2.plot(ecg_signal, color='#ff3333', linewidth=2.0, label="ECG Signal")
                ax2.set_ylabel("ECG Amplitude", fontsize=11, color='#ff3333')
                ax2.tick_params(axis='y', labelcolor='#ff3333')
                
                ax1.set_xlim(0, len(ecg_signal) - 1)
                
                plt.title(f"{name} - Layer {i} Attention Mapping", fontsize=13, fontweight='bold')
                
                fig.tight_layout()
                plt.savefig(f"{save_path}/{save_prefix}_{name}_layer{i}.png", dpi=300)
                plt.close(fig)


def plot_sparsity_vs_accuracy(models, val_loader, device, save_path):
    accs = {}
    sparsities = {}

    for name, model in models.items():
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds.extend(logits.argmax(-1).cpu().numpy())
                targets.extend(y.cpu().numpy())
        accs[name] = accuracy_score(targets, preds)

        sparsity_list = []
        x_vis, _ = next(iter(val_loader))
        x_vis = x_vis.to(device)
        with torch.no_grad():
            _ = model(x_vis)
            for layer in model.layers:
                A = None
                if hasattr(layer.attn, "last_adj") and layer.attn.last_adj is not None:
                    A = layer.attn.last_adj
                    if A.ndim == 4:
                        A = A.mean(0).mean(0).cpu().numpy()
                    elif A.ndim == 3:
                        A = A.mean(0).cpu().numpy()
                elif hasattr(layer.attn, "last_attn") and layer.attn.last_attn is not None:
                    A = layer.attn.last_attn
                    if A.ndim == 4:
                        A = A.mean(0).mean(0).cpu().numpy()
                    elif A.ndim == 3:
                        A = A.mean(0).cpu().numpy()
                if A is not None:
                    sparsity_list.append((A > 1e-3).mean().item())
        if sparsity_list:
            sparsities[name] = np.mean(sparsity_list)
        else:
            sparsities[name] = 0.0

    plt.figure(figsize=(6,6))
    for name in models.keys():
        plt.scatter(sparsities[name], accs[name], label=name, s=100)
    plt.xlabel("Average Sparsity")
    plt.ylabel("Validation Accuracy")
    plt.title("Sparsity vs Accuracy")
    plt.legend()
    plt.savefig(f"{save_path}/sparsity_vs_acc.png")
    plt.close()