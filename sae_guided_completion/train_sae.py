#!/usr/bin/env python3
"""
train_sae.py

Train a k-sparse autoencoder on precomputed CLIP activations, with train/val/test splits,
early stopping, LR scheduler, checkpointing, CSV metrics logging, automatic loss plot and
separate saving of test activations for later evaluation.

Usage example:
  python train_sae.py \
    --activations-file clip_activations.pt \
    --epochs 20 \
    --batch-size 4096 \
    --k 32 \
    --expansion-factor 32 \
    --lr 5e-4 \
    --device mps \
    --val-split 0.1 \
    --test-split 0.1 \
    --patience 3 \
    --checkpoint-dir sae_checkpoints \
    --metrics-file metrics.csv \
    --scheduler step \
    --step-size 10 \
    --gamma 0.1
"""
import argparse
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, k, normalize_decoder=True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.k = k
        self.normalize_decoder = normalize_decoder
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x):
        z = self.encoder(x)
        if self.k < z.size(1):
            vals, idx = torch.topk(z.abs(), self.k, dim=1)
            mask = torch.zeros_like(z, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            z = z * mask
        return self.decoder(z)

    def normalize_decoder_weights(self):
        if not self.normalize_decoder:
            return
        with torch.no_grad():
            w = self.decoder.weight
            norm = w.norm(dim=0, keepdim=True)
            w.div_(norm + 1e-8)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for (X,) in tqdm(loader, desc="  Train", leave=False):
        X = X.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, X)
        loss.backward()
        optimizer.step()
        model.normalize_decoder_weights()
        running += loss.item() * X.size(0)
    return running / len(loader.dataset)


def eval_epoch(model, loader, criterion, device, tag):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for (X,) in tqdm(loader, desc=f"  {tag}", leave=False):
            X = X.to(device)
            out = model(X)
            running += criterion(out, X).item() * X.size(0)
    return running / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train SAE on CLIP activations")
    parser.add_argument("--activations-file", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--expansion-factor", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--checkpoint-dir", type=str, default="sae_checkpoints")
    parser.add_argument("--metrics-file", type=str, default="metrics.csv")
    parser.add_argument("--scheduler", choices=["none", "step"], default="none")
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    args = parser.parse_args()

    # Device selection with fallback
    default_device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    if args.device:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available, falling back to default device.")
            device = default_device
        else:
            device = torch.device(args.device)
    else:
        device = default_device
    print(f"[INFO] Using device: {device}")

    # Load activations
    acts = torch.load(args.activations_file, map_location="cpu", weights_only=True)
    if acts.ndim > 2:
        D = acts.shape[-1]
        acts = acts.view(-1, D)
    N, D = acts.shape
    print(f"[INFO] Loaded activations shape: {acts.shape}")

    # Splits
    ds = TensorDataset(acts)
    n_val  = int(N * args.val_split)
    n_test = int(N * args.test_split)
    n_train = N - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size) if n_test>0 else None

    # Prepare output dirs
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = os.path.join("training_data", ts)
    os.makedirs(out_root, exist_ok=True)
    metrics_csv = os.path.join(out_root, args.metrics_file)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Save test activations
    if n_test>0:
        # Extract raw test activations
        indices = test_ds.indices if hasattr(test_ds, 'indices') else test_ds.dataset.indices
        test_acts = acts[indices]
        test_path = os.path.join(out_root, "test_activations.pt")
        torch.save(test_acts, test_path)
        print(f"[INFO] Test activations saved to: {test_path}")

    # Model, optimizer, loss
    latent_dim = args.expansion_factor * D
    model = SparseAutoencoder(D, latent_dim, args.k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = (
        optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        if args.scheduler == "step" else None
    )

    # Training
    best_val = float('inf')
    no_impr = 0
    records = []

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss = eval_epoch(model, val_loader, criterion, device, "Val")
        if scheduler: scheduler.step()
        print(f" ▶ train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}")
        records.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": vl_loss})

        # Early stopping
        if vl_loss < best_val:
            best_val = vl_loss
            no_impr = 0
            ckpt = os.path.join(args.checkpoint_dir, f"sae_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"[INFO] Saved checkpoint: {ckpt}")
        else:
            no_impr += 1
            if no_impr >= args.patience:
                print(f"[INFO] Early stopping after {no_impr} non-impr. epochs")
                break

    # Final test
    if test_loader:
        te_loss = eval_epoch(model, test_loader, criterion, device, "Test")
        print(f"\nTest loss: {te_loss:.4f}")
        records.append({"epoch": "test", "train_loss": None, "val_loss": te_loss})

    # Save metrics and plot
    df = pd.DataFrame(records)
    df.to_csv(metrics_csv, index=False)
    print(f"[INFO] Metrics saved to: {metrics_csv}")

    plt.figure(figsize=(6,4))
    epochs = df[df.epoch != 'test'].epoch.astype(int)
    plt.plot(epochs, df.train_loss[:-1], label='train')
    plt.plot(epochs, df.val_loss[:-1], label='val')
    best_epoch = int(df.loc[df.val_loss[:-1].idxmin(), 'epoch'])
    plt.axvline(best_epoch, linestyle='--', label='best epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(out_root, 'loss_curve.png')
    plt.savefig(plot_path)
    print(f"[INFO] Loss curve saved to: {plot_path}")


if __name__ == "__main__":
    main()
