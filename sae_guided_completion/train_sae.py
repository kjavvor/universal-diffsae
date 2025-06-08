#!/usr/bin/env python3
"""
train_sae.py

Train a k-sparse autoencoder on precomputed CLIP activations, with train/val/test splits,
early stopping, LR scheduler, checkpointing, CSV metrics logging, automatic loss plot and
separate saving of test activations for later evaluation.

Usage example:
  python sae_guided_completion/train_sae.py \
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
import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm


class SparseAutoencoder(nn.Module):
    """
    A simple k-sparse autoencoder with linear encoder and decoder.
    """

    def __init__(self, input_dim, latent_dim, k, normalize_decoder=True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.k = k
        self.normalize_decoder = normalize_decoder
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x):
        """
        Forward pass: encode, apply top-k sparsity, then decode.
        """
        z = self.encoder(x)
        if self.k < z.size(1):
            # keep only top-k activations per sample
            vals, idx = torch.topk(z.abs(), self.k, dim=1)
            mask = torch.zeros_like(z, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            z = z * mask
        return self.decoder(z)

    def normalize_decoder_weights(self):
        """
        Normalize decoder weights to unit norm column-wise if enabled.
        """
        if not self.normalize_decoder:
            return
        with torch.no_grad():
            w = self.decoder.weight
            norm = w.norm(dim=0, keepdim=True)
            w.div_(norm + 1e-8)


def train_epoch(model, loader, loss_fn, optimizer, device):
    """
    Run one training epoch and return average loss.
    """
    model.train()
    total_loss = 0.0
    for (x_batch,) in tqdm(loader, desc="  Train", leave=False):
        x_batch = x_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_fn(output, x_batch)
        loss.backward()
        optimizer.step()
        model.normalize_decoder_weights()
        total_loss += loss.item() * x_batch.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, loss_fn, device, tag):
    """
    Evaluate model on validation or test split.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (x_batch,) in tqdm(loader, desc=f"  {tag}", leave=False):
            x_batch = x_batch.to(device)
            output = model(x_batch)
            total_loss += loss_fn(output, x_batch).item() * x_batch.size(0)
    return total_loss / len(loader.dataset)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train SAE on CLIP activations"
    )
    parser.add_argument(
        "--activations-file",
        required=True,
        help="Path to the .pt file containing activations",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for training",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of non-zero latent activations (sparsity level)",
    )
    parser.add_argument(
        "--expansion-factor",
        type=int,
        required=True,
        help="Multiplier for latent dimension relative to input dim",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu, cuda, mps). Auto-detect if unset",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data used for validation",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction of data used for testing",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience on validation loss",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="sae_checkpoints",
        help="Name of subdirectory for saving checkpoints",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="metrics.csv",
        help="Filename for recorded metrics CSV",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "step"],
        default="none",
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=10,
        help="Step size (epochs) for StepLR scheduler",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Learning rate decay factor for StepLR",
    )
    return parser.parse_args()


def main():
    # parse command-line arguments
    args = parse_args()

    # select computing device
    default_device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if args.device:
        try:
            device = torch.device(args.device)
        except Exception:
            print("[WARN] Invalid device, falling back to auto-detect")
            device = default_device
    else:
        device = default_device
    print(f"[INFO] Using device: {device}")

    # load activations from file
    acts = torch.load(
        args.activations_file, map_location="cpu", weights_only=True
    )
    if acts.ndim > 2:
        # flatten sequence dimension if present
        acts = acts.view(-1, acts.shape[-1])
    num_samples, input_dim = acts.shape
    print(f"[INFO] Loaded activations with shape: {acts.shape}")

    # split dataset into train/val/test
    ds = TensorDataset(acts)
    n_val = int(num_samples * args.val_split)
    n_test = int(num_samples * args.test_split)
    n_train = num_samples - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size) if n_test > 0 else None
    )

    # prepare output directories per run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = os.path.join("training_data", timestamp)
    os.makedirs(out_root, exist_ok=True)

    # checkpoints go under out_root/<checkpoint-dir>/
    ckpt_root = os.path.join(out_root, args.checkpoint_dir)
    os.makedirs(ckpt_root, exist_ok=True)

    # save raw test activations for later use
    if n_test > 0 and isinstance(test_ds, torch.utils.data.Subset):
        indices = test_ds.indices
        raw_test = acts[indices]
        test_path = os.path.join(out_root, "test_activations.pt")
        torch.save(raw_test, test_path)
        print(f"[INFO] Saved test activations to: {test_path}")

    # initialize model, optimizer, loss, scheduler
    latent_dim = args.expansion_factor * input_dim
    model = SparseAutoencoder(
        input_dim, latent_dim, args.k, normalize_decoder=True
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    scheduler = (
        optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
        if args.scheduler == "step"
        else None
    )

    # training loop with early stopping
    best_val_loss = float("inf")
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        val_loss = eval_epoch(
            model, val_loader, loss_fn, device, tag="Val"
        )
        if scheduler:
            scheduler.step()

        print(f" ▶ train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )

        # save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            ckpt_path = os.path.join(ckpt_root, f"sae_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] Saved checkpoint: {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(
                    f"[INFO] Early stopping after {no_improve}"
                    " epochs without improvement"
                )
                break

    # final test evaluation
    if test_loader:
        test_loss = eval_epoch(
            model, test_loader, loss_fn, device, tag="Test"
        )
        print(f"\nTest loss: {test_loss:.4f}")
        history.append(
            {"epoch": "test", "train_loss": None, "val_loss": test_loss}
        )

    # save metrics CSV
    metrics_path = os.path.join(out_root, args.metrics_file)
    df = pd.DataFrame(history)
    df.to_csv(metrics_path, index=False)
    print(f"[INFO] Metrics saved to: {metrics_path}")

    # plot loss curve
    epochs = [h["epoch"] for h in history if isinstance(h["epoch"], int)]
    train_vals = [h["train_loss"] for h in history if isinstance(h["epoch"], int)]
    val_vals = [h["val_loss"] for h in history if isinstance(h["epoch"], int)]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_vals, label="train")
    plt.plot(epochs, val_vals, label="val")
    best_epoch = int(
        min(
            (h for h in history if isinstance(h["epoch"], int)),
            key=lambda x: x["val_loss"],
        )["epoch"]
    )
    plt.axvline(best_epoch, linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(out_root, "loss_curve.png")
    plt.savefig(plot_path)
    print(f"[INFO] Loss curve saved to: {plot_path}")


if __name__ == "__main__":
    main()