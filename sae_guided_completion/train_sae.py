# File: sae_guided_completion/train_sae.py

import torch
import torch.nn as nn
import torch.optim as optim

class SparseAutoencoder(nn.Module):
    """
    A k-sparse autoencoder for language model activations.
    Encoder and decoder are linear layers without bias. 
    The encoder produces a high-dimensional latent vector, 
    then enforces sparsity by keeping only top-k activations.
    """
    def __init__(self, input_dim, latent_dim, k, normalize_decoder=True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.k = k
        self.normalize_decoder = normalize_decoder
        # Linear encoder and decoder. No bias for simplicity and to ease normalization.
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x):
        """
        Forward pass for the autoencoder.
        Returns reconstructed output and sparse latent code.
        """
        # Encoder: produce latent representation (no non-linearity for linear autoencoder)
        z = self.encoder(x)  # shape: (batch_size, latent_dim)
        # Enforce sparsity: zero out all but top-k absolute activations in each latent vector
        if self.k < z.shape[1]:
            # Find the threshold for top-k by absolute value
            # We do this per-sample (row). We use partial sort for efficiency.
            # Get the |z| values and indices of top k
            topk_values, topk_indices = torch.topk(z.abs(), self.k, dim=1)
            # Create a mask of the same shape as z, filled with zeros
            mask = torch.zeros_like(z, dtype=torch.bool)
            # Set True for the top-k indices
            mask.scatter_(1, topk_indices, True)
            # Zero-out everything not in top-k (preserve sign of z where mask is True)
            z = z * mask
        # (If k >= latent_dim, then no sparsity enforced, z remains fully dense)
        # Decoder: reconstruct input from sparse code
        x_recon = self.decoder(z)
        return x_recon, z

    def normalize_decoder_weights(self):
        """Normalize each latent basis vector (decoder weight column) to unit norm."""
        with torch.no_grad():
            # decoder.weight shape: (input_dim, latent_dim)
            # Normalize each column (latent basis vector) to norm 1
            w = self.decoder.weight  # shape [input_dim, latent_dim]
            # Compute norm of each latent dimension's weights (with small epsilon to avoid div by zero)
            eps = 1e-8
            norm = torch.norm(w, dim=0, keepdim=True)  # shape [1, latent_dim]
            w.div_(norm + eps)

def train_sae_on_activations(
    activations, sae_config, num_epochs=5, lr=None, device=None
):
    """
    Train a Sparse Autoencoder (SAE) on the given activation vectors.
    
    Args:
        activations (torch.Tensor or np.ndarray): Activation data of shape (N, input_dim) or (N, seq_len, input_dim).
            If 3D (sequence length included), it will be flattened or treated appropriately.
        sae_config: Configuration object with attributes expansion_factor, num_latents, k, normalize_decoder.
        num_epochs (int): Number of training epochs.
        lr (float or None): Learning rate. If None, use a default or a value based on latent count.
        device: Device to train on (defaults to CUDA if available, else MPS/CPU).
    Returns:
        trained_model (SparseAutoencoder)
    """
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(device)
    print(f"[INFO] Training SAE on device: {device}")

    # Convert activations to torch tensor if not already
    if isinstance(activations, torch.Tensor):
        activations_tensor = activations
    else:
        # assume numpy array
        activations_tensor = torch.from_numpy(activations)
    # If activations have a sequence dimension (N, seq_len, dim), merge them if needed:
    if activations_tensor.dim() == 3:
        # We can treat each token embedding as separate training sample by flattening batch*seq_len
        N, S, D = activations_tensor.shape
        activations_tensor = activations_tensor.view(N * S, D)
        print(f"[INFO] Flattened activations to shape: {activations_tensor.shape} for SAE training.")
    else:
        print(f"[INFO] Activations shape: {activations_tensor.shape} for SAE training.")
    input_dim = activations_tensor.shape[-1]

    # Determine latent dimension
    if sae_config.num_latents and sae_config.num_latents > 0:
        latent_dim = sae_config.num_latents
    else:
        latent_dim = sae_config.expansion_factor * input_dim
    k = sae_config.k
    normalize_decoder = sae_config.normalize_decoder

    # Initialize model and optimizer
    model = SparseAutoencoder(input_dim, latent_dim, k, normalize_decoder=normalize_decoder)
    model.to(device)
    # Choose learning rate: if not provided, base it on latent_dim (for example, smaller LR for larger models)
    if lr is None:
        lr = 1e-3 if latent_dim < 10000 else 5e-4  # heuristic: smaller LR if very high-dimensional
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Prepare data loader for batching
    dataset = torch.utils.data.TensorDataset(activations_tensor)
    # Effective batch size may be large (4096 as in config) if treating each token as an example
    batch_size = min(sae_config.effective_batch_size, len(activations_tensor))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"[INFO] Beginning training for {num_epochs} epochs, batch_size={batch_size}, latent_dim={latent_dim}, k={k}")

    model.train()
    for epoch in range(1, num_epochs+1):
        total_loss = 0.0
        for (batch_X,) in dataloader:
            batch_X = batch_X.to(device)
            optimizer.zero_grad()
            # Forward pass
            recon_X, latent = model(batch_X)
            # Compute reconstruction loss
            loss = criterion(recon_X, batch_X)
            loss.backward()
            optimizer.step()
            # Optionally normalize decoder weights to maintain unit norm (to prevent trivial scaling)
            if normalize_decoder:
                model.normalize_decoder_weights()
            total_loss += loss.item() * batch_X.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{num_epochs} - Reconstruction Loss: {avg_loss:.6f}")
    print("[INFO] Training complete.")
    return model

# Example usage (if run as script):
if __name__ == "__main__":
    import sys, os
    # Quick example: assume activations file path given
    if len(sys.argv) < 2:
        print("Usage: python train_sae.py <activations_file.pt>")
        sys.exit(1)
    act_file = sys.argv[1]
    if act_file.endswith(".pt"):
        acts = torch.load(act_file)
    elif act_file.endswith(".npy"):
        import numpy as np
        acts = np.load(act_file)
    else:
        raise ValueError("Unsupported file format for activations. Use .pt or .npy")
    # Example SAE config values (could be loaded from YAML or similar in practice)
    class Config:  # dummy config for demonstration
        expansion_factor = 32
        num_latents = 0  # will use expansion_factor * input_dim
        k = 32
        normalize_decoder = True
        effective_batch_size = 4096
    sae_model = train_sae_on_activations(acts, Config(), num_epochs=5)
    # Save trained SAE model weights
    torch.save(sae_model.state_dict(), "sae_model.pth")
    print("[INFO] SAE model saved to sae_model.pth")