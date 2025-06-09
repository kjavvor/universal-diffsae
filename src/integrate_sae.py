"""
usage:
python src/integrate_sae.py \
  --prompt "A dog wearing red glasses" \
  --k 32 \
  --steps 70 \
  --scale 7.5 \
  --blend 0.8

This script:
 1) Encodes text prompt with CLIP.
 2) Compresses/reconstructs the EOS embeddings via a k-sparse autoencoder.
 3) Blends the original and reconstructed embeddings.
 4) Feeds them into Stable Diffusion for generation.
"""

import argparse
import os
import re
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image

# Path to your trained SAE checkpoint (.pth)
SAE_CHECKPOINT = (
    "/Users/kamiljaworski/Projects/ZZSN/universal-diffsae/"
    "training_data/20250608-144852/sae_checkpoints/sae_epoch20.pth"
)

# Stable Diffusion model ID used for both CLIP and U-Net/VAE
SD_MODEL_ID = "CompVis/stable-diffusion-v1-4"

# Where to save generated images
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../outputs")


class SparseAutoencoder(nn.Module):
    """Linear encoder/decoder with top-k sparsity."""

    def __init__(self, input_dim: int, latent_dim: int, k: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        z = self.encoder(x)
        # Enforce k-sparsity per sample
        if self.k < z.size(1):
            _, idx = torch.topk(z.abs(), self.k, dim=1)
            mask = torch.zeros_like(z, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            z = z * mask
        # Decode
        return self.decoder(z)


def get_device() -> torch.device:
    """Select MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_sae(device: torch.device, k: int) -> SparseAutoencoder:
    """
    Load SAE checkpoint (weights_only=True), infer dims,
    and return an eval() model on `device`.
    """
    state_dict = torch.load(
        SAE_CHECKPOINT, map_location=device, weights_only=True
    )
    latent_dim, input_dim = state_dict["encoder.weight"].shape
    sae = SparseAutoencoder(input_dim, latent_dim, k).to(device)
    sae.load_state_dict(state_dict)
    sae.eval()
    return sae


def load_sd_pipeline(device: torch.device) -> StableDiffusionPipeline:
    """
    Load SD pipeline, enable slicing, switch scheduler,
    stub out safety checker and clamp NaNs.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    pipe.enable_attention_slicing()
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    # Stub safety checker (always returns safe)
    def dummy_safety(images, **kwargs):
        return images, [False] * len(images)

    pipe.safety_checker = dummy_safety  # type: ignore

    # Patch numpy_to_pil to sanitize NaNs
    orig_np2pil = pipe.image_processor.numpy_to_pil

    def safe_numpy_to_pil(images, *args, **kwargs):
        clean = np.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
        clean = np.clip(clean, 0.0, 1.0)
        return orig_np2pil(clean, *args, **kwargs)

    pipe.image_processor.numpy_to_pil = safe_numpy_to_pil  # type: ignore

    return pipe


def sanitize_filename(text: str) -> str:
    """Replace unsafe chars and spaces with underscores."""
    safe = re.sub(r"[^\w\-_\. ]", "_", text).strip()
    return safe.replace(" ", "_")


def incremental_filename(base: str, ext: str = ".png") -> str:
    """
    If base.png exists, append _1, _2, … until a free name is found.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, base + ext)
    if not os.path.exists(path):
        return path
    idx = 1
    while True:
        candidate = os.path.join(OUTPUT_DIR, f"{base}_{idx}{ext}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def generate_with_sae(
    prompt: str,
    sae: SparseAutoencoder,
    pipe: StableDiffusionPipeline,
    tokenizer: CLIPTokenizer,
    text_model: CLIPTextModel,
    device: torch.device,
    steps: int,
    scale: float,
    blend: float,
) -> Image.Image:
    """
    1) Tokenize & CLIP→embeddings
    2) SAE compress/decompress
    3) Blend embeddings
    4) SD inference with blended embeddings
    """
    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        clip_out = text_model(
            inputs.input_ids, attention_mask=inputs.attention_mask
        )
        orig_embeds = clip_out.last_hidden_state  # [1, seq_len, dim]

    # Flatten → SAE → reshape
    b, s, d = orig_embeds.shape
    flat = orig_embeds.view(b * s, d)
    with torch.no_grad():
        recon_flat = sae(flat)
    recon_embeds = recon_flat.view(b, s, d)

    # Blend original vs SAE
    blended = blend * orig_embeds + (1.0 - blend) * recon_embeds

    # Autocast only on CUDA
    ctx = torch.autocast("cuda") if device.type == "cuda" else nullcontext()
    with ctx:
        image = pipe(
            prompt_embeds=blended,
            attention_mask=inputs.attention_mask,
            guidance_scale=scale,
            num_inference_steps=steps,
        ).images[0]

    return image


def main():
    parser = argparse.ArgumentParser(
        description="Run Stable Diffusion with integrated SAE compression"
    )
    parser.add_argument(
        "--prompt",
        nargs="+",
        required=True,
        help="One or more text prompts to generate",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=64,
        help="Number of active latents in the SAE (higher = less compression)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of SD denoising steps (higher = more detail)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="Classifier-free guidance scale (higher = stricter adherence)",
    )
    parser.add_argument(
        "--blend",
        type=float,
        default=0.8,
        help=(
            "Blend factor between original and SAE embeddings "
            "(1.0=only original, 0.0=only SAE)"
        ),
    )

    args = parser.parse_args()
    device = get_device()
    print(f"[INFO] Using device: {device}")

    sae = load_sae(device, args.k)
    tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_ID, subfolder="tokenizer")
    text_model = CLIPTextModel.from_pretrained(
        SD_MODEL_ID, subfolder="text_encoder"
    ).to(device)
    pipe = load_sd_pipeline(device)

    for prompt in args.prompt:
        print(f"[INFO] Generating: '{prompt}'")
        img = generate_with_sae(
            prompt,
            sae,
            pipe,
            tokenizer,
            text_model,
            device,
            args.steps,
            args.scale,
            args.blend,
        )
        base = sanitize_filename(prompt)
        out_path = incremental_filename(base)
        img.save(out_path)
        print(f"[INFO] Saved → {out_path}")

    print("[INFO] All prompts done.")


if __name__ == "__main__":
    main()