# File: sae_guided_completion/run_experiment.py

import argparse
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

def main():
    """Main function to run the ablation experiment with device flexibility."""
    parser = argparse.ArgumentParser(description="Run SAE ablation experiment.")
    parser.add_argument(
        "--device", 
        choices=["cpu", "cuda", "mps"], 
        default=None,
        help="Device to run on (defaults to MPS if available, otherwise CUDA or CPU)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="HuggingFace model ID for Stable Diffusion (or path to local checkpoint)."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A photo of an astronaut riding a horse on Mars",
        help="Text prompt for image generation."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of inference steps for image generation."
    )
    args = parser.parse_args()

    # Determine device: default to MPS if available, else CUDA if available, else CPU.
    if args.device:
        # Use the user-specified device if provided
        device = torch.device(args.device)
    else:
        if torch.backends.mps.is_available():  # Apple Silicon Metal Performance Shaders
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # Choose data types: use float32 on MPS/CPU for compatibility, float16 on CUDA for speed
    text_dtype = torch.float32  # Text encoder on float32 for numerical stability
    if device.type == "cuda":
        unet_dtype = torch.float16
        vae_dtype = torch.float16
    else:
        unet_dtype = torch.float32
        vae_dtype = torch.float32

    # Load components of Stable Diffusion
    model_id = args.model
    # 1. CLIP text tokenizer & encoder
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=text_dtype)
    # 2. U-Net for noise prediction
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=unet_dtype)
    # 3. Variational Autoencoder (VAE) for image latent <-> pixel conversion
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=vae_dtype)

    # Move models to target device [oai_citation_attribution:2‡medium.com](https://medium.com/@onkarmishra/stable-diffusion-explained-1f101284484d#:~:text=patch14) [oai_citation_attribution:3‡medium.com](https://medium.com/@onkarmishra/stable-diffusion-explained-1f101284484d#:~:text=,to%28torch_device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    vae = vae.to(device)

    # Enable attention slicing for memory efficiency on smaller GPUs (especially useful on MPS)
    # This splits attention computation to reduce memory footprint.
    if hasattr(unet, "enable_attention_slicing"):
        unet.enable_attention_slicing()

    # Prepare the text prompt tokens
    prompt = args.prompt
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = text_input.input_ids.to(device)

    # Get text embeddings from CLIP text encoder (no gradient needed for inference)
    with torch.no_grad():
        text_outputs = text_encoder(input_ids)
        # `text_outputs[0]` is last_hidden_state (batch_size x seq_len x hidden_dim)
        text_embeddings = text_outputs[0]

    # Check shapes and device
    batch_size, seq_len, hidden_dim = text_embeddings.shape
    assert batch_size == 1, "This experiment expects a single prompt at a time."
    print(f"[DEBUG] text_embeddings shape: {text_embeddings.shape} on {text_embeddings.device}")

    # Prepare the diffusion scheduler (default: LMSDiscreteScheduler or Euler etc. can be chosen)
    from diffusers import DPMSolverMultistepScheduler
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    scheduler.set_timesteps(args.steps, device=device)

    # Generate initial latent noise
    latents = torch.randn(
        (batch_size, unet.config.in_channels, 64, 64),  # 64x64 latent by default for SD 512x512
        device=device,
        dtype=unet_dtype
    )
    latents = latents * scheduler.init_noise_sigma  # scale by initial noise sigma

    # Denoising loop
    print(f"[INFO] Running diffusion for {args.steps} steps...")
    for t in scheduler.timesteps:
        # 1. Scale latents as required by the scheduler
        latent_model_input = scheduler.scale_model_input(latents, timestep=t)

        # 2. Predict noise residual with the U-Net condition on text embeddings
        # The U-Net expects encoder hidden states (text embeddings) as conditioning.
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # 3. Compute previous noisy sample x_{t-1} via scheduler
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # After the final step, decode latents to image using VAE
    with torch.no_grad():
        latents = 1 / 0.18215 * latents  # scale factor used in Stable Diffusion for latent space
        image_tensor = vae.decode(latents).sample  # shape: [batch_size, 3, H, W] in [-1, 1]
        image_tensor = (image_tensor.clamp(-1, 1) + 1) / 2  # rescale to [0, 1]
        image_tensor = image_tensor.cpu().permute(0, 2, 3, 1).numpy()  # to numpy with shape (B, H, W, C)
        # Convert to PIL image for saving or display
        from PIL import Image
        image = Image.fromarray((image_tensor[0] * 255).astype("uint8"))

    # Save or show the output image
    output_path = "output_image.png"
    image.save(output_path)
    print(f"[INFO] Image saved to {output_path}")

if __name__ == "__main__":
    main()