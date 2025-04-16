# File: sae_guided_completion/edit_prompt_latent.py

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline

def edit_prompt_with_latent(
    original_prompt, concept, sae_model, 
    text_model, tokenizer, pipe=None, device=None
):
    """
    Edit the original prompt by injecting the concept via SAE latent manipulation.
    
    Args:
        original_prompt (str): The text prompt for the original image.
        concept (str): The concept or object to add to the prompt.
        sae_model (SparseAutoencoder): Trained SAE with encoder/decoder for CLIP embeddings.
        text_model (CLIPTextModel): CLIP text encoder (should match the one used in Stable Diffusion).
        tokenizer (CLIPTokenizer): Tokenizer for the CLIP text model.
        pipe (StableDiffusionPipeline or None): Diffusers pipeline for image generation. 
                                               If None, one will be created.
        device: Torch device to use (should be same device as models).
    Returns:
        edited_prompt_embeds (torch.Tensor): The modified prompt embeddings (for reference).
        image (PIL.Image.Image): Generated image with the concept injected.
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(device)
    text_model.to(device)
    sae_model.to(device)
    sae_model.eval()

    # Encode original prompt and concept prompt to CLIP embeddings
    inputs_orig = tokenizer(original_prompt, padding="max_length", max_length=tokenizer.model_max_length,
                             truncation=True, return_tensors="pt")
    inputs_concept = tokenizer(concept, padding="max_length", max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors="pt")
    input_ids_orig = inputs_orig.input_ids.to(device)
    input_ids_concept = inputs_concept.input_ids.to(device)
    attention_mask_orig = inputs_orig.attention_mask.to(device)
    attention_mask_concept = inputs_concept.attention_mask.to(device)
    with torch.no_grad():
        orig_outputs = text_model(input_ids_orig, attention_mask=attention_mask_orig)
        concept_outputs = text_model(input_ids_concept, attention_mask=attention_mask_concept)
    # Last hidden states (batch_size=1)
    orig_embedding_seq = orig_outputs.last_hidden_state  # shape [1, 77, 768] for SD1.x
    concept_embedding_seq = concept_outputs.last_hidden_state  # shape [1, 77, 768]
    # We will use the EOS token embedding as representation of the whole concept
    eos_token_id = tokenizer.eos_token_id
    # Find eos index in concept (should be at position of actual end-of-text)
    eos_idx_concept = (input_ids_concept == eos_token_id).nonzero(as_tuple=True)[1].item()
    concept_vector = concept_embedding_seq[0, eos_idx_concept, :].unsqueeze(0)  # shape [1, 768]
    # Similarly get eos embedding for original prompt (the overall prompt representation)
    eos_idx_orig = (input_ids_orig == eos_token_id).nonzero(as_tuple=True)[1].item()
    orig_vector = orig_embedding_seq[0, eos_idx_orig, :].unsqueeze(0)  # shape [1, 768]

    # Encode both through SAE to get latent sparse codes
    with torch.no_grad():
        orig_latent_full = sae_model.encoder(orig_vector)  # full latent (not yet sparse)
        concept_latent_full = sae_model.encoder(concept_vector)
        # Enforce sparsity (top-k) on these latents
        if sae_model.k < sae_model.latent_dim:
            # zero-out outside top-k for each
            def topk_mask(z, k):
                vals, idx = torch.topk(z.abs(), k, dim=1)
                mask = torch.zeros_like(z, dtype=torch.bool)
                mask.scatter_(1, idx, True)
                return mask
            mask_orig = topk_mask(orig_latent_full, sae_model.k)
            mask_concept = topk_mask(concept_latent_full, sae_model.k)
            orig_latent = orig_latent_full * mask_orig
            concept_latent = concept_latent_full * mask_concept
        else:
            orig_latent = orig_latent_full
            concept_latent = concept_latent_full

    # Combine latents: here we add the concept latent to the original latent
    combined_latent = orig_latent + concept_latent
    # Enforce sparsity on combined latent as well (to keep it at k nonzeros)
    if sae_model.k < sae_model.latent_dim:
        mask_combined = torch.zeros_like(combined_latent, dtype=torch.bool)
        # Instead of arbitrary top-k, we ensure all indices that were active in either original or concept remain.
        # This could result in up to 2k active. To strictly keep k, one strategy is to take union and then prune lowest.
        active_indices = torch.where((mask_orig | mask_concept)[0])[0]
        if active_indices.numel() > sae_model.k:
            # prune to k by magnitude
            vals, idx = torch.topk(combined_latent.abs(), sae_model.k, dim=1)
            mask_combined.scatter_(1, idx, True)
        else:
            mask_combined[0, active_indices] = True
        combined_latent = combined_latent * mask_combined

    # Decode the combined latent to get modified embedding vector
    with torch.no_grad():
        modified_vector = sae_model.decoder(combined_latent)  # shape [1, 768]
    # Now we need to integrate this modified vector back into the full token embedding sequence.
    # We will replace the original prompt's EOS embedding with this modified vector, 
    # and also insert the concept as a new token embedding if possible.
    edited_embedding_seq = orig_embedding_seq.clone()
    # Option 1: replace EOS token's embedding (global prompt context) with modified vector
    edited_embedding_seq[0, eos_idx_orig, :] = modified_vector
    # Option 2: if there's a padding slot available, insert the concept vector there as well
    # Find first padding index after eos in original prompt
    pad_start = eos_idx_orig + 1
    if pad_start < edited_embedding_seq.size(1):
        # Use one pad position to represent the new concept explicitly
        edited_embedding_seq[0, pad_start, :] = concept_vector
        # We should also adjust the attention mask to consider this position as valid
        attention_mask_orig[0, pad_start] = 1

    # Use or create a StableDiffusionPipeline to generate image from edited embeddings
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float32 if device.type != "cuda" else torch.float16
        )
        pipe = pipe.to(device)
    # Generate image using the pipeline with custom prompt embeddings
    with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.float32):
        image = pipe(prompt_embeds=edited_embedding_seq, attention_mask=attention_mask_orig, num_inference_steps=50).images[0]
    return edited_embedding_seq, image

# Example usage:
if __name__ == "__main__":
    # This example assumes you have trained sae_model and have it loaded
    # and also have tokenizer, text_model (CLIP) and a pipeline ready.
    from PIL import Image
    # Dummy load (in practice, load actual trained weights and models)
    sae_model = torch.load("sae_model.pth") if torch.cuda.is_available() else torch.load("sae_model.pth", map_location="cpu")
    # (Note: if saved with state_dict, you'd initialize SparseAutoencoder and load state_dict)
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    edited_embeds, out_img = edit_prompt_with_latent(
        "A cat sitting on a chair", "a ball", sae_model, text_model, tokenizer
    )
    out_img.save("edited_output.png")
    print("[INFO] Saved edited prompt image to edited_output.png")
