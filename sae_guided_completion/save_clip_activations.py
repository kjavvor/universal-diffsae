# File: sae_guided_completion/save_clip_activations.py

import torch
from transformers import CLIPTextModel, CLIPTokenizer

def save_clip_activations(
    prompts, output_file, layer_index=None, model_id="openai/clip-vit-large-patch14", device=None
):
    """
    Save CLIP text encoder activations for given prompts.
    
    Args:
        prompts (List[str]): List of text prompts to encode.
        output_file (str): File path to save the activations (as .pt or .npy).
        layer_index (int or None): Which hidden layer to extract. 
                                   None means use final hidden state (last layer).
        model_id (str): HuggingFace model ID for the CLIP text encoder.
        device (torch.device or str or None): Device to run on (defaults to CPU or MPS if available).
    """
    # Set up device (default to MPS if available, else CUDA, else CPU)
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    print(f"[INFO] Using device: {device} for CLIP activation extraction")

    # Load tokenizer and text model
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    text_model = CLIPTextModel.from_pretrained(model_id)
    text_model.to(device)
    
    # If a specific layer is requested, enable output of all hidden states
    output_hidden = layer_index is not None
    if output_hidden:
        text_model.config.output_hidden_states = True

    all_activations = []
    # Loop through prompts and collect activations
    for prompt in prompts:
        # Tokenize prompt (CLIP's max length is typically 77 tokens)
        inputs = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        with torch.no_grad():
            outputs = text_model(input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden)
        if output_hidden:
            hidden_states = outputs.hidden_states  # tuple of layer outputs
            # If layer_index = -1 (last layer) or specific index
            selected_layer = hidden_states[layer_index] if layer_index is not None else hidden_states[-1]
            activations = selected_layer.squeeze(0)  # remove batch dim -> shape (seq_len, hidden_dim)
        else:
            # Use the last hidden state (final layer output)
            last_hidden = outputs.last_hidden_state.squeeze(0)  # shape (seq_len, hidden_dim)
            activations = last_hidden
        # Optionally, focus only on the EOS token's embedding as a summary of the prompt:
        # Find EOS token position: tokenizer.eos_token_id is typically 49407 for CLIP models.
        eos_token_id = tokenizer.eos_token_id
        eos_indices = (input_ids == eos_token_id).nonzero(as_tuple=True)
        if len(eos_indices[1]) > 0:  # if eos token found in the sequence
            eos_idx = eos_indices[1].item()
            eos_embed = activations[eos_idx]  # embedding at EOS position
            # We could choose to save only eos_embed if we want one vector per prompt
            # For now, let's save the full sequence of activations:
        # Append the full sequence (or eos-only, depending on design choice)
        all_activations.append(activations.cpu())
    
    # Stack activations into one tensor (if sequence lengths vary, they should actually be the same due to padding)
    all_activations_tensor = torch.stack(all_activations, dim=0)  # shape: (num_prompts, seq_len, hidden_dim)
    # Save to disk
    if output_file.endswith(".pt"):
        torch.save(all_activations_tensor, output_file)
    elif output_file.endswith(".npy"):
        # Save as numpy array
        import numpy as np
        np.save(output_file, all_activations_tensor.numpy())
    else:
        torch.save(all_activations_tensor, output_file + ".pt")
    print(f"[INFO] Saved CLIP activations for {len(prompts)} prompts to {output_file}")

# Example usage (if run as script):
if __name__ == "__main__":
    sample_prompts = [
        "A red car on a sunny day",
        "A cat sitting on a chair",
        "An astronaut riding a horse on Mars"
    ]
    save_clip_activations(sample_prompts, "clip_activations.pt", layer_index=None)