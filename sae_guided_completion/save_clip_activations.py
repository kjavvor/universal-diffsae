#!/usr/bin/env python3
"""
Extract and save CLIP text encoder activations for a list of prompts.
Usage:
  python save_clip_activations.py \
    --prompts-file prompts.txt \
    --output-file activations.pt \
    [--layer-index LAYER] \
    [--model-id MODEL] \
    [--device DEVICE]
"""
import argparse
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

def save_clip_activations(
    prompts, output_file, layer_index=None, model_id="openai/clip-vit-large-patch14", device=None
):
    # Setup device
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

    # Load tokenizer and model
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    text_model = CLIPTextModel.from_pretrained(model_id)
    text_model.to(device)

    # Enable hidden states if specific layer requested
    if layer_index is not None:
        text_model.config.output_hidden_states = True

    all_activations = []
    # Iterate with progress bar
    for prompt in tqdm(prompts, desc="Extracting CLIP activations"):
        inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        with torch.no_grad():
            outputs = text_model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=(layer_index is not None)
            )

        if layer_index is not None:
            hidden_states = outputs.hidden_states
            idx = layer_index if 0 <= layer_index < len(hidden_states) else -1
            activations = hidden_states[idx].squeeze(0)
        else:
            activations = outputs.last_hidden_state.squeeze(0)

        # Optionally extract EOS embedding position (not modifying activations here)
        try:
            eos_token_id = tokenizer.eos_token_id
            positions = (input_ids == eos_token_id).nonzero(as_tuple=True)[1]
            if len(positions) > 0:
                eos_idx = positions[-1].item()
                # You could select only eos embedding with:
                # activations = activations[eos_idx].unsqueeze(0)
        except Exception:
            pass

        all_activations.append(activations.cpu())

    # Stack to tensor
    activations_tensor = torch.stack(all_activations, dim=0)

    # Save
    if output_file.endswith(".npy"):
        import numpy as np
        np.save(output_file, activations_tensor.numpy())
        print(f"[INFO] Saved as .npy: {output_file}")
    else:
        torch.save(activations_tensor, output_file)
        print(f"[INFO] Saved as .pt: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save CLIP text encoder activations.")
    parser.add_argument(
        "--prompts-file", required=True, help="Path to text file with one prompt per line."
    )
    parser.add_argument(
        "--output-file", required=True, help=".pt or .npy output file path."
    )
    parser.add_argument(
        "--layer-index", type=int, default=None,
        help="Hidden layer to extract (None for last_hidden_state, -1 for top layer)."
    )
    parser.add_argument(
        "--model-id", type=str, default="openai/clip-vit-large-patch14",
        help="HuggingFace model ID for CLIP text encoder."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device: cpu, cuda or mps."
    )
    args = parser.parse_args()

    # Read prompts
    with open(args.prompts_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    save_clip_activations(
        prompts,
        args.output_file,
        layer_index=args.layer_index,
        model_id=args.model_id,
        device=args.device
    )