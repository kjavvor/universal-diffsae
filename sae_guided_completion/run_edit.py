
import torch
from sae_guided_completion.train_sae import SparseAutoencoder
from sae_guided_completion.edit_prompt_latent import edit_prompt_with_latent
from transformers import CLIPTextModel, CLIPTokenizer

def main():
    # 1) Device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    # 2) Załaduj SAE
    input_dim   = 768               # wymiar CLIP‐owego EOS embeddingu
    expansion  = 32                # taki użyliśmy w train_sae.py
    latent_dim  = input_dim * expansion
    k           = 32               # stała sparsity
    sae = SparseAutoencoder(input_dim, latent_dim, k, normalize_decoder=True)
    sae.load_state_dict(torch.load("sae_model.pth", map_location=device))
    sae.to(device).eval()

    # 3) Załaduj CLIP text encoder + tokenizer
    model_id = "openai/clip-vit-large-patch14"
    tokenizer   = CLIPTokenizer.from_pretrained(model_id)
    text_model  = CLIPTextModel.from_pretrained(model_id).to(device)

    # 4) Przygotuj prompty
    prompt  = "A photo of an astronaut riding a horse on Mars"
    concept = "in Van Gogh style"

    # 5) Wygeneruj baseline (bez wstrzykiwania koncepcji)
    _, img_baseline = edit_prompt_with_latent(
        prompt, "", sae, text_model, tokenizer, device=device
    )
    img_baseline.save("output_baseline.png")
    print("[INFO] Saved baseline image to output_baseline.png")

    # 6) Wygeneruj obraz z wstrzykniętą koncepcją
    _, img_edited = edit_prompt_with_latent(
        prompt, concept, sae, text_model, tokenizer, device=device
    )
    img_edited.save("output_edited.png")
    print("[INFO] Saved edited image to output_edited.png")

if __name__ == "__main__":
    main()