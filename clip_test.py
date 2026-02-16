"""Minimal SAE feature visualization on a single image."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import open_clip
from PIL import Image
from huggingface_hub import hf_hub_download

from vit_prisma.sae import SparseAutoencoder
from vit_prisma.models.model_loader import load_hooked_model
from vit_prisma.transforms import get_clip_val_transforms
from vit_prisma.dataloaders.imagenet_index import imagenet_index

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-10-hook_mlp_out-l1-1e-05"
IMAGE_PATH = "src/vit_prisma/sample_images/n01818515_39.JPEG"
N_TOP_FEATURES = 10
IMAGENET_CLASSES = [imagenet_index[str(i)][1] for i in range(1000)]


def main():
    print("Loading SAE...")
    sae = load_sae(REPO_ID).to(DEVICE)

    print(f"Loading model: {sae.cfg.model_name}")
    model = load_hooked_model(sae.cfg.model_name).to(DEVICE)

    print(f"Loading image: {IMAGE_PATH}")
    image = load_image(IMAGE_PATH).to(DEVICE)

    print_clip_predictions(IMAGE_PATH)

    with torch.no_grad():
        feature_acts = get_feature_activations(image, model, sae)

    print(f"\nFeature activations shape: {feature_acts.shape}")
    print_top_features(feature_acts, N_TOP_FEATURES)

    spatial_top = get_top_spatial_features(feature_acts, N_TOP_FEATURES)
    print(f"\nTop {N_TOP_FEATURES} spatially active features (by patch sum):")
    for idx in spatial_top:
        print(f"  Feature {idx}")

    plot_patch_heatmap(IMAGE_PATH, feature_acts, spatial_top)


def load_sae(repo_id: str) -> SparseAutoencoder:
    weights_path = hf_hub_download(repo_id, "weights.pt")
    hf_hub_download(repo_id, "config.json")
    return SparseAutoencoder.load_from_pretrained(weights_path)


def load_image(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    transform = get_clip_val_transforms()
    return transform(image).unsqueeze(0)


def print_clip_predictions(image_path: str, n_top: int = 10):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    image = preprocess(Image.open(image_path).convert("RGB")
                       ).unsqueeze(0).to(DEVICE)
    prompts = [f"a photo of a {name}" for name in IMAGENET_CLASSES]
    text_tokens = tokenizer(prompts).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).squeeze(0)

    probs = similarity.softmax(dim=-1).cpu().numpy()
    top_indices = np.argsort(probs)[-n_top:][::-1]

    print(f"\nTop {n_top} CLIP predictions:")
    for idx in top_indices:
        print(f"  {IMAGENET_CLASSES[idx]:>30s}: {probs[idx]:.4f}")


def get_feature_activations(
    image: torch.Tensor, model: torch.nn.Module, sae: SparseAutoencoder
) -> torch.Tensor:
    _, cache = model.run_with_cache(image, names_filter=sae.cfg.hook_point)
    activations = cache[sae.cfg.hook_point].to(DEVICE)
    _, feature_acts, *_ = sae(activations)
    return feature_acts


def print_top_features(feature_acts: torch.Tensor, n_top: int) -> np.ndarray:
    cls_acts = feature_acts[0, 0, :].detach().cpu().numpy()
    top_indices = np.argsort(cls_acts)[-n_top:][::-1]
    print(f"\nTop {n_top} SAE features (CLS token):")
    for idx in top_indices:
        print(f"  Feature {idx}: {cls_acts[idx]:.4f}")
    return top_indices


def get_top_spatial_features(feature_acts: torch.Tensor, n_top: int) -> np.ndarray:
    patch_acts = feature_acts[0, 1:, :].detach().cpu().numpy()
    total_per_feature = patch_acts.sum(axis=0)
    return np.argsort(total_per_feature)[-n_top:][::-1]


def plot_patch_heatmap(
    image_path: str, feature_acts: torch.Tensor, top_indices: np.ndarray
):
    patch_acts = feature_acts[0, 1:, :]
    num_patches = patch_acts.shape[0]
    grid_size = int(np.sqrt(num_patches))

    raw_image = Image.open(image_path).convert("RGB").resize((224, 224))

    n_features = min(len(top_indices), 5)
    fig, axes = plt.subplots(
        1, n_features + 1, figsize=(4 * (n_features + 1), 4))

    axes[0].imshow(raw_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, feat_idx in enumerate(top_indices[:n_features]):
        heatmap = patch_acts[:, feat_idx].detach().cpu().numpy()
        heatmap = heatmap.reshape(grid_size, grid_size)

        axes[i + 1].imshow(raw_image, alpha=0.5)
        axes[i + 1].imshow(heatmap, cmap="hot", alpha=0.5, interpolation="bilinear",
                           extent=[0, 224, 224, 0])
        axes[i + 1].set_title(f"Feature {feat_idx}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig("sae_features.png", dpi=150, bbox_inches="tight")
    print("\nSaved visualization to sae_features.png")
    plt.show()


if __name__ == "__main__":
    main()
