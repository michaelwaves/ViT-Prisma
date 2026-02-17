"""Find top activating ImageNet images per SAE feature."""

import click
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from vit_prisma.sae import SparseAutoencoder
from vit_prisma.models.model_loader import load_hooked_model
from vit_prisma.transforms import get_clip_val_transforms
from vit_prisma.dataloaders.imagenet_index import imagenet_index

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAE_REPO = "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-10-hook_mlp_out-l1-1e-05"
IMAGENET_DATASET = "evanarlian/imagenet_1k_resized_256"
IMAGENET_CLASSES = [imagenet_index[str(i)][1] for i in range(1000)]


@click.command()
@click.option("--batch-size", default=64)
@click.option("--top-k", default=5, help="Top images per feature")
@click.option("--features", default=None, help="Comma-separated feature indices to visualize")
@click.option("--n-features", default=10, help="Auto-select N most active features")
@click.option("--output", default="top_images_per_feature.png")
@click.option("--cache", default="topk_cache.pt", help="Cache file for top-k results")
def main(batch_size, top_k, features, n_features, output, cache):
    print("Loading SAE...")
    sae = load_sae().to(DEVICE)
    print(f"Loading model: {sae.cfg.model_name}")
    model = load_hooked_model(sae.cfg.model_name).to(DEVICE)
    print("Loading ImageNet validation set...")
    hf_dataset = load_dataset(IMAGENET_DATASET, split="val")
    dataset = ImageNetDataset(hf_dataset, get_clip_val_transforms())
    print(f"Finding top-{top_k} images across {len(dataset)} images...")
    topk_values, topk_indices = load_or_compute_topk(
        cache, dataset, model, sae, batch_size, top_k
    )
    feature_ids = select_feature_indices(features, topk_values, n_features)
    print(f"Visualizing features: {feature_ids}")
    visualize_top_images(dataset.hf_dataset, topk_values, topk_indices, feature_ids, output)


def load_sae() -> SparseAutoencoder:
    weights_path = hf_hub_download(SAE_REPO, "weights.pt")
    hf_hub_download(SAE_REPO, "config.json")
    return SparseAutoencoder.load_from_pretrained(weights_path)


def load_or_compute_topk(cache_path, dataset, model, sae, batch_size, top_k):
    if Path(cache_path).exists():
        print(f"Loading cached top-k from {cache_path}")
        cached = torch.load(cache_path, weights_only=True)
        return cached["values"], cached["indices"]
    values, indices = find_top_activating_images(dataset, model, sae, batch_size, top_k)
    torch.save({"values": values, "indices": indices}, cache_path)
    print(f"Cached results to {cache_path}")
    return values, indices


def find_top_activating_images(dataset, model, sae, batch_size, top_k):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    n_features = sae.cfg.d_sae
    vals = torch.full((n_features, top_k), -float("inf"))
    idxs = torch.full((n_features, top_k), -1, dtype=torch.long)

    for batch_num, (images, indices) in enumerate(loader):
        with torch.no_grad():
            _, cache = model.run_with_cache(
                images.to(DEVICE), names_filter=sae.cfg.hook_point
            )
            _, feature_acts, *_ = sae(cache[sae.cfg.hook_point])

        max_acts = feature_acts[:, 1:, :].max(dim=1).values.cpu()
        combined_vals = torch.cat([vals, max_acts.T], dim=1)
        combined_idxs = torch.cat(
            [idxs, indices.unsqueeze(0).expand(n_features, -1)], dim=1
        )
        best = combined_vals.topk(top_k, dim=1)
        vals = best.values
        idxs = combined_idxs.gather(1, best.indices)

        if batch_num % 50 == 0:
            print(f"  Batch {batch_num}/{len(loader)}")

    return vals, idxs


def select_feature_indices(features_str, topk_values, n_auto):
    if features_str:
        return [int(x) for x in features_str.split(",")]
    total_activation = topk_values.sum(dim=1)
    return total_activation.topk(n_auto).indices.tolist()


def visualize_top_images(hf_dataset, topk_values, topk_indices, feature_ids, output_path):
    n_rows = len(feature_ids)
    top_k = topk_indices.shape[1]
    fig, axes = plt.subplots(n_rows, top_k, figsize=(3 * top_k, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, feat_id in enumerate(feature_ids):
        for col in range(top_k):
            img_idx = topk_indices[feat_id, col].item()
            act_val = topk_values[feat_id, col].item()
            sample = hf_dataset[img_idx]
            image = sample["image"].convert("RGB").resize((224, 224))
            label = IMAGENET_CLASSES[sample["label"]]

            axes[row, col].imshow(image)
            axes[row, col].set_title(f"{label}\n{act_val:.2f}", fontsize=8)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(
            f"Feature {feat_id}", fontsize=10, rotation=0, labelpad=70
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


class ImageNetDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        image = self.transform(self.hf_dataset[idx]["image"].convert("RGB"))
        return image, idx


if __name__ == "__main__":
    main()
