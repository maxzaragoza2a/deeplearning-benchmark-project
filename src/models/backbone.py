"""Embedding : DINOv2 pre-entraine et GELE, puis une petite tete entrainee par la loss.

Le backbone ne recoit jamais de gradient : ses embeddings sont calcules une seule fois
puis mis en cache (`cached_embeddings`). Seule la tete (`ProjectionHead`) est entrainee,
directement sur ces embeddings.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from transformers import Dinov2Model

PATCH_SIZE = 14

# Variantes officielles (Hugging Face) ; la valeur est le checkpoint charge.
CHECKPOINTS: dict[str, str] = {
    "dinov2_s": "facebook/dinov2-small",
    "dinov2_b": "facebook/dinov2-base",
    "dinov2_l": "facebook/dinov2-large",
}

# Normalisation ImageNet, celle du pretraitement officiel de DINOv2.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_backbone(name: str) -> Dinov2Model:
    """DINOv2 pre-entraine, en mode eval et sans gradient."""
    if name not in CHECKPOINTS:
        raise KeyError(f"Modele inconnu : {name}. Disponibles : {sorted(CHECKPOINTS)}")
    model = Dinov2Model.from_pretrained(CHECKPOINTS[name])
    model.requires_grad_(False)
    return model.eval()


@torch.no_grad()
def extract_embeddings(
    model: Dinov2Model, images: torch.Tensor, device: str, batch_size: int = 64
) -> torch.Tensor:
    """Images uint8 (n, 3, H, W) -> jetons CLS apres la norme finale (n, dim), float32 sur CPU."""
    height, width = images.shape[-2:]
    if height % PATCH_SIZE or width % PATCH_SIZE:
        raise ValueError(f"Taille d'image multiple de {PATCH_SIZE} attendue, recu {width}x{height}")
    model = model.to(device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    chunks = []
    for start in range(0, len(images), batch_size):
        pixels = (images[start : start + batch_size].to(device).float() / 255.0 - mean) / std
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            chunks.append(model(pixel_values=pixels).pooler_output.float().cpu())
    return torch.cat(chunks)


def cached_embeddings(
    name: str,
    paths: list[Path],
    input_size: tuple[int, int],
    load_images,
    device: str,
    cache_dir: Path,
    batch_size: int = 64,
) -> torch.Tensor:
    """Embeddings du backbone gele pour `paths`, relus depuis `cache_dir` s'ils existent.

    Le fichier de cache est nomme d'apres le modele, la taille d'entree et la liste exacte
    des images : changer de dataset, de dedoublonnage ou de `limit_per_class` en cree un
    nouveau. `load_images()` n'est appele qu'en l'absence de cache.
    """
    content = [f"{input_size[0]}x{input_size[1]}"] + [p.as_posix() for p in paths]
    key = hashlib.md5("\n".join(content).encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f"{name}_{key}.pt"
    if cache.exists():
        print(f"Embeddings relus depuis {cache}")
        return torch.load(cache)
    embeddings = extract_embeddings(load_backbone(name), load_images(), device, batch_size)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, cache)
    print(f"Embeddings {tuple(embeddings.shape)} mis en cache dans {cache}")
    return embeddings


class ProjectionHead(nn.Module):
    """Tete entrainee sur les embeddings geles : lineaire, ou MLP a une couche cachee.

    `hidden_dim=None` (defaut) donne une simple projection lineaire.
    """

    def __init__(self, in_dim: int, out_dim: int = 384, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(in_dim, out_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)
