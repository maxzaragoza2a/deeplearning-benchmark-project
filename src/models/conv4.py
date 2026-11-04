"""Backbone Conv-4 (Snell et al., 2017), entraine de zero.

Quatre blocs conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2, a 64 canaux. Sur des
images 84x84 (cadre du papier), la sortie fait 64 x 5 x 5 = 1 600 valeurs. Nos
images sont plus grandes (224x224) : un pooling moyen ramene la carte a 5x5, ce
qui garde la meme taille d'embedding que le papier.

Contrairement a DINOv2, ce backbone est entraine par la loss : pas de cache
d'embeddings, les images passent dans le reseau a chaque episode.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class Conv4(nn.Module):
    """Images normalisees (n, 3, H, W) -> embeddings (n, out_dim).

    `pool_size=None` garde la carte de sortie telle quelle (dimension qui depend
    alors de la taille d'entree).
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 64, pool_size: int | None = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hidden_channels),
            conv_block(hidden_channels, hidden_channels),
            conv_block(hidden_channels, hidden_channels),
            conv_block(hidden_channels, hidden_channels),
        )
        self.pool = nn.AdaptiveAvgPool2d(pool_size) if pool_size else nn.Identity()
        self.hidden_channels = hidden_channels
        self.pool_size = pool_size

    @property
    def out_dim(self) -> int | None:
        """Taille de l'embedding, ou None si elle depend de l'entree."""
        return self.hidden_channels * self.pool_size**2 if self.pool_size else None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.pool(self.encoder(images)).flatten(1)


class ImagePreprocessor(nn.Module):
    """uint8 (n, 3, H, W) -> float normalise, avec augmentation en entrainement.

    La moyenne et l'ecart-type par canal sont ceux des images d'entrainement : les
    statistiques ImageNet n'ont pas de sens pour un reseau entraine de zero.
    Augmentations (mode `train()` uniquement), toutes deux sans deformer l'aile :
    - symetrie horizontale aleatoire ;
    - decalage aleatoire d'au plus `max_shift` pixels, complete en noir comme le fond.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, hflip: bool = True, max_shift: int = 0):
        super().__init__()
        self.register_buffer("mean", mean.view(1, -1, 1, 1).float())
        self.register_buffer("std", std.view(1, -1, 1, 1).float())
        self.hflip = hflip
        self.max_shift = max_shift

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float() / 255.0
        if self.training:
            images = self.augment(images)
        return (images - self.mean) / self.std

    def augment(self, images: torch.Tensor) -> torch.Tensor:
        if self.hflip:
            flip = torch.rand(len(images), device=images.device) < 0.5
            images = torch.where(flip.view(-1, 1, 1, 1), images.flip(-1), images)
        if self.max_shift > 0:
            s = self.max_shift
            height, width = images.shape[-2:]
            padded = F.pad(images, (s, s, s, s))  # zeros = noir, la couleur du fond
            offsets = torch.randint(0, 2 * s + 1, (len(images), 2)).tolist()
            images = torch.stack([
                padded[i, :, dy : dy + height, dx : dx + width] for i, (dy, dx) in enumerate(offsets)
            ])
        return images


def channel_stats(images: torch.Tensor, chunk: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Moyenne et ecart-type par canal d'images uint8 (n, 3, H, W), ramenees dans [0, 1].

    Calcule par morceaux pour ne pas convertir tout le dataset en float d'un coup.
    """
    total = torch.zeros(images.shape[1], dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    for start in range(0, len(images), chunk):
        batch = images[start : start + chunk].double() / 255.0
        total += batch.sum(dim=(0, 2, 3)).cpu()
        total_sq += batch.pow(2).sum(dim=(0, 2, 3)).cpu()
    count = images.shape[0] * images.shape[2] * images.shape[3]
    mean = total / count
    std = (total_sq / count - mean.pow(2)).clamp_min(1e-12).sqrt()
    return mean.float(), std.float()


@torch.no_grad()
def embed_images(
    model: nn.Module, preprocess: nn.Module, images: torch.Tensor, device: str, batch_size: int = 256
) -> torch.Tensor:
    """Embeddings (n, d) en mode evaluation (BatchNorm figee, sans augmentation), sur CPU."""
    model.eval()
    preprocess.eval()
    chunks = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device, non_blocking=True)
        chunks.append(model(preprocess(batch)).float().cpu())
    return torch.cat(chunks)
