"""Indexation du dataset et chargement des images en memoire.

Le dataset est organise en `data_root/<split>/<classe>/*.jpg`, avec les memes
classes dans les trois splits (split par images). Ce split d'origine n'est pas
utilisable en few-shot : on indexe tout, et le split par classes est refait dans
`episodes.split_classes`.

Aucun fichier n'est jamais supprime ; les doublons sont ecartes dans l'index.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps

# Une grande partie du dataset d'origine est en JPEG progressif tronque (marqueur
# de fin manquant) : sans ceci Pillow leve "image file is truncated".
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetIndex:
    """Index du dataset apres exclusions. Le label est l'indice dans `class_names`."""

    paths: list[Path]
    labels: np.ndarray  # (n_images,) entiers
    class_names: list[str]  # trie
    excluded_duplicates: list[str]

    def __len__(self) -> int:
        return len(self.paths)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_index(
    data_root: Path,
    splits: list[str],
    dedup: bool = True,
    limit_per_class: int | None = None,
    verbose: bool = True,
) -> DatasetIndex:
    """Parcourt `data_root/<split>/<classe>/` et regroupe les splits par classe."""
    by_class: dict[str, list[Path]] = {}
    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split manquant : {split_dir}")
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            images = sorted(
                p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            by_class.setdefault(class_dir.name, []).extend(images)

    excluded: list[str] = []
    if dedup:
        # Un doublon exact place en support et en requete fausserait un episode.
        seen: set[str] = set()
        for name, images in by_class.items():
            kept = []
            for path in images:
                digest = _md5(path)
                if digest in seen:
                    excluded.append(path.relative_to(data_root).as_posix())
                else:
                    seen.add(digest)
                    kept.append(path)
            by_class[name] = kept

    class_names = sorted(name for name, images in by_class.items() if images)
    paths: list[Path] = []
    labels: list[int] = []
    for label, name in enumerate(class_names):
        images = by_class[name][:limit_per_class]
        paths.extend(images)
        labels.extend([label] * len(images))

    index = DatasetIndex(paths, np.array(labels, dtype=np.int64), class_names, excluded)
    if verbose:
        print(
            f"Images retenues : {len(index)} | classes : {len(class_names)} "
            f"| doublons ecartes : {len(excluded)}"
        )
    return index


def _load_one(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        # Letterbox : ratio preserve, bandes noires (le fond des images est deja noir).
        image = ImageOps.pad(image, size, method=Image.LANCZOS, color=(0, 0, 0))
        return np.asarray(image)


def load_images(index: DatasetIndex, input_size: tuple[int, int]) -> torch.Tensor:
    """Charge tout le dataset en un tenseur uint8 (n_images, 3, hauteur, largeur).

    ~150 Ko par image en 224x224 : les 5 456 images tiennent dans moins d'1 Go, ce
    qui evite de relire le disque a chaque episode.
    """
    with ThreadPoolExecutor() as pool:
        arrays = list(pool.map(lambda path: _load_one(path, input_size), index.paths))
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
