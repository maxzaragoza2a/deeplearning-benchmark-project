"""Shared evaluation protocol for the three methods (NCA, ProtoNet, PRWN).

Everyone evaluates the same way:
- images of `ds_FSL_224x204` read as is (224x204, no resize, no augmentation);
- `ToTensor()` then ImageNet normalization (`TRANSFORM` below);
- the 1000 episodes of `test_episodes.json`, 5-way, 1-shot and 5-shot, 1 query per class;
- nearest prototype (mean of the support embeddings), squared Euclidean distance;
- mean accuracy over episodes, 95% CI = 1.96 * std / sqrt(n_episodes).

Two scores per setting: on raw embeddings (main result) and after CL2N (secondary
result): embeddings centered on the mean embedding of the `train` images, then
L2-normalized.

Only depends on numpy, torch, torchvision and Pillow: it can be copied or imported
from any script or notebook.

Example:
    import sys; sys.path.append("path/to/repo/common")
    from protocol import TRANSFORM, evaluate, print_results

    model.eval()
    results = evaluate(model, data_root="path/to/ds_FSL_224x204", device="cuda")
    print_results(results)
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

# Many original files are truncated progressive JPEGs: without this Pillow raises
# "image file is truncated" instead of decoding them.
ImageFile.LOAD_TRUNCATED_IMAGES = True

EPISODES_FILE = Path(__file__).resolve().with_name("test_episodes.json")
IMAGE_SIZE = (224, 204)  # (width, height)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class Transform:
    """The only preprocessing allowed, for training and evaluation alike.

    Same computation as torchvision's
    `Compose([ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])`, without needing
    torchvision. Works as the `transform` of a torchvision `ImageFolder`.
    """

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        pixels = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).copy())
        tensor = pixels.permute(2, 0, 1).float().div(255)  # ToTensor: (3, H, W) in [0, 1]
        return (tensor - self.mean) / self.std  # Normalize

    def __repr__(self) -> str:
        return f"Transform(ToTensor + Normalize(mean={IMAGENET_MEAN}, std={IMAGENET_STD}))"


TRANSFORM = Transform()


# --------------------------------------------------------------------------- images

def load_image(path: str | Path) -> torch.Tensor:
    """One image -> normalized tensor (3, 204, 224). Fails loudly instead of returning a blank image."""
    with Image.open(path) as image:
        if image.size != IMAGE_SIZE:
            raise ValueError(
                f"{path} is {image.size[0]}x{image.size[1]}, expected {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}: "
                "use the ds_FSL_224x204 dataset, without resizing."
            )
        return TRANSFORM(image.convert("RGB"))


def list_split(data_root: str | Path, split: str) -> list[Path]:
    """Every image of `data_root/split/<class>/`, in sorted order."""
    split_dir = Path(data_root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Folder not found: {split_dir}")
    return sorted(
        f for class_dir in split_dir.iterdir() if class_dir.is_dir()
        for f in class_dir.iterdir() if f.suffix.lower() in EXTENSIONS
    )


@torch.no_grad()
def embed_paths(
    embed: Callable[[torch.Tensor], torch.Tensor],
    paths: list[Path],
    device: str = "cpu",
    batch_size: int = 64,
) -> torch.Tensor:
    """Embeddings (n, d) of `paths`, computed by `embed` batch by batch, returned on CPU."""
    chunks = []
    with ThreadPoolExecutor() as pool:
        for start in range(0, len(paths), batch_size):
            batch = torch.stack(list(pool.map(load_image, paths[start : start + batch_size])))
            output = embed(batch.to(device))
            chunks.append(output.flatten(1).float().cpu())
    return torch.cat(chunks)


# --------------------------------------------------------------------------- episodes

def load_episodes(path: str | Path = EPISODES_FILE) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def episode_paths(data_root: str | Path, episodes: dict) -> list[Path]:
    """Absolute paths of the images listed in the episode file, in file order."""
    split_dir = Path(data_root) / episodes["protocol"]["split"]
    return [split_dir / unicodedata.normalize("NFC", image["path"]) for image in episodes["images"]]


def check_dataset(data_root: str | Path, episodes: dict) -> None:
    """Refuses a dataset copy that differs from the one the episodes were drawn from."""
    problems = []
    for path, image in zip(episode_paths(data_root, episodes), episodes["images"]):
        if not path.is_file():
            problems.append(f"missing: {image['path']}")
        elif hashlib.md5(path.read_bytes()).hexdigest() != image["md5"]:
            problems.append(f"different content: {image['path']}")
    if problems:
        shown = "\n  ".join(problems[:10])
        raise ValueError(
            f"{len(problems)} image(s) do not match the episode file "
            f"(is `data_root` the ds_FSL_224x204 folder from the Drive?):\n  {shown}"
        )


def cl2n(embeddings: torch.Tensor, train_mean: torch.Tensor) -> torch.Tensor:
    """CL2N (SimpleShot, Wang et al. 2019): center on the train mean, then L2-normalize."""
    return F.normalize(embeddings - train_mean, p=2, dim=1)


def episode_accuracies(embeddings: torch.Tensor, episodes: dict, n_shot: int, chunk: int = 100) -> np.ndarray:
    """Accuracy of each episode with `n_shot` support images per class, nearest prototype."""
    n_query = episodes["protocol"]["n_query"]
    indices = torch.tensor([episode["images"] for episode in episodes["episodes"]])  # (E, n_way, per_class)
    n_way = indices.shape[1]
    targets = torch.arange(n_way).repeat_interleave(n_query)  # queries are grouped by class
    accuracies = []
    for start in range(0, len(indices), chunk):
        block = embeddings[indices[start : start + chunk]]  # (e, n_way, per_class, d)
        prototypes = block[:, :, :n_shot].mean(dim=2)  # (e, n_way, d)
        queries = block[:, :, -n_query:].flatten(1, 2)  # (e, n_way * n_query, d)
        distances = torch.cdist(queries, prototypes).pow(2)  # squared Euclidean
        correct = distances.argmin(dim=2) == targets
        accuracies.append(correct.float().mean(dim=1))
    return torch.cat(accuracies).numpy()


def summarize(accuracies: np.ndarray) -> dict:
    """Mean accuracy and 95% CI, in percent."""
    return {
        "accuracy": float(100 * accuracies.mean()),
        "ci95": float(100 * 1.96 * accuracies.std(ddof=1) / math.sqrt(len(accuracies))),
    }


# --------------------------------------------------------------------------- entry point

def evaluate(
    embed: Callable[[torch.Tensor], torch.Tensor],
    data_root: str | Path,
    device: str = "cpu",
    batch_size: int = 64,
    episodes_file: str | Path = EPISODES_FILE,
    check: bool = True,
) -> dict:
    """Runs the shared protocol and returns the four scores.

    embed     : function from a normalized batch (n, 3, 204, 224) on `device` to embeddings (n, d).
                Usually the model itself, already in `eval()` mode. A model that returns several
                outputs needs a small wrapper, e.g. `lambda x: model(x)[0]`.
    data_root : the ds_FSL_224x204 folder (the one containing train/, valid/, test/).
    """
    episodes = load_episodes(episodes_file)
    if check:
        check_dataset(data_root, episodes)

    test = embed_paths(embed, episode_paths(data_root, episodes), device, batch_size)
    train_mean = embed_paths(embed, list_split(data_root, "train"), device, batch_size).mean(dim=0, keepdim=True)
    test_cl2n = cl2n(test, train_mean)

    results = {"n_episodes": len(episodes["episodes"]), "n_way": episodes["protocol"]["n_way"]}
    for n_shot in episodes["protocol"]["shots"]:
        results[f"{n_shot}shot"] = summarize(episode_accuracies(test, episodes, n_shot))
        results[f"{n_shot}shot_cl2n"] = summarize(episode_accuracies(test_cl2n, episodes, n_shot))
    return results


def print_results(results: dict, title: str = "") -> None:
    header = f"{title} - " if title else ""
    print(f"{header}{results['n_way']}-way, {results['n_episodes']} shared test episodes")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"  {key:<12} {value['accuracy']:6.2f} +/- {value['ci95']:.2f} %")
