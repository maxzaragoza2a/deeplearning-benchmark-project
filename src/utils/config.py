"""Chargement de la configuration YAML vers des dataclasses typees.

Un fichier de config par run. Un fichier peut heriter d'un autre avec la cle
`base` (chemin relatif au fichier) : seules les valeurs qui changent sont reecrites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

# Racine du depot = deux niveaux au-dessus de src/utils/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ClassSplitConfig:
    """Split par classes (entrainement / validation / test)."""

    n_val_classes: int = 20
    n_test_classes: int = 39
    n_bins: int = 5
    seed: int = 0


@dataclass
class HeadConfig:
    """Tete entrainee sur les embeddings geles ; `hidden_dim: null` = tete lineaire."""

    out_dim: int = 384
    hidden_dim: Optional[int] = None
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """Episodes d'entrainement semi-supervises, loss PRW et optimisation de la tete."""

    n_way: int = 5
    k_shot: int = 1
    n_query: int = 5
    n_unlabeled: int = 5
    n_distractor_classes: int = 0
    labeled_ratio: float = 0.4
    lambda_rw: float = 0.5
    rw_steps: int = 3
    rw_decay: float = 0.7
    rw_smoothing: float = 0.1
    visit_weight: float = 1.0
    distance_scale: float = 1.0
    n_steps: int = 20000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eval_every: int = 500
    n_val_episodes: int = 300


@dataclass
class EvalConfig:
    """Evaluation episodique sur les classes de validation et de test."""

    n_way: int = 5
    k_shots: List[int] = field(default_factory=lambda: [1, 5])
    n_query: int = 10
    n_unlabeled: int = 5
    n_episodes: int = 600


@dataclass
class Config:
    """Configuration complete d'un run."""

    run_name: str
    data_root: Path
    splits: List[str]
    dedup: bool
    limit_per_class: Optional[int]
    model: str
    input_size: tuple[int, int]  # (largeur, hauteur)
    extract_batch_size: int
    device: str
    seed: int
    results_dir: Path
    class_split: ClassSplitConfig
    head: HeadConfig
    train: TrainConfig
    eval: EvalConfig
    raw: dict  # contenu YAML fusionne, recopie tel quel dans les resultats

    def resolve_device(self) -> str:
        """Resout `device: auto` en `cuda` si disponible, sinon `cpu`."""
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def run_dir(self) -> Path:
        return self.results_dir / self.run_name

    @property
    def cache_dir(self) -> Path:
        """Embeddings du backbone gele, partages par tous les runs."""
        return self.results_dir / "embeddings"


def _as_project_path(value: str | Path) -> Path:
    """Transforme un chemin de config en Path absolu (relatif = relatif au depot)."""
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _merge(base: dict, override: dict) -> dict:
    """Fusion recursive : `override` l'emporte, les sous-dictionnaires sont fusionnes."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict:
    """Lit un YAML en resolvant la chaine d'heritage `base`."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration introuvable : {path}")
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    base = raw.pop("base", None)
    if base is None:
        return raw
    return _merge(_read_yaml(path.parent / base), raw)


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    """Charge un fichier de config ; `overrides` (ligne de commande) l'emporte."""
    raw = _merge(_read_yaml(Path(path)), overrides or {})

    data_root = _as_project_path(raw["data_root"])
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"data_root introuvable : {data_root}\n"
            "Verifie `data_root` dans la config ou passe --data-root."
        )

    width, height = (int(v) for v in raw.get("input_size", [224, 224]))
    limit = raw.get("limit_per_class")

    return Config(
        run_name=str(raw.get("run_name", "run")),
        data_root=data_root,
        splits=list(raw.get("splits", ["train", "valid", "test"])),
        dedup=bool(raw.get("dedup", True)),
        limit_per_class=None if limit is None else int(limit),
        model=str(raw.get("model", "dinov2_s")),
        input_size=(width, height),
        extract_batch_size=int(raw.get("extract_batch_size", 64)),
        device=str(raw.get("device", "auto")),
        seed=int(raw.get("seed", 0)),
        results_dir=_as_project_path(raw.get("results_dir", "results")),
        class_split=ClassSplitConfig(**raw.get("class_split", {})),
        head=HeadConfig(**raw.get("head", {})),
        train=TrainConfig(**raw.get("train", {})),
        eval=EvalConfig(**raw.get("eval", {})),
        raw=raw,
    )
