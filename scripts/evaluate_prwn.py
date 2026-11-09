"""Evaluate an already trained PRWN head on the test classes, without retraining.

Reloads `results/<run_name>/best_head.pt`, rebuilds the same class split as
training (same seeds) and runs the test episodes, next to the backbone alone
as a reference.

Examples:
    python scripts/evaluate_prwn.py --config configs/prwn.yaml
    python scripts/evaluate_prwn.py --config configs/prwn.yaml --n-episodes 2000 --seed 1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_prwn import evaluate, print_test  # noqa: E402
from src.data.dataset import build_index, load_images  # noqa: E402
from src.data.episodes import split_classes  # noqa: E402
from src.models.backbone import ProjectionHead, cached_embeddings  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="config of the trained run")
    p.add_argument("--checkpoint", type=Path, help="head to evaluate (default: results/<run_name>/best_head.pt)")
    p.add_argument("--n-episodes", type=int, help="overrides eval.n_episodes")
    p.add_argument("--seed", type=int, help="seed of the test episodes (default: seed from the config)")
    p.add_argument("--data-root", type=Path, help="overrides data_root")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), help="overrides device")
    args = p.parse_args()

    overrides = {}
    if args.data_root:
        overrides["data_root"] = str(args.data_root.resolve())
    if args.device:
        overrides["device"] = args.device
    config = load_config(args.config, overrides)
    eval_cfg, device = config.eval, config.resolve_device()
    n_episodes = args.n_episodes or eval_cfg.n_episodes
    seed = config.seed if args.seed is None else args.seed
    checkpoint = args.checkpoint or config.run_dir / "best_head.pt"
    if not checkpoint.exists():
        sys.exit(f"Head not found: {checkpoint} (run train_prwn.py first)")

    # Same index and same class split as during training.
    index = build_index(config.data_root, config.splits, config.dedup, config.limit_per_class)
    counts = {name: int((index.labels == i).sum()) for i, name in enumerate(index.class_names)}
    split = config.class_split
    _, _, test_names = split_classes(counts, split.n_val_classes, split.n_test_classes, split.n_bins, split.seed)
    label_of = {name: i for i, name in enumerate(index.class_names)}
    test_positions = np.flatnonzero(np.isin(index.labels, [label_of[name] for name in test_names]))
    test_labels = index.labels[test_positions]
    print(f"Test classes: {len(test_names)} | images: {len(test_positions)}")

    features = cached_embeddings(
        config.model, index.paths, config.input_size,
        lambda: load_images(index, config.input_size),
        device, config.cache_dir, config.extract_batch_size,
    )[test_positions]

    head = ProjectionHead(features.shape[1], **vars(config.head))
    head.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    head.eval()
    with torch.no_grad():
        projected = head(features)
    print(f"Head loaded from {checkpoint}")

    def run(embeddings, k):
        return evaluate(
            embeddings, test_labels, eval_cfg.n_way, k, eval_cfg.n_query, eval_cfg.n_unlabeled,
            n_episodes, seed, config.train.distance_scale,
        )

    test = {f"{k}shot": run(projected, k) for k in eval_cfg.k_shots}
    backbone_only = {f"{k}shot": run(features, k) for k in eval_cfg.k_shots}
    print_test("test", test, eval_cfg.n_way)
    print_test("backbone only", backbone_only, eval_cfg.n_way)

    out = config.run_dir / f"test_seed{seed}_{n_episodes}ep.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(
            {"checkpoint": str(checkpoint), "seed": seed, "n_episodes": n_episodes,
             "test_classes": test_names, "test": test, "test_backbone_only": backbone_only},
            handle, indent=2, ensure_ascii=False,
        )
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
