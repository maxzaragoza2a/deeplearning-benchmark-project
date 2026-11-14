"""Train PRWN (or its ProtoNet control) under the group's shared protocol.

Backbone chosen by `model` in the config, trained from scratch:
- `resnet12`: the same ResNet-12 as the NCA run (main runs);
- `conv4`: the Conv-4 of Snell et al. (secondary runs).

Shared protocol (common/README.md):
- original folders of ds_FSL_224x204: `train` to learn, `valid` to pick the best
  checkpoint, `test` used once at the end;
- images read as is at 224x204, ImageNet normalization, no augmentation;
- test on the 1000 frozen episodes of common/test_episodes.json, through
  `common.protocol.evaluate` (with and without CL2N).

PRWN with every label: the random walk needs points without labels, so it goes
through the query embeddings of the episode, whose labels it never sees. The ProtoNet
control (`lambda_rw: 0`) therefore trains on exactly the same episodes: the random-walk
loss is the only difference between the two runs.

Everything is written to `results/<run_name>/`.

Examples:
    python scripts/train_harmonized.py --config configs/harmonized/prwn_resnet12.yaml
    python scripts/train_harmonized.py --config configs/harmonized/prwn_resnet12_pn.yaml
    python scripts/train_harmonized.py --config configs/harmonized/prwn_conv4.yaml
    python scripts/train_harmonized.py --config configs/harmonized/smoke.yaml --skip-test   # quick CPU check
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.protocol import IMAGENET_MEAN, IMAGENET_STD, evaluate, print_results  # noqa: E402
from src.data.dataset import build_index, load_images  # noqa: E402
from src.data.episodes import EpisodeSampler  # noqa: E402
from src.models.conv4 import Conv4, ImagePreprocessor, embed_images  # noqa: E402
from src.models.prwn import compute_prototypes, prototype_logits, prwn_loss  # noqa: E402
from src.models.resnet12 import ResNet12  # noqa: E402
from src.utils.config import load_config  # noqa: E402

BACKBONES = ("resnet12", "conv4")


def build_backbone(config) -> nn.Module:
    """Backbone named by `config.model`, randomly initialized."""
    if config.model == "resnet12":
        return ResNet12()
    if config.model == "conv4":
        return Conv4(hidden_channels=config.conv4.hidden_channels, pool_size=config.conv4.pool_size)
    raise ValueError(f"Unknown model for the harmonized runs: {config.model} (expected one of {BACKBONES})")


def class_pools(labels: np.ndarray) -> dict[int, np.ndarray]:
    """{class: positions of its images} for one folder."""
    return {int(c): np.flatnonzero(labels == c) for c in np.unique(labels)}


def validate(embeddings: torch.Tensor, sampler: EpisodeSampler, n_episodes: int, seed: int, scale: float) -> dict:
    """Nearest-prototype accuracy on `n_episodes` valid episodes, always the same ones (fixed seed)."""
    rng = np.random.default_rng(seed)
    targets = torch.arange(sampler.n_way).repeat_interleave(sampler.n_query)
    scores = []
    for _ in range(n_episodes):
        episode = sampler.sample(rng)
        prototypes = compute_prototypes(embeddings[episode.support])
        predictions = prototype_logits(embeddings[episode.query], prototypes, scale).argmax(1)
        scores.append((predictions == targets).float().mean().item())
    scores = np.array(scores)
    return {
        "accuracy": float(100 * scores.mean()),
        "ci95": float(100 * 1.96 * scores.std(ddof=1) / math.sqrt(len(scores))),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="run config")
    p.add_argument("--model", choices=BACKBONES, help="overrides model")
    p.add_argument("--data-root", type=Path, help="overrides data_root")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), help="overrides device")
    p.add_argument("--results-dir", type=Path, help="overrides results_dir")
    p.add_argument("--skip-test", action="store_true", help="stop after training (quick checks only)")
    args = p.parse_args()

    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.data_root:
        overrides["data_root"] = str(args.data_root.resolve())
    if args.device:
        overrides["device"] = args.device
    if args.results_dir:
        overrides["results_dir"] = str(args.results_dir.resolve())
    config = load_config(args.config, overrides)
    cfg, device = config.train, config.resolve_device()
    if cfg.labeled_ratio < 1.0:
        sys.exit("The shared protocol trains with every label: set train.labeled_ratio to 1.0.")

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    torch.backends.cudnn.benchmark = True  # fixed input size: cuDNN picks its fastest algorithm

    # --- Data: one index per original folder, images read as is ---
    index = {
        part: build_index(config.data_root, [folder], dedup=config.dedup,
                          limit_per_class=config.limit_per_class, verbose=False)
        for part, folder in (("train", "train"), ("val", "valid"))
    }
    train_sampler = EpisodeSampler(
        class_pools(index["train"].labels), {}, cfg.n_way, cfg.k_shot, cfg.n_query, n_unlabeled=0,
    )
    val_sampler = EpisodeSampler(
        class_pools(index["val"].labels), {}, cfg.n_way, 1, 1, n_unlabeled=0,
    )
    print(
        f"Train: {len(index['train'])} images, {len(train_sampler.eligible)} classes usable in "
        f"{cfg.n_way}-way {cfg.k_shot}-shot {cfg.n_query}-query episodes | "
        f"Valid: {len(index['val'])} images, {len(val_sampler.eligible)} classes usable"
    )

    start = time.time()
    images = {part: load_images(idx, config.input_size, letterbox=False) for part, idx in index.items()}
    print(f"Images loaded in {time.time() - start:.0f} s: "
          + ", ".join(f"{part} {tuple(tensor.shape)}" for part, tensor in images.items()))
    train_images = images["train"].to(device)

    # --- Model. Preprocessing = the shared TRANSFORM (ImageNet normalization), no augmentation ---
    model = build_backbone(config).to(device)
    preprocess = ImagePreprocessor(torch.tensor(IMAGENET_MEAN), torch.tensor(IMAGENET_STD), hflip=False, max_shift=0)
    preprocess = preprocess.to(device)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    with torch.no_grad():
        dim = model.eval()(preprocess.eval()(train_images[:1])).shape[1]
    print(f"{config.model}: {n_params / 1e6:.2f} M parameters, embedding dimension {dim}, on {device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.n_steps)

    config.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.run_dir / "best_model.pt"
    use_rw = cfg.lambda_rw > 0

    # --- Episodic training ---
    history, best, window = [], {"accuracy": -1.0, "step": 0}, []
    start = time.time()
    for step in range(1, cfg.n_steps + 1):
        model.train()
        episode = train_sampler.sample(rng)
        order = np.concatenate([episode.support.ravel(), episode.query.ravel()])
        embeddings = model(preprocess(train_images[torch.as_tensor(order, device=device)]))
        n_support = episode.support.size
        query = embeddings[n_support:]
        loss, stats = prwn_loss(
            embeddings[:n_support].view(cfg.n_way, cfg.k_shot, -1),
            query.view(cfg.n_way, cfg.n_query, -1),
            query if use_rw else None,  # the walk sees the queries, never their labels
            cfg.lambda_rw, cfg.rw_steps, cfg.rw_decay, cfg.rw_smoothing, cfg.visit_weight,
            cfg.distance_scale,
        )
        if not math.isfinite(stats["loss"]):
            sys.exit(f"Non-finite loss at step {step}: {stats}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        window.append(stats)

        if step % cfg.eval_every == 0 or step == cfg.n_steps:
            val_embeddings = embed_images(model, preprocess, images["val"], device, config.extract_batch_size)
            val = validate(val_embeddings, val_sampler, cfg.n_val_episodes, config.seed, cfg.distance_scale)
            row = {"step": step, "seconds": round(time.time() - start)}
            row.update({f"train_{key}": float(np.mean([s[key] for s in window])) for key in window[0]})
            row.update({"val_accuracy": val["accuracy"], "val_ci95": val["ci95"]})
            history.append(row)
            window = []
            print(
                f"step {step:>6}/{cfg.n_steps} | loss {row['train_loss']:.3f} "
                f"| train acc {100 * row['train_accuracy']:.1f} % | val acc {val['accuracy']:.2f} "
                f"+/- {val['ci95']:.2f} % | {row['seconds']} s",
                flush=True,
            )
            if val["accuracy"] > best["accuracy"]:
                best = {"accuracy": val["accuracy"], "ci95": val["ci95"], "step": step}
                torch.save({"model": model.state_dict(), "step": step, "config": config.raw}, checkpoint)

    with open(config.run_dir / "history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    metrics = {
        "config": config.raw,
        "device": device,
        "method": "PRWN" if use_rw else "ProtoNet (PRWN code, lambda_rw = 0)",
        "backbone": config.model,
        "n_parameters": n_params,
        "embedding_dim": dim,
        "n_train_images": len(index["train"]),
        "n_train_classes_usable": len(train_sampler.eligible),
        "n_valid_classes_usable": len(val_sampler.eligible),
        "best_val": best,
    }

    # --- Test, once: best checkpoint on valid, shared episodes ---
    if args.skip_test:
        print("Test skipped (--skip-test).")
    else:
        del train_images, images
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
        model.eval()
        metrics["test"] = evaluate(model, config.data_root, device, config.extract_batch_size)
        print_results(metrics["test"], f"{config.run_name} (best valid step {best['step']})")

    with open(config.run_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(f"Results written to {config.run_dir}")


if __name__ == "__main__":
    main()
