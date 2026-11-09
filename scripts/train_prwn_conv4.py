"""Entraine et evalue un PRWN dont l'embedding est un Conv-4 entraine de zero.

Variante de `train_prwn.py` : au lieu d'une tete sur DINOv2 gele, c'est tout le
backbone Conv-4 (src/models/conv4.py) qui apprend avec la loss PRW. Meme split par
classes, memes episodes d'entrainement et de test : les scores se comparent
directement a ceux des runs DINOv2.

Les images sont chargees une fois en memoire (uint8, letterbox), puis normalisees et
augmentees a la volee. Le meilleur modele en validation est evalue sur les classes
de test ; tout est ecrit dans `results/<run_name>/`.

Exemples :
    python scripts/train_prwn_conv4.py --config configs/prwn_conv4.yaml
    python scripts/train_prwn_conv4.py --config configs/prwn_conv4_pn.yaml     # controle sans marche aleatoire
    python scripts/train_prwn_conv4.py --config configs/prwn_conv4_smoke.yaml  # test rapide sur CPU
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.dataset import build_index, load_images  # noqa: E402
from src.data.episodes import EpisodeSampler, split_classes, split_labeled  # noqa: E402
from src.models.conv4 import Conv4, ImagePreprocessor, channel_stats, embed_images  # noqa: E402
from src.models.prwn import prwn_loss  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from train_prwn import evaluate, print_test  # noqa: E402  (meme evaluation que le run DINOv2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="fichier de config du run")
    p.add_argument("--data-root", type=Path, help="remplace data_root")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), help="remplace device")
    p.add_argument("--results-dir", type=Path, help="remplace results_dir")
    args = p.parse_args()

    overrides = {}
    if args.data_root:
        overrides["data_root"] = str(args.data_root.resolve())
    if args.device:
        overrides["device"] = args.device
    if args.results_dir:
        overrides["results_dir"] = str(args.results_dir.resolve())
    config = load_config(args.config, overrides)
    cfg, eval_cfg, device = config.train, config.eval, config.resolve_device()

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    torch.backends.cudnn.benchmark = True  # taille d'entree fixe : cuDNN choisit l'algo le plus rapide

    # --- Donnees : index, split par classes, split etiquete / non etiquete (comme train_prwn.py) ---
    index = build_index(config.data_root, config.splits, config.dedup, config.limit_per_class)
    counts = {name: int((index.labels == i).sum()) for i, name in enumerate(index.class_names)}
    split = config.class_split
    names = dict(zip(("train", "val", "test"), split_classes(
        counts, split.n_val_classes, split.n_test_classes, split.n_bins, split.seed
    )))
    label_of = {name: i for i, name in enumerate(index.class_names)}
    positions = {
        part: np.flatnonzero(np.isin(index.labels, [label_of[name] for name in classes]))
        for part, classes in names.items()
    }

    labeled_mask = np.zeros(len(index), dtype=bool)
    labeled_mask[positions["train"]] = split_labeled(
        index.labels[positions["train"]], cfg.labeled_ratio, config.seed
    )
    train_labels = [label_of[name] for name in names["train"]]
    labeled = {c: np.flatnonzero((index.labels == c) & labeled_mask) for c in train_labels}
    unlabeled = {c: np.flatnonzero((index.labels == c) & ~labeled_mask) for c in train_labels}

    use_rw = cfg.lambda_rw > 0
    sampler = EpisodeSampler(
        labeled, unlabeled, cfg.n_way, cfg.k_shot, cfg.n_query,
        cfg.n_unlabeled if use_rw else 0, cfg.n_distractor_classes,
    )
    print(
        f"Classes : {len(names['train'])} train ({len(sampler.eligible)} eligibles aux episodes), "
        f"{len(names['val'])} val, {len(names['test'])} test"
    )
    print(
        f"Images d'entrainement : {int(labeled_mask.sum())} etiquetees, "
        f"{len(positions['train']) - int(labeled_mask.sum())} non etiquetees"
    )

    # --- Images en memoire, sur le GPU si possible (~150 Ko par image en 224x224) ---
    start = time.time()
    images = load_images(index, config.input_size)
    print(f"Images {tuple(images.shape)} chargees en {time.time() - start:.0f} s")
    # Normalisation : statistiques des seules images d'entrainement (etiquetees ou non).
    mean, std = channel_stats(images[positions["train"]])
    images = images.to(device)

    # --- Modele et optimisation ---
    conv = config.conv4
    model = Conv4(hidden_channels=conv.hidden_channels, pool_size=conv.pool_size).to(device)
    preprocess = ImagePreprocessor(mean, std, conv.hflip, conv.max_shift).to(device)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    with torch.no_grad():
        dim = model.eval()(preprocess.eval()(images[:1])).shape[1]
    print(f"Conv-4 : {n_params / 1e3:.0f} k parametres, embedding de dimension {dim}, sur {device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.n_steps)

    config.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.run_dir / "best_model.pt"

    def run_eval(part, k_shot, n_episodes):
        embeddings = embed_images(model, preprocess, images[positions[part]], device, config.extract_batch_size)
        return evaluate(
            embeddings, index.labels[positions[part]], eval_cfg.n_way, k_shot, eval_cfg.n_query,
            eval_cfg.n_unlabeled, n_episodes, config.seed, cfg.distance_scale,
        )

    # --- Entrainement episodique de bout en bout ---
    history, best, window = [], {"accuracy": -1.0, "step": 0}, []
    start = time.time()
    for step in range(1, cfg.n_steps + 1):
        model.train()
        preprocess.train()
        episode = sampler.sample(rng)
        order = np.concatenate([episode.support.ravel(), episode.query.ravel(), episode.unlabeled])
        batch = images[torch.as_tensor(order, device=images.device)]
        embeddings = model(preprocess(batch))
        n_support, n_query = episode.support.size, episode.query.size
        loss, stats = prwn_loss(
            embeddings[:n_support].view(cfg.n_way, cfg.k_shot, -1),
            embeddings[n_support : n_support + n_query].view(cfg.n_way, cfg.n_query, -1),
            embeddings[n_support + n_query :] if use_rw else None,
            cfg.lambda_rw, cfg.rw_steps, cfg.rw_decay, cfg.rw_smoothing, cfg.visit_weight,
            cfg.distance_scale,
        )
        if not math.isfinite(stats["loss"]):
            sys.exit(f"Loss non finie au pas {step} : {stats}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        window.append(stats)

        if step % cfg.eval_every == 0 or step == cfg.n_steps:
            val = run_eval("val", cfg.k_shot, cfg.n_val_episodes)
            row = {"step": step, "seconds": round(time.time() - start)}
            row.update({f"train_{key}": float(np.mean([s[key] for s in window])) for key in window[0]})
            row.update({"val_accuracy": val["accuracy"], "val_ci95": val["ci95"]})
            history.append(row)
            window = []
            print(
                f"pas {step:>6}/{cfg.n_steps} | loss {row['train_loss']:.3f} "
                f"| acc train {row['train_accuracy']:.3f} | acc val {val['accuracy']:.3f} "
                f"+/- {val['ci95']:.3f} | {row['seconds']} s",
                flush=True,
            )
            if val["accuracy"] > best["accuracy"]:
                best = {"accuracy": val["accuracy"], "step": step}
                torch.save({"model": model.state_dict(), "preprocess": preprocess.state_dict()}, checkpoint)

    # --- Test : meilleur modele en validation ---
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    preprocess.load_state_dict(state["preprocess"])
    test = {f"{k}shot": run_eval("test", k, eval_cfg.n_episodes) for k in eval_cfg.k_shots}
    print_test("test", test, eval_cfg.n_way)

    with open(config.run_dir / "history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metrics = {
        "config": config.raw,
        "device": device,
        "backbone": "conv4",
        "n_parameters": n_params,
        "embedding_dim": dim,
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "classes": names,
        "n_train_eligible_classes": len(sampler.eligible),
        "n_labeled_images": int(labeled_mask.sum()),
        "n_unlabeled_images": len(positions["train"]) - int(labeled_mask.sum()),
        "excluded_duplicates": index.excluded_duplicates,
        "best_val": best,
        "test": test,
    }
    with open(config.run_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(f"Resultats ecrits dans {config.run_dir}")


if __name__ == "__main__":
    main()
