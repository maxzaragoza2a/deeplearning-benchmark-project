"""Entraine et evalue un Prototypical Random Walk Network sur les ailes d'insectes.

L'embedding est DINOv2 pre-entraine et gele : ses embeddings sont calcules une fois
(et mis en cache), puis une petite tete est entrainee par-dessus avec la loss PRW.
La meilleure tete en validation est evaluee sur les classes de test, a cote du
backbone seul (sans tete) pour reference ; tout est ecrit dans `results/<run_name>/`.

Exemples :
    python scripts/train_prwn.py --config configs/prwn.yaml
    python scripts/train_prwn.py --config configs/prwn_pn.yaml      # controle sans marche aleatoire
    python scripts/train_prwn.py --config configs/prwn_smoke.yaml   # test rapide sur CPU
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import build_index, load_images  # noqa: E402
from src.data.episodes import EpisodeSampler, split_classes, split_labeled  # noqa: E402
from src.models.backbone import ProjectionHead, cached_embeddings  # noqa: E402
from src.models.prwn import compute_prototypes, prototype_logits, prwn_loss, refine_prototypes  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def evaluate(embeddings, labels, n_way, k_shot, n_query, n_unlabeled, n_episodes, seed, scale):
    """Evaluation episodique sur des embeddings deja calcules.

    Renvoie la precision au plus proche prototype, et celle apres raffinement des
    prototypes par soft k-means sur `n_unlabeled` images supplementaires par classe.
    Les episodes ne dependent que de la graine : identiques d'un run a l'autre.
    """
    pools = {int(c): np.flatnonzero(labels == c) for c in np.unique(labels)}
    sampler = EpisodeSampler(pools, pools, n_way, k_shot, n_query, n_unlabeled, shared_pool=True)
    rng = np.random.default_rng(seed)
    targets = torch.arange(n_way).repeat_interleave(n_query)

    plain, refined = [], []
    for _ in range(n_episodes):
        episode = sampler.sample(rng)
        support = embeddings[episode.support]
        query = embeddings[episode.query]
        prototypes = compute_prototypes(support)
        plain.append((prototype_logits(query, prototypes, scale).argmax(1) == targets).float().mean().item())
        if n_unlabeled:
            prototypes = refine_prototypes(prototypes, support, embeddings[episode.unlabeled], scale)
            refined.append((prototype_logits(query, prototypes, scale).argmax(1) == targets).float().mean().item())

    def summary(scores):
        scores = np.array(scores)
        # IC 95 % sur la moyenne des episodes : 1.96 * sigma / sqrt(n).
        return float(scores.mean()), float(1.96 * scores.std(ddof=1) / math.sqrt(len(scores)))

    result = {"n_eligible_classes": len(sampler.eligible), "n_episodes": n_episodes}
    result["accuracy"], result["ci95"] = summary(plain)
    if refined:
        result["accuracy_refined"], result["ci95_refined"] = summary(refined)
    return result


def print_test(title, results, n_way):
    for name, result in results.items():
        line = f"{title} {n_way}-way {name} : {result['accuracy']:.4f} +/- {result['ci95']:.4f}"
        if "accuracy_refined" in result:
            line += f" | avec soft k-means : {result['accuracy_refined']:.4f} +/- {result['ci95_refined']:.4f}"
        print(line + f" ({result['n_eligible_classes']} classes eligibles)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="fichier de config du run")
    p.add_argument("--data-root", type=Path, help="remplace data_root (ex. sur Colab)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), help="remplace device")
    p.add_argument("--results-dir", type=Path, help="remplace results_dir (ex. un dossier Drive)")
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

    # --- Donnees : index, split par classes, split etiquete / non etiquete ---
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

    # --- Embeddings du backbone gele, calcules une seule fois ---
    start = time.time()
    features = cached_embeddings(
        config.model, index.paths, config.input_size,
        lambda: load_images(index, config.input_size),
        device, config.cache_dir, config.extract_batch_size,
    ).to(device)
    print(f"Embeddings {config.model} : {tuple(features.shape)} en {time.time() - start:.0f} s")

    # --- Tete et optimisation ---
    head = ProjectionHead(features.shape[1], **vars(config.head)).to(device)
    n_params = sum(parameter.numel() for parameter in head.parameters())
    print(f"Tete : {n_params / 1e3:.0f} k parametres entraines, sur {device}")
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.n_steps)

    config.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.run_dir / "best_head.pt"

    def run_eval(part, k_shot, n_episodes, use_head=True):
        embeddings = features[positions[part]]
        if use_head:
            head.eval()
            with torch.no_grad():
                embeddings = head(embeddings)
        return evaluate(
            embeddings.cpu(), index.labels[positions[part]], eval_cfg.n_way, k_shot, eval_cfg.n_query,
            eval_cfg.n_unlabeled, n_episodes, config.seed, cfg.distance_scale,
        )

    # --- Entrainement episodique de la tete ---
    history, best, window = [], {"accuracy": -1.0, "step": 0}, []
    start = time.time()
    for step in range(1, cfg.n_steps + 1):
        head.train()
        episode = sampler.sample(rng)
        order = np.concatenate([episode.support.ravel(), episode.query.ravel(), episode.unlabeled])
        embeddings = head(features[order])
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
                torch.save(head.state_dict(), checkpoint)

    # --- Test : meilleure tete en validation, et backbone seul pour reference ---
    head.load_state_dict(torch.load(checkpoint, map_location=device))
    test = {f"{k}shot": run_eval("test", k, eval_cfg.n_episodes) for k in eval_cfg.k_shots}
    backbone_only = {
        f"{k}shot": run_eval("test", k, eval_cfg.n_episodes, use_head=False) for k in eval_cfg.k_shots
    }
    print_test("test", test, eval_cfg.n_way)
    print_test("backbone seul", backbone_only, eval_cfg.n_way)

    with open(config.run_dir / "history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metrics = {
        "config": config.raw,
        "device": device,
        "n_head_parameters": n_params,
        "classes": names,
        "n_train_eligible_classes": len(sampler.eligible),
        "n_labeled_images": int(labeled_mask.sum()),
        "n_unlabeled_images": len(positions["train"]) - int(labeled_mask.sum()),
        "excluded_duplicates": index.excluded_duplicates,
        "best_val": best,
        "test": test,
        "test_backbone_only": backbone_only,
    }
    with open(config.run_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(f"Resultats ecrits dans {config.run_dir}")


if __name__ == "__main__":
    main()
