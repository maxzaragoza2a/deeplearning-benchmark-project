"""Per-species accuracy of a harmonized run on the 1000 shared test episodes.

For every species of the shared test pool (46 species with at least 6 test images), the
share of its queries classified correctly, over all the shared episodes it appears in
(nearest prototype, squared Euclidean distance, raw embeddings: the main evaluation).
Each species is a query about 100 times, so a per-species accuracy carries a 95% CI of
roughly +/- 8 points: compare species only when their gap is larger.

Outputs, in figures/insights/<run>/:
- species_accuracy_1shot.png and species_accuracy_5shot.png: the 10 best and 10 worst
  species (same layout as the ProtoNet slide), with the overall mean and chance;
- species_accuracy.csv: every species, 1-shot and 5-shot, with its number of queries.

Embeddings come from the cache of scripts/plot_embedding_map.py (computed if missing).

Examples:
    python scripts/plot_species_accuracy.py
    python scripts/plot_species_accuracy.py --config configs/harmonized/prwn_resnet12.yaml --label "PRWN + ResNet-12 (AdamW)"
"""

import argparse
import csv
import math
import sys
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from common.protocol import load_episodes  # noqa: E402
from plot_embedding_map import INK_2, INK_3, PALETTE, short, split_embeddings  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def species_accuracy(embeddings: torch.Tensor, cached_paths: list[str], episodes: dict, n_shot: int) -> dict:
    """{species: (correct, total)} over the shared episodes, nearest prototype on raw embeddings."""
    position = {nfc(path): k for k, path in enumerate(cached_paths)}
    split = episodes["protocol"]["split"]
    image_rows = [position[nfc(f"{split}/{image['path']}")] for image in episodes["images"]]
    n_query = episodes["protocol"]["n_query"]
    tally: dict[str, list[int]] = {}
    for episode in episodes["episodes"]:
        rows = [[image_rows[i] for i in picks] for picks in episode["images"]]
        block = embeddings[torch.tensor(rows)]  # (n_way, per_class, d)
        prototypes = block[:, :n_shot].mean(dim=1)
        queries = block[:, -n_query:].reshape(-1, block.shape[-1])
        predicted = torch.cdist(queries, prototypes).argmin(dim=1).tolist()
        for q, guess in enumerate(predicted):
            name = episode["classes"][q // n_query]
            counts = tally.setdefault(name, [0, 0])
            counts[0] += int(guess == q // n_query)
            counts[1] += 1
    return tally


def draw(accuracy: dict[str, float], totals: dict[str, int], overall: float, n_shot: int,
         label: str, out: Path, n_side: int = 10) -> None:
    ranked = sorted(accuracy, key=lambda s: (-accuracy[s], s))
    best, worst = ranked[:n_side], ranked[-n_side:]
    shown = best + worst  # top to bottom
    values = [100 * accuracy[s] for s in shown]
    colors = [PALETTE[0]] * len(best) + [PALETTE[1]] * len(worst)
    y = np.arange(len(shown))[::-1].astype(float)
    y[len(best):] -= 0.8  # small gap between the two groups

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(y, values, height=0.72, color=colors, edgecolor="#fcfcfb", linewidth=1)
    for yy, value in zip(y, values):
        ax.text(value + 0.8, yy, f"{value:.0f}", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{short(s)}  (n={totals[s]})" for s in shown], fontsize=8, color=INK_2)
    ax.axvline(100 * overall, color=INK_2, linewidth=1.2, linestyle="--")
    ax.text(100 * overall - 0.8, y.max() + 0.9, f"all species {100 * overall:.1f}%", ha="right", fontsize=7.5,
            color=INK_2)
    ax.axvline(20, color=INK_3, linewidth=1, linestyle=":")
    ax.text(20.8, y.max() + 0.9, "chance 20%", ha="left", fontsize=7.5, color=INK_3)
    ax.set_xlim(0, 105)
    ax.set_ylim(y.min() - 0.8, y.max() + 1.6)
    ax.set_xlabel(f"Accuracy on the species' queries, 5-way {n_shot}-shot (%)")
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"{label}: per-species accuracy, 5-way {n_shot}-shot", pad=34)
    ax.text(0, 1.02, f"The {len(best)} best (blue) and {len(worst)} worst (orange) of the {len(accuracy)} species "
            "of the shared test pool, over the 1000 shared episodes.\nn = number of queries of the species; "
            f"with n of about 100, a per-species accuracy is known to about +/- 8 points.",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "harmonized" / "prwn_resnet12_sgd.yaml")
    p.add_argument("--label", default="PRWN + ResNet-12 (SGD)", help="model name in the title")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), help="overrides device")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-dir", type=Path, help="default: figures/insights/<run_name without 'harmonized_'>/")
    args = p.parse_args()

    config = load_config(args.config, {"device": args.device} if args.device else None)
    device = config.resolve_device()
    embeddings, _ = split_embeddings(config, config.run_dir, device, args.batch_size, "test")
    cached_paths = torch.load(config.run_dir / "test_embeddings.pt")["paths"]
    episodes = load_episodes()

    out_dir = args.out_dir or ROOT / "figures" / "insights" / config.run_name.removeprefix("harmonized_")
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {}
    for n_shot in episodes["protocol"]["shots"]:
        tally = species_accuracy(embeddings, cached_paths, episodes, n_shot)
        accuracy = {s: c / t for s, (c, t) in tally.items()}
        totals = {s: t for s, (_, t) in tally.items()}
        overall = sum(c for c, _ in tally.values()) / sum(t for _, t in tally.values())
        draw(accuracy, totals, overall, n_shot, args.label, out_dir / f"species_accuracy_{n_shot}shot.png")
        for s in tally:
            table.setdefault(s, {"species": s, "genus": s.split()[0], "queries": totals[s]})
            table[s][f"accuracy_{n_shot}shot"] = round(100 * accuracy[s], 1)
        print(f"{n_shot}-shot: overall {100 * overall:.2f}% (matches the shared test score)")

    rows = sorted(table.values(), key=lambda r: -r["accuracy_1shot"])
    with open(out_dir / "species_accuracy.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ci = 1.96 * math.sqrt(0.8 * 0.2 / 100) * 100
    print(f"{len(rows)} species; a per-species accuracy near 80% over ~100 queries has a 95% CI of +/- {ci:.0f} points")
    print(f"Written to {out_dir}")


if __name__ == "__main__":
    main()
