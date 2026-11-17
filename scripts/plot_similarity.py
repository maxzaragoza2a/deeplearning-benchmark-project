"""Intra- vs inter-species similarity of the test embeddings of a harmonized run.

For every pair of test images, the cosine similarity of their embeddings after centering
on the mean `train` embedding (the centering of CL2N, as in common/protocol.py): from -1
(opposite) to 1 (same direction). Without centering, ResNet embeddings (after a ReLU) are
all positive and every similarity is squeezed close to 1.

Two figures, in figures/insights/<run>/:
- similarity.png: pairs of the same species vs pairs of different species (same layout as
  the ProtoNet slide, for a direct comparison);
- similarity_by_relatedness.png: different-species pairs split into same genus vs
  different genera, to check whether the overlap comes from close relatives.

Embeddings come from the cache of scripts/plot_embedding_map.py (computed if missing).

Examples:
    python scripts/plot_similarity.py
    python scripts/plot_similarity.py --config configs/harmonized/prwn_resnet12.yaml --label "PRWN + ResNet-12 (AdamW)"
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_embedding_map import INK_2, INK_3, PALETTE, split_embeddings  # noqa: E402  (also sets the style)
from src.utils.config import load_config  # noqa: E402

BINS = np.linspace(-1, 1, 81)


def pair_similarities(test: torch.Tensor, train_mean: torch.Tensor, species: list[str]) -> dict[str, np.ndarray]:
    """Centered cosine similarity of every unordered pair of test images, grouped by relatedness."""
    z = F.normalize(test - train_mean, dim=1)
    similarity = (z @ z.T).numpy()
    species = np.array(species)
    genus = np.array([s.split()[0] for s in species])
    i, j = np.triu_indices(len(species), k=1)
    sim = similarity[i, j]
    same_species = species[i] == species[j]
    same_genus = genus[i] == genus[j]
    return {
        "same species": sim[same_species],
        "different species": sim[~same_species],
        "different species, same genus": sim[~same_species & same_genus],
        "different genera": sim[~same_genus],
        "_same_genus_among_different": same_genus[~same_species],  # aligned with "different species"
    }


def draw(groups: dict[str, np.ndarray], colors: list[str], title: str, subtitle: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for (name, values), color in zip(groups.items(), colors):
        ax.hist(values, bins=BINS, density=True, color=color, alpha=0.45, linewidth=0,
                label=f"{name} ({len(values):,} pairs, mean {values.mean():.2f})")
        ax.hist(values, bins=BINS, density=True, histtype="step", color=color, linewidth=1.6)
        ax.axvline(values.mean(), color=color, linewidth=1.2, linestyle="--")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Cosine similarity of the centered embeddings (-1 = opposite, 1 = same direction)")
    ax.set_ylabel("Density")
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, pad=34)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")
    ax.legend(loc="upper left", labelcolor=INK_2)
    fig.text(0.125, -0.02, "Dashed lines: means. Each pair of test images counted once.",
             fontsize=7.5, color=INK_3)
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
    test, species = split_embeddings(config, config.run_dir, device, args.batch_size, "test")
    train, _ = split_embeddings(config, config.run_dir, device, args.batch_size, "train")
    groups = pair_similarities(test, train.mean(dim=0, keepdim=True), species)

    same, diff = groups["same species"], groups["different species"]
    close, far = groups["different species, same genus"], groups["different genera"]
    threshold = np.median(same)
    above = diff > threshold  # different-species pairs as similar as a typical same-species pair
    share_close = 100 * groups["_same_genus_among_different"][above].mean() if above.any() else 0.0

    out_dir = args.out_dir or ROOT / "figures" / "insights" / config.run_name.removeprefix("harmonized_")
    out_dir.mkdir(parents=True, exist_ok=True)

    draw({"same species": same, "different species": diff}, PALETTE[:2],
         f"{args.label}: intra- vs inter-species similarity",
         f"Mean similarity: same species {same.mean():.2f}, different species {diff.mean():.2f}, gap "
         f"{same.mean() - diff.mean():.2f}.\n{100 * above.mean():.1f}% of different-species pairs are more "
         f"similar than the median same-species pair ({threshold:.2f}).",
         out_dir / "similarity.png")

    draw({"same species": same, "different species, same genus": close, "different genera": far},
         [PALETTE[0], PALETTE[1], PALETTE[2]],
         f"{args.label}: similarity by relatedness",
         f"Mean similarity: same species {same.mean():.2f}, different species of the same genus "
         f"{close.mean():.2f}, different genera {far.mean():.2f}.\nAmong the different-species pairs more "
         f"similar than the median same-species pair, {share_close:.0f}% are of the same genus.",
         out_dir / "similarity_by_relatedness.png")

    print(f"Same species {same.mean():.3f} | different species {diff.mean():.3f} "
          f"(same genus {close.mean():.3f}, different genera {far.mean():.3f})")
    print(f"Different-species pairs above the median same-species similarity: {100 * above.mean():.2f}% "
          f"({share_close:.0f}% of them same genus)")
    print(f"Written to {out_dir}")


if __name__ == "__main__":
    main()
