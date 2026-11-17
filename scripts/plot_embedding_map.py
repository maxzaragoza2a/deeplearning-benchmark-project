"""2D maps (t-SNE) of the test embeddings of a harmonized run, colored by genus and by species.

Loads `results/<run>/best_model.pt`, embeds every image of the `test` folder with the
shared preprocessing (common/protocol.py), and draws one t-SNE map, highlighted in two
figures (highlighted groups in color, every other test image in grey): `tsne_genus.png` (the 7
most represented genera) and `tsne_species.png` (the 8 most represented species among the
shared test pool, at least 6 images). The embeddings are cached in
`results/<run>/test_embeddings.pt`, so re-plotting is instant.

Each subtitle gives how often the nearest neighbour of a test image, in the 640-d space the
classifier actually uses, is of the same genus or species: a number that does not depend on
the t-SNE picture, which only preserves neighborhoods.

Examples:
    python scripts/plot_embedding_map.py
    python scripts/plot_embedding_map.py --config configs/harmonized/prwn_resnet12.yaml --label "PRWN + ResNet-12 (AdamW)"
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from common.protocol import embed_paths, list_split  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from train_harmonized import build_backbone  # noqa: E402

# Same style as scripts/analyze_results.py (reference categorical palette, fixed order).
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
OTHER = "#c9c8c3"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "legend.frameon": False, "legend.fontsize": 8, "text.color": INK, "font.size": 9,
})


def short(name: str) -> str:
    """Species name as shown on figures: the full name. Abbreviating the genus is ambiguous
    here (A. = Anopheles or Aedes, C. = Culex or Culicoides)."""
    return name


def split_embeddings(config, run_dir: Path, device: str, batch_size: int,
                     split: str = "test") -> tuple[torch.Tensor, list[str]]:
    """Embeddings (n, d) of every image of `split` and their species, cached next to the checkpoint."""
    paths = list_split(config.data_root, split)
    relative = [p.relative_to(config.data_root).as_posix() for p in paths]
    cache = run_dir / f"{split}_embeddings.pt"
    if cache.exists():
        cached = torch.load(cache)
        if cached["paths"] == relative:
            print(f"Embeddings read from {cache}")
            return cached["embeddings"], cached["species"]

    state = torch.load(run_dir / "best_model.pt", map_location=device)
    config.model = state.get("config", {}).get("model", config.model)  # the checkpoint knows its backbone
    model = build_backbone(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Embedding {len(paths)} {split} images with the step-{state.get('step', '?')} checkpoint on {device}...")
    embeddings = embed_paths(model, paths, device, batch_size)
    species = [p.parent.name for p in paths]
    torch.save({"embeddings": embeddings, "species": species, "paths": relative}, cache)
    return embeddings, species


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
    embeddings, species = split_embeddings(config, config.run_dir, device, args.batch_size)

    species = np.array(species)
    genus = np.array([s.split()[0] for s in species])
    species_count = Counter(species)

    # Nearest neighbour in the 640-d space (the one the classifier uses), each image excluded
    # from its own search. Only images whose species has at least 2 test images count.
    distances = torch.cdist(embeddings, embeddings)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.argmin(dim=1).numpy()
    counted = np.array([species_count[s] >= 2 for s in species])
    same_species = 100 * (species[nearest] == species)[counted].mean()
    same_genus = 100 * (genus[nearest] == genus)[counted].mean()
    n_species_counted = len({s for s in species if species_count[s] >= 2})

    print("t-SNE...")
    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(embeddings.numpy())

    out_dir = args.out_dir or ROOT / "figures" / "insights" / config.run_name.removeprefix("harmonized_")
    out_dir.mkdir(parents=True, exist_ok=True)
    caption = (f"{len(species)} test images, {len(set(species))} species, {len(set(genus))} genera. "
               "t-SNE of the 640-d embeddings: it keeps neighborhoods, not distances "
               "(cluster sizes and gaps carry no meaning).")

    # Figure 1: colored by genus (the 7 most represented genera, the others in grey).
    genera = [g for g, _ in Counter(genus).most_common(len(PALETTE) - 1)]
    draw_map(xy, genus, genera, f"other genera ({len(set(genus)) - len(genera)})", "Genus (test images)",
             f"{args.label}: test images by genus",
             f"Nearest neighbour in the embedding space is of the same genus for {same_genus:.0f}% of the "
             f"test images.\n{caption}", out_dir / "tsne_genus.png", name=lambda g: g)

    # Figure 2: the 8 most represented species among the shared test pool (>= 6 images).
    eligible = [s for s, n in species_count.items() if n >= 6]
    top = sorted(eligible, key=lambda s: (-species_count[s], s))[:len(PALETTE)]
    draw_map(xy, species, top, f"other species ({len(species_count) - len(top)})", "Species (test images)",
             f"{args.label}: the {len(top)} most represented test species",
             f"Nearest neighbour in the embedding space is of the same species for {same_species:.0f}% of "
             f"the test images ({n_species_counted} species with at least 2 images).\n{caption}",
             out_dir / "tsne_species.png", name=short)

    print(f"Nearest neighbour: same species {same_species:.1f} %, same genus {same_genus:.1f} %")
    print(f"Written to {out_dir}")


def draw_map(xy, groups, highlight, other_label, legend_title, title, subtitle, out, name):
    """Scatter of the t-SNE map: `highlight` groups in the palette order, the rest in grey."""
    fig, ax = plt.subplots(figsize=(9, 7.2))
    other = ~np.isin(groups, highlight)
    ax.scatter(xy[other, 0], xy[other, 1], s=7, color=OTHER, linewidth=0,
               label=f"{other_label}: {int(other.sum())}")
    for k, group in enumerate(highlight):
        mask = groups == group
        ax.scatter(xy[mask, 0], xy[mask, 1], s=24, color=PALETTE[k], marker=MARKERS[k],
                   edgecolor=SURFACE, linewidth=0.6, label=f"{name(group)}: {int(mask.sum())}")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, pad=34)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), title=legend_title,
              title_fontsize=8, labelcolor=INK_2, markerscale=1.1)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
