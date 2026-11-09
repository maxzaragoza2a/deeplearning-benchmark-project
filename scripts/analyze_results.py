"""Analysis figures for the PRWN runs: training curves, test scores, and how
each model classifies the test images.

Reloads the saved models, recomputes the embeddings of the test classes and
replays exactly the same episodes as the evaluation (same seed), keeping every
prediction. Nothing is retrained.

Compared models:
- PRWN Conv-4      (results/prwn_conv4/best_model.pt)
- ProtoNet Conv-4  (results/prwn_conv4_pn/best_model.pt)
- PRWN DINOv2      (results/prwn/best_head.pt, on the cached DINOv2 embeddings)
- DINOv2 only      (raw DINOv2 embeddings, no head and no training)

Figures written to figures/ (versioned, at the repository root):
    insights/<model>/           one model on its own
        training.png            accuracy, error and loss over training
        classification.png      per-class accuracy, confusions, confidence, calibration
        tsne.png                2D map of the test embeddings
    comparison/<question>/      several models side by side
        all_models/                         the 4 models
        random_walk_prwn_vs_protonet/       PRWN Conv-4 vs ProtoNet Conv-4
        backbone_conv4_vs_dinov2/           PRWN Conv-4 vs PRWN DINOv2
        (test, training, per_class, tsne, separability).png
    analysis.json               silhouette, calibration and per-class accuracy

Example:
    python scripts/analyze_results.py --device cuda
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import build_index, load_images  # noqa: E402
from src.data.episodes import EpisodeSampler, split_classes  # noqa: E402
from src.models.backbone import ProjectionHead, cached_embeddings  # noqa: E402
from src.models.conv4 import Conv4, ImagePreprocessor, embed_images  # noqa: E402
from src.models.prwn import compute_prototypes, prototype_logits  # noqa: E402
from src.utils.config import load_config  # noqa: E402

# --- Style: reference categorical palette, in a fixed order -----------------------
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Each model keeps its color across all figures.
# `folder` = folder name under figures/insights/.
MODELS = {
    "prwn_conv4": {"label": "PRWN Conv-4", "color": PALETTE[0], "marker": "o", "folder": "prwn_conv4"},
    "prwn_conv4_pn": {"label": "ProtoNet Conv-4", "color": PALETTE[1], "marker": "s", "folder": "protonet_conv4"},
    "prwn": {"label": "PRWN DINOv2", "color": PALETTE[2], "marker": "^", "folder": "prwn_dinov2"},
    "dinov2": {"label": "DINOv2 only", "color": PALETTE[3], "marker": "D", "folder": "dinov2_only"},
}
TRAINED = ["prwn_conv4", "prwn_conv4_pn", "prwn"]

# Comparisons: folder under figures/comparison/ -> compared models.
COMPARISONS = {
    "all_models": list(MODELS),
    "random_walk_prwn_vs_protonet": ["prwn_conv4", "prwn_conv4_pn"],
    "backbone_conv4_vs_dinov2": ["prwn_conv4", "prwn"],
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": INK_3, "ytick.color": INK_3, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "legend.fontsize": 8, "text.color": INK,
    "lines.linewidth": 2, "font.size": 9, "figure.titlesize": 13, "figure.titleweight": "bold",
})


def pct(ax, axis="y"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


# --- Data --------------------------------------------------------------------------
def read_history(run):
    with open(ROOT / "results" / run / "history.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def read_metrics(run):
    with open(ROOT / "results" / run / "metrics.json", encoding="utf-8") as handle:
        return json.load(handle)


def test_embeddings(device):
    """Embeddings of the test images for the 4 models, and their labels."""
    config = load_config(ROOT / "configs" / "prwn.yaml", {"device": device})
    device = config.resolve_device()
    index = build_index(config.data_root, config.splits, config.dedup, config.limit_per_class)
    counts = {name: int((index.labels == i).sum()) for i, name in enumerate(index.class_names)}
    s = config.class_split
    _, _, test_names = split_classes(counts, s.n_val_classes, s.n_test_classes, s.n_bins, s.seed)
    label_of = {name: i for i, name in enumerate(index.class_names)}
    positions = np.flatnonzero(np.isin(index.labels, [label_of[n] for n in test_names]))

    out = {}
    # DINOv2: cached embeddings, then the trained PRWN head.
    features = cached_embeddings(
        config.model, index.paths, config.input_size, lambda: load_images(index, config.input_size),
        device, config.cache_dir, config.extract_batch_size,
    )[positions]
    head = ProjectionHead(features.shape[1], **vars(config.head))
    head.load_state_dict(torch.load(config.run_dir / "best_head.pt", map_location="cpu"))
    with torch.no_grad():
        out["prwn"] = head.eval()(features)
    out["dinov2"] = features

    # Conv-4: test images in memory, then the two saved models.
    conv_config = load_config(ROOT / "configs" / "prwn_conv4.yaml", {"device": device})
    images = load_images(index, conv_config.input_size)[positions]
    for run in ("prwn_conv4", "prwn_conv4_pn"):
        cfg = load_config(ROOT / "configs" / f"{run}.yaml", {"device": device}).conv4
        model = Conv4(hidden_channels=cfg.hidden_channels, pool_size=cfg.pool_size).to(device)
        preprocess = ImagePreprocessor(torch.zeros(3), torch.ones(3), cfg.hflip, cfg.max_shift).to(device)
        state = torch.load(ROOT / "results" / run / "best_model.pt", map_location=device)
        model.load_state_dict(state["model"])
        preprocess.load_state_dict(state["preprocess"])  # training mean / std
        out[run] = embed_images(model, preprocess, images, device, conv_config.extract_batch_size)
    return out, index.labels[positions], index.class_names, config


def replay_episodes(embeddings, labels, config, k_shot):
    """Replay the test episodes (same draws as train_prwn.evaluate) and keep every query."""
    e = config.eval
    pools = {int(c): np.flatnonzero(labels == c) for c in np.unique(labels)}
    sampler = EpisodeSampler(pools, pools, e.n_way, k_shot, e.n_query, e.n_unlabeled, shared_pool=True)
    rng = np.random.default_rng(config.seed)
    true, pred, conf, episode_classes, accs = [], [], [], [], []
    for _ in range(e.n_episodes):
        ep = sampler.sample(rng)
        classes = labels[ep.support[:, 0]]
        probs = F.softmax(prototype_logits(
            embeddings[ep.query], compute_prototypes(embeddings[ep.support]), config.train.distance_scale
        ), dim=1)
        best = probs.argmax(1).numpy()
        true.append(classes[ep.query_class])
        pred.append(classes[best])
        conf.append(probs.max(1).values.numpy())
        episode_classes.append(np.repeat(classes[None], len(ep.query), axis=0))
        accs.append(float((best == ep.query_class).mean()))
    return {
        "true": np.concatenate(true), "pred": np.concatenate(pred), "conf": np.concatenate(conf),
        "episode_classes": np.concatenate(episode_classes), "accuracy": float(np.mean(accs)),
        "eligible": sampler.eligible,
    }


def confusion(r, classes):
    """P(predicted j | true class i, j present in the episode): corrects for the random class draw."""
    pos = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    hits, together = np.zeros((n, n)), np.zeros((n, n))
    for t, p, ep in zip(r["true"], r["pred"], r["episode_classes"]):
        hits[pos[t], pos[p]] += 1
        for c in ep:
            together[pos[t], pos[c]] += 1
    return hits / np.maximum(together, 1)


def per_class_accuracy(r, classes):
    out = []
    for c in classes:
        mask = r["true"] == c
        out.append((r["pred"][mask] == c).mean() if mask.any() else np.nan)  # class missing in 5-shot
    return np.array(out)


def calibration(r, n_bins=10, min_count=30):
    """Accuracy per confidence bin, and expected calibration error (ECE)."""
    edges = np.linspace(0.2, 1.0, n_bins + 1)  # confidence >= 1/5 in 5-way
    correct = r["true"] == r["pred"]
    centers, accs, sizes = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (r["conf"] >= lo) & (r["conf"] < hi if hi < 1 else r["conf"] <= hi)
        if mask.sum() >= min_count:  # bin too small = very noisy point
            centers.append(r["conf"][mask].mean())
            accs.append(correct[mask].mean())
            sizes.append(mask.sum())
    sizes = np.array(sizes)
    ece = float(np.sum(sizes * np.abs(np.array(accs) - np.array(centers))) / sizes.sum())
    return np.array(centers), np.array(accs), sizes, ece


def short(name):
    genus, *rest = name.split(" ")
    return f"{genus[0]}. {' '.join(rest)}" if rest else name


# --- Single-model figures ----------------------------------------------------------
def plot_training(run, hist, metrics, out_dir):
    label = MODELS[run]["label"]
    steps = hist["step"]
    best = metrics["best_val"]
    has_rw = "train_walker" in hist and hist["train_walker"].max() > 0 and run != "prwn_conv4_pn"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle(f"{label}: training", x=0.01, ha="left")

    ax = axes[0]
    ax.plot(steps, hist["train_accuracy"], color=PALETTE[0], label="train (1-shot episodes)")
    ax.plot(steps, hist["val_accuracy"], color=PALETTE[1], label="validation")
    ax.fill_between(steps, hist["val_accuracy"] - hist["val_ci95"], hist["val_accuracy"] + hist["val_ci95"],
                    color=PALETTE[1], alpha=0.15, linewidth=0)
    ax.scatter([best["step"]], [best["accuracy"]], s=50, color=PALETTE[1], edgecolor=SURFACE, linewidth=2, zorder=5)
    ax.annotate(f"best val {best['accuracy']:.1%}\nstep {best['step']:,}",
                (best["step"], best["accuracy"]), xytext=(0, -34), textcoords="offset points",
                ha="center", fontsize=8, color=INK_2)
    ax.set_title("Accuracy")
    ax.set_xlabel("steps (episodes)")
    pct(ax)
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(steps, 1 - hist["val_accuracy"], color=PALETTE[1], label="validation error")
    ax.plot(steps, 1 - hist["train_accuracy"], color=PALETTE[0], label="train error")
    ax.set_title("Error (1 - accuracy)")
    ax.set_xlabel("steps (episodes)")
    pct(ax)
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.plot(steps, hist["train_supervised"], color=PALETTE[0], label="prototypical loss (supervised)")
    if has_rw:
        ax.plot(steps, hist["train_walker"], color=PALETTE[2], label="random walk loss (walker)")
        ax.plot(steps, hist["train_visit"], color=PALETTE[3], label="visit loss")
        ax.plot(steps, hist["train_loss"], color=INK_2, linewidth=1.2, label="total loss")
    ax.set_title("Training loss")
    ax.set_xlabel("steps (episodes)")
    ax.set_yscale("log")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    save(fig, out_dir / "training.png")


def plot_classification(run, results, classes, names, tsne_xy, highlight, labels, out_dir):
    label = MODELS[run]["label"]
    r1, r5 = results[1], results[5]
    fig = plt.figure(figsize=(18, 12), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.15, h_pad=0.15, wspace=0.08, hspace=0.06)
    fig.suptitle(f"{label}: how the model classifies the test images", x=0.01, ha="left")
    grid = fig.add_gridspec(2, 3, width_ratios=[0.9, 1.2, 1])

    # Per-class accuracy, sorted.
    ax = fig.add_subplot(grid[:, 0])
    a1, a5 = per_class_accuracy(r1, classes), per_class_accuracy(r5, classes)
    order = np.argsort(a1)
    y = np.arange(len(classes))
    ax.barh(y + 0.2, a1[order], height=0.38, color=PALETTE[0], label="1-shot")
    ax.barh(y - 0.2, a5[order], height=0.38, color=PALETTE[1], label="5-shot")
    ax.set_yticks(y, [short(names[classes[i]]) for i in order], fontsize=8)
    ax.set_xlim(0, 1)
    pct(ax, "x")
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, a5[order]):
        if np.isnan(v):
            ax.text(0.01, yi - 0.2, "missing in 5-shot (too few images)", va="center", fontsize=7, color=INK_3)
    ax.set_title("Accuracy per test class")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.04), ncol=2)

    # 1-shot confusions, conditioned on the class being present in the episode.
    ax = fig.add_subplot(grid[0, 1])
    m = confusion(r1, classes)
    ordered = [classes[i] for i in order[::-1]]
    idx = [classes.index(c) for c in ordered]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    im = ax.imshow(m[np.ix_(idx, idx)], cmap=cmap, vmin=0, vmax=1)
    ticks = [short(names[c]) for c in ordered]
    ax.set_xticks(range(len(ticks)), ticks, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(len(ticks)), ticks, fontsize=7)
    ax.grid(False)
    ax.set_title("1-shot confusions: P(predicted | true class)")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, format=matplotlib.ticker.PercentFormatter(1.0, 0))

    # Most confused pairs.
    ax = fig.add_subplot(grid[1, 1])
    off = m.copy()
    np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:8], off.shape))[0]
    vals = [off[i, j] for i, j in top]
    tick = [f"{short(names[classes[i]])} -> {short(names[classes[j]])}" for i, j in top]
    ax.barh(range(len(vals))[::-1], vals, height=0.6, color=PALETTE[0])
    ax.set_yticks(range(len(vals))[::-1], tick, fontsize=8)
    for k, v in enumerate(vals):
        ax.text(v + 0.005, len(vals) - 1 - k, f"{v:.0%}", va="center", fontsize=7, color=INK_2)
    pct(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_title("Most confused pairs (1-shot)")
    ax.set_xlabel("share of the true class queries sent to the other class")

    # Confidence: correct vs errors, and calibration.
    ax = fig.add_subplot(grid[0, 2])
    correct = r1["true"] == r1["pred"]
    bins = np.linspace(0.2, 1, 33)
    ax.hist(r1["conf"][correct], bins=bins, color=PALETTE[0], alpha=0.85, label="correct predictions",
            density=True, edgecolor=SURFACE, linewidth=0.5)
    ax.hist(r1["conf"][~correct], bins=bins, color=PALETTE[1], alpha=0.75, label="errors",
            density=True, edgecolor=SURFACE, linewidth=0.5)
    pct(ax, "x")
    ax.set_title("Prediction confidence (1-shot)")
    ax.set_xlabel("probability of the predicted class")
    ax.set_ylabel("density")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(grid[1, 2])
    for k, color in ((1, PALETTE[0]), (5, PALETTE[1])):
        c, a, _, ece = calibration(results[k])
        ax.plot(c, a, marker="o", markersize=5, color=color, label=f"{k}-shot (ECE {ece:.1%})")
    ax.plot([0.2, 1], [0.2, 1], color=INK_3, linewidth=1, label="perfect calibration")
    pct(ax)
    pct(ax, "x")
    ax.set_title("Calibration")
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("actual accuracy")
    ax.legend(loc="upper left")
    save(fig, out_dir / "classification.png")

    # t-SNE on its own, larger.
    fig, ax = plt.subplots(figsize=(8, 7))
    draw_tsne(ax, tsne_xy, labels, highlight, names)
    ax.set_title(f"{label}: test image embeddings (t-SNE)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=1.5)
    save(fig, out_dir / "tsne.png")


def draw_tsne(ax, xy, labels, highlight, names):
    other = ~np.isin(labels, highlight)
    ax.scatter(xy[other, 0], xy[other, 1], s=6, color="#c9c8c3", linewidth=0, label="other test classes")
    for k, c in enumerate(highlight):
        mask = labels == c
        ax.scatter(xy[mask, 0], xy[mask, 1], s=22, color=PALETTE[k], marker=MARKERS[k],
                   edgecolor=SURFACE, linewidth=0.6, label=short(names[c]))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


# --- Comparison figures ------------------------------------------------------------
def title_of(runs):
    return " vs ".join(MODELS[r]["label"] for r in runs) if len(runs) < len(MODELS) else "all models"


def plot_test_bars(metrics, runs, out_dir):
    tests = {run: metrics[run]["test"] for run in TRAINED}
    tests["dinov2"] = metrics["prwn"]["test_backbone_only"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    fig.suptitle(f"Test accuracy, {title_of(runs)} (5-way, 600 episodes, bar = 95% CI)", x=0.01, ha="left")
    width = 0.8 / max(len(runs), 3)
    for ax, (key, ci, title) in zip(axes, (
        ("accuracy", "ci95", "Nearest prototype"),
        ("accuracy_refined", "ci95_refined", "Refined prototypes (soft k-means)"),
    )):
        for i, run in enumerate(runs):
            vals = [tests[run][f"{k}shot"][key] for k in (1, 5)]
            errs = [tests[run][f"{k}shot"][ci] for k in (1, 5)]
            x = np.arange(2) + (i - (len(runs) - 1) / 2) * (width + 0.02)
            ax.bar(x, vals, width, yerr=errs, color=MODELS[run]["color"], label=MODELS[run]["label"],
                   error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
            for xi, v, e in zip(x, vals, errs):
                ax.text(xi, v + e + 0.004, f"{v:.1%}", ha="center", fontsize=7.5, color=INK_2)
        ax.set_xticks(range(2), ["1-shot", "5-shot"])
        ax.set_ylim(0.7, 0.97)
        ax.grid(axis="x", visible=False)
        pct(ax)
        ax.set_title(title)
    axes[0].legend(loc="upper left", ncol=2)
    axes[0].text(0, -0.13, "Vertical axis truncated at 70%.", transform=axes[0].transAxes, fontsize=7.5, color=INK_3)
    fig.tight_layout()
    save(fig, out_dir / "test.png")


def plot_curves(histories, metrics, runs, out_dir):
    runs = [r for r in runs if r in TRAINED]  # DINOv2 only has no training
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    fig.suptitle(f"Training compared, {title_of(runs)} (5-way 1-shot episodes)", x=0.01, ha="left")
    panels = [
        (axes[0, 0], lambda h: h["val_accuracy"], "Validation accuracy"),
        (axes[0, 1], lambda h: h["train_accuracy"], "Accuracy on training episodes"),
        (axes[1, 0], lambda h: 1 - h["val_accuracy"], "Validation error"),
        (axes[1, 1], lambda h: h["train_accuracy"] - h["val_accuracy"], "Train - validation gap (overfitting)"),
    ]
    for ax, fn, title in panels:
        for run in runs:
            h = histories[run]
            ax.plot(h["step"], fn(h), color=MODELS[run]["color"], label=MODELS[run]["label"])
        ax.set_title(title)
        ax.set_xlabel("steps (episodes)")
        pct(ax)
    for run in runs:
        b = metrics[run]["best_val"]
        axes[0, 0].scatter([b["step"]], [b["accuracy"]], s=55, color=MODELS[run]["color"],
                           edgecolor=SURFACE, linewidth=2, zorder=5)
        axes[0, 0].annotate(f"{b['accuracy']:.1%}", (b["step"], b["accuracy"]), xytext=(0, 7),
                            textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].text(0.01, 0.02, "dot = best validation model (the one evaluated on test)",
                    transform=axes[0, 0].transAxes, fontsize=7.5, color=INK_3)
    fig.tight_layout()
    save(fig, out_dir / "training.png")


def plot_per_class(results, classes, names, runs, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    fig.suptitle(f"Accuracy per test class, {title_of(runs)}", x=0.01, ha="left")
    accs = {k: {run: per_class_accuracy(results[run][k], classes) for run in runs} for k in (1, 5)}
    order = np.argsort(np.mean([accs[1][run] for run in runs], axis=0))
    y = np.arange(len(classes))
    for ax, k in zip(axes, (1, 5)):
        lo = np.min([accs[k][run][order] for run in runs], axis=0)
        hi = np.max([accs[k][run][order] for run in runs], axis=0)
        ax.hlines(y, lo, hi, color=GRID, linewidth=3, zorder=1)
        for run in runs:
            ax.scatter(accs[k][run][order], y, s=42, color=MODELS[run]["color"], marker=MODELS[run]["marker"],
                       edgecolor=SURFACE, linewidth=1, zorder=3, label=MODELS[run]["label"])
        ax.set_title(f"{k}-shot")
        ax.set_xlim(0.3, 1.01)
        pct(ax, "x")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [short(names[classes[i]]) for i in order], fontsize=8)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", ncol=4, bbox_to_anchor=(1, 1.0))
    fig.text(0.01, -0.01, "Grey bar = gap between the best and the worst model. "
             "G. calliginea does not have enough images for 5-shot episodes.", fontsize=7.5, color=INK_3)
    fig.tight_layout()
    save(fig, out_dir / "per_class.png")


def plot_tsne_grid(tsne, labels, highlight, names, sil, runs, out_dir):
    n_rows = 1 if len(runs) <= 2 else 2
    n_cols = int(np.ceil(len(runs) / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows + 0.8), squeeze=False)
    fig.suptitle(f"Test image embeddings (t-SNE), {title_of(runs)}", x=0.01, ha="left")
    for ax, run in zip(axes.ravel(), runs):
        draw_tsne(ax, tsne[run], labels, highlight, names)
        ax.set_title(f"{MODELS[run]['label']}  ·  silhouette {sil[run]:.2f}")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=5, markerscale=1.5)
    fig.tight_layout(rect=(0, 0.1 if n_rows == 1 else 0.05, 1, 0.97))
    save(fig, out_dir / "tsne.png")


def plot_separability(results, sep, runs, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    fig.suptitle(f"Class separation and calibration, {title_of(runs)}", x=0.01, ha="left")
    x = np.arange(len(runs))
    colors = [MODELS[r]["color"] for r in runs]
    ticks = [MODELS[r]["label"].replace(" ", "\n", 1) for r in runs]

    for ax, key, title, fmt in (
        (axes[0], "silhouette", "Silhouette of the test classes\n(higher = better separated classes)", "{:.2f}"),
        (axes[1], "ratio", "Inter / intra-class distance\n(higher = better separated classes)", "{:.2f}"),
    ):
        vals = [sep[r][key] for r in runs]
        ax.bar(x, vals, 0.6, color=colors)
        for xi, v in zip(x, vals):
            ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.set_xticks(x, ticks, fontsize=8)
        pad = 0.6 + 0.3 * max(0, 3 - len(runs))  # keeps bars thin and centered with few models
        ax.set_xlim(-pad, len(runs) - 1 + pad)
        ax.grid(axis="x", visible=False)
        ax.set_title(title)

    ax = axes[2]
    for run in runs:
        c, a, _, ece = calibration(results[run][1])
        ax.plot(c, a, marker=MODELS[run]["marker"], markersize=5, color=MODELS[run]["color"],
                label=f"{MODELS[run]['label']} (ECE {ece:.1%})")
    ax.plot([0.2, 1], [0.2, 1], color=INK_3, linewidth=1, label="perfect calibration")
    pct(ax)
    pct(ax, "x")
    ax.set_title("1-shot calibration\n(stated confidence vs actual accuracy)")
    if any(r in ("prwn", "dinov2") for r in runs):
        ax.text(0.98, 0.03, "DINOv2: near-binary softmax (large embedding\nnorms, distance_scale = 1), "
                "so confidence is\nnear 100% even when the model is wrong.",
                transform=ax.transAxes, ha="right", fontsize=7, color=INK_3)
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("actual accuracy")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, out_dir / "separability.png")


def separability(emb, labels):
    """Silhouette (Euclidean distance) and ratio of mean inter / intra-class distance."""
    x = emb.numpy().astype(np.float64)
    classes = np.unique(labels)
    centers = np.stack([x[labels == c].mean(0) for c in classes])
    intra = np.mean([np.linalg.norm(x[labels == c] - centers[i], axis=1).mean() for i, c in enumerate(classes)])
    d = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    inter = d[np.triu_indices(len(classes), 1)].mean()
    return {"silhouette": float(silhouette_score(x, labels)), "ratio": float(inter / intra)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {run: read_metrics(run) for run in TRAINED}
    histories = {run: read_history(run) for run in TRAINED}

    print("Test image embeddings...")
    embeddings, labels, names, config = test_embeddings(args.device)

    print("Replaying test episodes...")
    results = {run: {k: replay_episodes(embeddings[run], labels, config, k) for k in (1, 5)} for run in MODELS}
    for run in MODELS:
        ref = metrics["prwn"]["test_backbone_only"] if run == "dinov2" else metrics[run]["test"]
        for k in (1, 5):
            got, want = results[run][k]["accuracy"], ref[f"{k}shot"]["accuracy"]
            flag = "ok" if abs(got - want) < 1e-3 else "MISMATCH"
            print(f"  {MODELS[run]['label']:<16} {k}-shot: replayed {got:.4f} | metrics.json {want:.4f} [{flag}]")

    # Classes seen in 1-shot (eligible): common base for all per-class figures.
    classes = [int(c) for c in results["prwn_conv4"][1]["eligible"]]
    counts = {c: int((labels == c).sum()) for c in classes}
    highlight = sorted(classes, key=lambda c: -counts[c])[:8]

    print("t-SNE and separability...")
    tsne, sep = {}, {}
    for run in MODELS:
        x = embeddings[run].numpy()
        tsne[run] = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(x)
        sep[run] = separability(embeddings[run], labels)
        print(f"  {MODELS[run]['label']:<16} silhouette {sep[run]['silhouette']:.3f} | inter/intra {sep[run]['ratio']:.2f}")

    print("Per-model figures (insights/):")
    for run in MODELS:
        model_dir = args.out_dir / "insights" / MODELS[run]["folder"]
        model_dir.mkdir(parents=True, exist_ok=True)
        if run in TRAINED:
            plot_training(run, histories[run], metrics[run], model_dir)
        plot_classification(run, results[run], classes, names, tsne[run], highlight, labels, model_dir)

    print("Comparison figures (comparison/):")
    silhouettes = {r: sep[r]["silhouette"] for r in MODELS}
    for name, runs in COMPARISONS.items():
        cmp_dir = args.out_dir / "comparison" / name
        cmp_dir.mkdir(parents=True, exist_ok=True)
        plot_test_bars(metrics, runs, cmp_dir)
        plot_curves(histories, metrics, runs, cmp_dir)
        plot_per_class(results, classes, names, runs, cmp_dir)
        plot_tsne_grid(tsne, labels, highlight, names, silhouettes, runs, cmp_dir)
        plot_separability(results, sep, runs, cmp_dir)

    summary = {
        run: {
            "separability": sep[run],
            "ece": {f"{k}shot": calibration(results[run][k])[3] for k in (1, 5)},
            "per_class_accuracy_1shot": dict(zip(
                [names[c] for c in classes], per_class_accuracy(results[run][1], classes).round(4).tolist()
            )),
        }
        for run in MODELS
    }
    with open(args.out_dir / "analysis.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Summary written to {(args.out_dir / 'analysis.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
