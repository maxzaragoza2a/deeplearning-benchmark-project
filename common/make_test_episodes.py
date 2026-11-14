"""Build the test episode file shared by the three methods (NCA, ProtoNet, PRWN).

The output, `common/test_episodes.json`, is committed: everyone evaluates on exactly
the same tasks. There is no need to run this script again; it stays in the repo to
show how the file was produced.

Protocol (group decisions):
- `test` folder of the `ds_FSL_224x204` dataset, images read as is;
- eligible classes: at least `max_shot + n_query` images (6 by default);
- 1000 5-way episodes; per class, 6 images drawn without replacement;
- 5-shot: images 1 to 5 are the support, image 6 is the query;
- 1-shot: image 1 is the support, same query (nested episodes).

The md5 of every image is stored: the evaluation refuses to run on a copy of the
dataset that differs by even one image.

Examples:
    python common/make_test_episodes.py
    python common/make_test_episodes.py --data-root ../ds_FSL_224x204 --seed 0
"""

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGE_SIZE = (224, 204)  # (width, height) of the Drive images


def nfc(name: str) -> str:
    """Same Unicode form for accented names (e.g. Culicoides) on every operating system."""
    return unicodedata.normalize("NFC", name)


def list_classes(split_dir: Path) -> dict[str, list[Path]]:
    """{class: [images]}, classes and images sorted: same result on Windows and Linux."""
    classes = {}
    for class_dir in sorted((p for p in split_dir.iterdir() if p.is_dir()), key=lambda p: nfc(p.name)):
        images = sorted(
            (f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in EXTENSIONS),
            key=lambda f: nfc(f.name),
        )
        classes[nfc(class_dir.name)] = images
    return classes


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def check_size(path: Path) -> None:
    with Image.open(path) as image:  # reads the header only
        if image.size != IMAGE_SIZE:
            sys.exit(f"{path} is {image.size[0]}x{image.size[1]}, expected {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}: "
                     "this is not the ds_FSL_224x204 dataset.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=ROOT.parent / "ds_FSL_224x204", help="dataset root")
    p.add_argument("--split", default="test", help="folder the episodes are drawn from")
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--max-shot", type=int, default=5, help="support images per class in 5-shot (1-shot uses 1)")
    p.add_argument("--n-query", type=int, default=1, help="query images per class")
    p.add_argument("--n-episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "common" / "test_episodes.json")
    args = p.parse_args()

    split_dir = args.data_root / args.split
    if not split_dir.is_dir():
        sys.exit(f"Folder not found: {split_dir}")
    per_class = args.max_shot + args.n_query

    # --- Eligible classes and list of usable images ---
    classes = list_classes(split_dir)
    eligible = [name for name, images in classes.items() if len(images) >= per_class]
    if len(eligible) < args.n_way:
        sys.exit(f"Only {len(eligible)} classes have at least {per_class} images, {args.n_way} are needed.")

    images, positions = [], {}  # positions[class] = indices of its images in `images`
    for name in eligible:
        positions[name] = []
        for path in classes[name]:
            check_size(path)
            positions[name].append(len(images))
            images.append({"path": f"{name}/{nfc(path.name)}", "md5": md5(path)})

    # --- Draw: 5 classes, then 6 images per class, without replacement ---
    rng = np.random.default_rng(args.seed)
    episodes = []
    for _ in range(args.n_episodes):
        chosen = [eligible[i] for i in rng.choice(len(eligible), size=args.n_way, replace=False)]
        picks = [
            [int(i) for i in rng.choice(positions[name], size=per_class, replace=False)]
            for name in chosen
        ]
        episodes.append({"classes": chosen, "images": picks})

    # --- Sanity checks before writing ---
    for episode in episodes:
        assert len(set(episode["classes"])) == args.n_way
        flat = [i for picks in episode["images"] for i in picks]
        assert len(flat) == len(set(flat)) == args.n_way * per_class
        for name, picks in zip(episode["classes"], episode["images"]):
            assert all(images[i]["path"].startswith(name + "/") for i in picks)

    # --- Write: one element per line, readable and diff-friendly ---
    protocol = {
        "dataset": "ds_FSL_224x204",
        "split": args.split,
        "image_size": list(IMAGE_SIZE),
        "n_way": args.n_way,
        "shots": [1, args.max_shot],
        "n_query": args.n_query,
        "min_images_per_class": per_class,
        "n_episodes": args.n_episodes,
        "seed": args.seed,
        "layout": (
            f"episodes[e].images[c] = {per_class} indices into `images` for class c; "
            f"k-shot: the first k are the support, the last {args.n_query} the query"
        ),
    }
    dump = lambda value: json.dumps(value, ensure_ascii=False)  # noqa: E731
    lines = ["{", ' "protocol": {']
    lines += [f"  {dump(key)}: {dump(value)}," for key, value in protocol.items()]
    lines[-1] = lines[-1].rstrip(",")
    lines += [" },", ' "images": [']
    lines += [f"  {dump(image)}," for image in images]
    lines[-1] = lines[-1].rstrip(",")
    lines += [" ],", ' "episodes": [']
    lines += [f"  {dump(episode)}," for episode in episodes]
    lines[-1] = lines[-1].rstrip(",")
    lines += [" ]", "}"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.loads(args.out.read_text(encoding="utf-8"))  # the file reads back

    # --- Summary ---
    used = {i for episode in episodes for picks in episode["images"] for i in picks}
    appearances = Counter(name for episode in episodes for name in episode["classes"])
    print(f"{len(eligible)} eligible classes out of {len(classes)} (at least {per_class} images)")
    print(f"{len(images)} images indexed, {len(used)} used at least once")
    print(f"{args.n_episodes} {args.n_way}-way episodes, shots {protocol['shots']}, {args.n_query} query per class")
    print(f"Appearances per class: min {min(appearances.values())}, max {max(appearances.values())}")
    print(f"Written to {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
