"""Redimensionne un dataset d'images vers une resolution fixe.

L'arborescence du dataset source (split/classe/image.jpg) est reproduite
telle quelle dans le dossier de sortie.

Exemple :
    python scripts/resize_dataset.py --src ../ds_FSL/ds_FSL --dst ../ds_FSL_224x204
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageFile, ImageOps

# Une grande partie du dataset est en JPEG tronque (marqueur de fin manquant) :
# sans ceci Pillow leve "image file is truncated" au lieu de decoder l'image.
ImageFile.LOAD_TRUNCATED_IMAGES = True

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resize(im, size, mode, fill):
    """Ramene `im` a `size` (largeur, hauteur) selon la strategie `mode`."""
    if mode == "stretch":
        return im.resize(size, Image.LANCZOS)
    if mode == "crop":
        return ImageOps.fit(im, size, method=Image.LANCZOS, centering=(0.5, 0.5))
    if mode == "pad":
        return ImageOps.pad(im, size, method=Image.LANCZOS, color=fill, centering=(0.5, 0.5))
    raise ValueError(f"mode inconnu : {mode}")


def process(job):
    src, dst, size, mode, fill, quality = job
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode != "RGB":
                im = im.convert("RGB")
            out = resize(im, size, mode, fill)
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst, quality=quality, subsampling=0)
        return None
    except Exception as exc:  # image corrompue, droits, etc.
        return f"{src} : {exc}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, type=Path, help="racine du dataset source")
    p.add_argument("--dst", required=True, type=Path, help="racine du dataset de sortie")
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--height", type=int, default=204)
    p.add_argument("--mode", choices=("pad", "crop", "stretch"), default="pad",
                   help="pad = garde le ratio + bandes, crop = garde le ratio + rogne, stretch = deforme")
    p.add_argument("--fill", default="0,0,0", help="couleur des bandes en mode pad, format R,G,B")
    p.add_argument("--quality", type=int, default=95, help="qualite JPEG de sortie")
    p.add_argument("--workers", type=int, default=None, help="processus paralleles (defaut : nb de coeurs)")
    p.add_argument("--dry-run", action="store_true", help="compte les fichiers sans rien ecrire")
    args = p.parse_args()

    src_root = args.src.resolve()
    dst_root = args.dst.resolve()
    if not src_root.is_dir():
        sys.exit(f"source introuvable : {src_root}")
    if dst_root == src_root or dst_root in src_root.parents:
        sys.exit("le dossier de sortie ne doit pas contenir la source")

    size = (args.width, args.height)
    fill = tuple(int(c) for c in args.fill.split(","))

    files = sorted(f for f in src_root.rglob("*") if f.suffix.lower() in EXTENSIONS)
    print(f"{len(files)} images trouvees dans {src_root}")
    if args.dry_run:
        return

    jobs = [
        (f, dst_root / f.relative_to(src_root).with_suffix(".jpg"), size, args.mode, fill, args.quality)
        for f in files
    ]

    errors = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for err in pool.map(process, jobs, chunksize=16):
            done += 1
            if err:
                errors.append(err)
            if done % 250 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}", flush=True)

    print(f"\n{done - len(errors)} images ecrites en {size[0]}x{size[1]} ({args.mode}) dans {dst_root}")
    if errors:
        print(f"{len(errors)} echecs :")
        for e in errors[:20]:
            print("  -", e)


if __name__ == "__main__":
    main()
