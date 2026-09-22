# deeplearning-benchmark-project

## Prétraitement du dataset

Le dataset d'images est conservé hors du dépôt. `scripts/resize_dataset.py` le ramène
à une résolution fixe en reproduisant l'arborescence `split/classe/image.jpg` dans un
nouveau dossier, sans toucher à la source.

```bash
python scripts/resize_dataset.py \
    --src /chemin/vers/ds_FSL \
    --dst /chemin/vers/ds_FSL_224x204 \
    --width 224 --height 204 --mode pad
```

| Option | Rôle |
| --- | --- |
| `--mode pad` | garde le ratio et complète avec des bandes (`--fill R,G,B`, noir par défaut) |
| `--mode crop` | garde le ratio et rogne au centre |
| `--mode stretch` | redimensionne sans garder le ratio |
| `--workers` | processus parallèles (défaut : nombre de cœurs) |
| `--dry-run` | compte les images sans rien écrire |

Le script corrige l'orientation EXIF, convertit tout en RGB et décode les JPEG tronqués
(`ImageFile.LOAD_TRUNCATED_IMAGES`), ce dont la majorité des images de `ds_FSL` a besoin.

Dépendance : `Pillow`.
