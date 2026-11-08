# Runs PRWN Conv-4 sur le serveur GPU

Instructions pour la session Claude lancée sur le serveur SSH. Objectif : entraîner les
deux runs Conv-4 de la branche `feat/prwn-conv4`, puis rapporter leurs scores face au run
DINOv2 de référence.

## Contexte

- `prwn_conv4` : PRWN (Prototypical Random Walk Network), avec un Conv-4 entraîné de zéro.
- `prwn_conv4_pn` : contrôle, même Conv-4 mais sans marche aléatoire (`lambda_rw: 0`),
  donc un Prototypical Network classique.
- Les deux runs héritent de `configs/prwn.yaml` : même dataset, même split par classes
  (70 train / 20 val / 39 test), mêmes épisodes de test. Les scores se comparent
  directement à ceux du run DINOv2.

## Règles

- Ne jamais modifier le dataset en place.
- Ne pas changer les configs pour « améliorer » un score. Si un run échoue, diagnostiquer
  et prévenir avant de toucher aux hyperparamètres.
- `results/` et `*.log` sont ignorés par git : ne pas les forcer dans un commit.
- Ne rien commiter ni pousser sans demande explicite.

## 1. Préparer

```bash
git fetch origin

git checkout feat/prwn-conv4

git pull

pip install -r requirements.txt

nvidia-smi

python -m pytest -q tests
```

Les 28 tests doivent passer.

Le dataset doit se trouver dans `../ds_FSL_224x204` (à côté du dépôt), déjà redimensionné
par `scripts/resize_dataset.py`. Si ce dossier est ailleurs, passer `--data-root <chemin>`
aux commandes ci-dessous. S'il n'existe pas, s'arrêter et demander.

## 2. Vérifier le split par classes

Le chargement doit afficher `Images retenues : 5451 | classes : 129 | doublons ecartes : 5`.
Si `results/prwn/metrics.json` existe sur le serveur, vérifier que le split est identique
(lire le JSON en UTF-8, sinon les noms avec « ï » semblent différer à tort) :

```bash
python -c "
import json, sys; sys.path.insert(0, '.')
from src.utils.config import load_config
from src.data.dataset import build_index
from src.data.episodes import split_classes
c = load_config('configs/prwn_conv4.yaml')
idx = build_index(c.data_root, c.splits, c.dedup, c.limit_per_class)
counts = {n: int((idx.labels == i).sum()) for i, n in enumerate(idx.class_names)}
s = c.class_split
got = split_classes(counts, s.n_val_classes, s.n_test_classes, s.n_bins, s.seed)
ref = json.load(open('results/prwn/metrics.json', encoding='utf-8'))['classes']
print([sorted(ref[k]) == sorted(v) for k, v in zip(('train', 'val', 'test'), got)])
"
```

Attendu : `[True, True, True]`.

## 3. Lancer les runs

D'abord l'essai rapide sur GPU (quelques secondes, vérifie que CUDA fonctionne) :

```bash
python scripts/train_prwn_conv4.py --config configs/prwn_conv4_smoke.yaml --device cuda
```

Puis les deux runs complets (20 000 pas chacun), dans `tmux` ou avec `nohup` pour
survivre à une déconnexion :

```bash
nohup python scripts/train_prwn_conv4.py --config configs/prwn_conv4.yaml --device cuda > prwn_conv4.log 2>&1 &

nohup python scripts/train_prwn_conv4.py --config configs/prwn_conv4_pn.yaml --device cuda > prwn_conv4_pn.log 2>&1 &
```

Les lancer l'un après l'autre si la mémoire GPU est juste. Une ligne de progression
s'affiche tous les 1 000 pas (loss, acc train, acc val).

Chaque run écrit dans `results/<run_name>/` : `best_model.pt`, `history.csv` et
`metrics.json`. Le meilleur modèle en validation est évalué sur les classes de test,
en 5-way 1-shot et 5-shot, sur 600 épisodes.

## 4. Rapporter

Pour chaque run, donner :

- le meilleur `acc val` et le pas où il est atteint (`best_val`) ;
- la précision de test en 1-shot et 5-shot, avec l'intervalle de confiance à 95 %
  (`test` dans `metrics.json`), et les variantes `accuracy_refined` ;
- la durée totale et tout signe de sur-apprentissage dans `history.csv` (acc train qui
  monte pendant qu'acc val baisse).

Référence DINOv2 gelé + tête linéaire (`results/prwn`, 5-way, 600 épisodes) :

| Run | 1-shot | 5-shot |
|---|---|---|
| PRWN sur DINOv2 | 80,8 % ± 0,8 (affiné 82,1 %) | 90,2 % ± 0,6 (affiné 90,3 %) |
| DINOv2 seul, sans tête | 80,3 % ± 0,8 (affiné 81,2 %) | 91,0 % ± 0,6 (affiné 90,8 %) |

Meilleure val DINOv2 : 85,2 % au pas 7 000.

Questions à trancher avec les résultats :

1. PRWN Conv-4 fait-il mieux que le ProtoNet Conv-4 (effet de la marche aléatoire) ?
2. Quel écart entre le Conv-4 entraîné de zéro et DINOv2 pré-entraîné ?
