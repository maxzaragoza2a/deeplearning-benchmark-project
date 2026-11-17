# Runs harmonisés sur le serveur GPU

Instructions pour la session Claude lancée sur le serveur SSH. Objectif : entraîner les
quatre runs de la branche `feat/harmonized-training` selon le protocole commun du groupe
(`common/README.md`), puis rapporter leurs scores.

## Contexte

Les trois méthodes du groupe (NCA, ProtoNet, PRWN) sont réévaluées dans les mêmes
conditions : mêmes images (`ds_FSL_224x204`, lues telles quelles en 224x204), même
prétraitement (normalisation ImageNet, aucune augmentation), mêmes dossiers d'origine
(`train` pour apprendre, `valid` pour choisir le checkpoint, `test` une seule fois à la
fin) et mêmes 1 000 épisodes de test figés (`common/test_episodes.json`).

Quatre runs, par ordre de priorité :

| # | Config | Méthode | Backbone |
|---|---|---|---|
| 1 | `configs/harmonized/prwn_resnet12.yaml` | PRWN | ResNet-12 (le même que NCA), entraîné de zéro |
| 2 | `configs/harmonized/prwn_resnet12_pn.yaml` | ProtoNet (contrôle, `lambda_rw: 0`) | ResNet-12 |
| 3 | `configs/harmonized/prwn_conv4.yaml` | PRWN | Conv-4 |
| 4 | `configs/harmonized/prwn_conv4_pn.yaml` | ProtoNet (contrôle) | Conv-4 |

Un run PRWN et son contrôle tirent exactement les mêmes épisodes d'entraînement : seule la
loss de marche aléatoire les distingue.

## Règles

- Ne jamais modifier le dataset, ni `common/test_episodes.json`.
- Ne pas changer les configs pour « améliorer » un score. Si un run échoue, diagnostiquer
  et prévenir avant de toucher aux hyperparamètres.
- Seule exception : en cas de manque de mémoire GPU pendant la validation ou le test, on
  peut baisser `extract_batch_size` (64 par défaut pour le ResNet-12). Cette valeur ne
  change aucun résultat ; le signaler dans le rapport.
- `results/` et `*.log` sont ignorés par git : ne pas les forcer dans un commit.
- Ne rien commiter ni pousser sans demande explicite.

## 1. Préparer

```bash
git fetch origin

git checkout feat/harmonized-training

git pull

pip install -r requirements.txt

nvidia-smi

python -m pytest -q tests
```

Les 38 tests doivent passer.

Le dataset doit se trouver dans `../ds_FSL_224x204` (à côté du dépôt). Si ce dossier est
ailleurs, passer `--data-root <chemin>` à toutes les commandes ci-dessous. S'il n'existe
pas, s'arrêter et demander.

## 2. Vérifier le dataset

Les épisodes de test contiennent le md5 de chaque image : cette commande vérifie que la
copie du serveur est exactement celle du Drive.

```bash
python -c "
import sys; sys.path.insert(0, '.')
from common.protocol import load_episodes, check_dataset
check_dataset('../ds_FSL_224x204', load_episodes())
print('Dataset OK')
"
```

Attendu : `Dataset OK`. Sinon, s'arrêter et rapporter le message d'erreur.

## 3. Essai rapide sur GPU

Quatre pas d'entraînement puis le test complet : vérifie CUDA, la mémoire et toute la
chaîne en quelques minutes.

```bash
python scripts/train_harmonized.py --config configs/harmonized/smoke.yaml --device cuda

python scripts/train_harmonized.py --config configs/harmonized/smoke.yaml --model conv4 --device cuda
```

Chaque commande doit finir par un tableau de 4 scores (`1shot`, `1shot_cl2n`, `5shot`,
`5shot_cl2n`). Les valeurs n'ont pas de sens après 4 pas. Noter la mémoire GPU utilisée
(`nvidia-smi`) pendant l'essai ResNet-12.

## 4. Lancer les runs

Les deux runs ResNet-12 l'un après l'autre (ils sont lourds), dans une seule commande qui
survit à une déconnexion :

```bash
nohup bash -c "python scripts/train_harmonized.py --config configs/harmonized/prwn_resnet12.yaml --device cuda > prwn_resnet12.log 2>&1; python scripts/train_harmonized.py --config configs/harmonized/prwn_resnet12_pn.yaml --device cuda > prwn_resnet12_pn.log 2>&1" > /dev/null 2>&1 &
```

Les runs Conv-4 sont légers. Les lancer en parallèle seulement si l'essai ResNet-12 a
laissé assez de mémoire GPU libre, sinon après les runs ResNet-12 :

```bash
nohup bash -c "python scripts/train_harmonized.py --config configs/harmonized/prwn_conv4.yaml --device cuda > prwn_conv4.log 2>&1; python scripts/train_harmonized.py --config configs/harmonized/prwn_conv4_pn.yaml --device cuda > prwn_conv4_pn.log 2>&1" > /dev/null 2>&1 &
```

Suivre l'avancement :

```bash
tail -f prwn_resnet12.log
```

Une ligne s'affiche tous les 500 pas (loss, acc train, acc val sur `valid`), sur 20 000 pas.
Noter le temps du premier palier de 500 pas pour estimer la durée totale.

Chaque run écrit dans `results/<run_name>/` : `best_model.pt` (meilleur checkpoint sur
`valid`), `history.csv` et `metrics.json` (avec les scores de test dans `test`).

Pour réévaluer un checkpoint sans ré-entraîner :

```bash
python scripts/evaluate_harmonized.py --config configs/harmonized/prwn_resnet12.yaml --device cuda
```

## 5. Rapporter

Pour chaque run :

- le meilleur `acc val` et le pas où il est atteint (`best_val` dans `metrics.json`) ;
- les 4 scores de test avec leur intervalle de confiance à 95 % (`test` dans
  `metrics.json`) ;
- la durée totale, la mémoire GPU utilisée, et tout signe de sur-apprentissage dans
  `history.csv` (acc train qui monte pendant qu'acc val baisse) ;
- toute modification de `extract_batch_size`.

Point de comparaison sur les mêmes épisodes : moyenner les pixels bruts en une grille 8x8,
sans aucun apprentissage, donne déjà 68,3 % ± 1,3 (1-shot) et 80,5 % ± 1,1 (5-shot).

Questions à trancher avec les résultats :

1. PRWN fait-il mieux que son contrôle ProtoNet, avec le ResNet-12 et avec le Conv-4
   (effet de la marche aléatoire) ?
2. Quel écart entre le ResNet-12 et le Conv-4 ?
3. Les modèles entraînés battent-ils nettement le point de comparaison sur les pixels ?
4. Le CL2N change-t-il les scores, et dans le même sens pour tous les runs ?

## 6. Ré-entraîner PRWN ResNet-12 avec SGD

Objectif : comparer SGD à AdamW pour PRWN + ResNet-12, tout le reste étant identique
(données, épisodes, 20 000 pas, validation, test commun). Run AdamW de référence :
`results/harmonized_prwn_resnet12` (81,22 % en 1-shot, 89,58 % en 5-shot).

Momentum 0,9, Nesterov et weight decay 5e-4 sont ceux du run NCA. Le taux d'apprentissage
de NCA (0,1) vise des lots de 128 images, alors que nos épisodes en ont 10 : il est choisi
par un balayage court, **jugé uniquement sur `valid`** (le test n'est jamais lancé).

### 6.1 Balayage (6 candidats x 4 000 pas, environ 30 min)

```bash
nohup python scripts/sweep_optimizer.py --device cuda > sweep_optimizer.log 2>&1 &

tail -f sweep_optimizer.log
```

Candidats : AdamW 1e-3 (référence), SGD 0,1 avec palier x0,1 à 70 % (réglages exacts de
NCA), et SGD en cosinus à 0,1, 0,03, 0,01 et 0,003. Le classement s'affiche à la fin et
est écrit dans `results/sweep_optimizer/summary.csv`.

Choisir le meilleur candidat SGD sur `best val`. Si plusieurs sont à moins d'un
intervalle de confiance du meilleur, prendre le planning cosinus. Un candidat `failed`
a divergé (loss non finie) : l'écarter.

### 6.2 Run complet avec le meilleur réglage SGD

Reporter le taux (et le planning s'il gagne en `step`) dans
`configs/harmonized/prwn_resnet12_sgd.yaml`, puis :

```bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_harmonized.py --config configs/harmonized/prwn_resnet12_sgd.yaml --device cuda > prwn_resnet12_sgd.log 2>&1 &
```

Résultats dans `results/harmonized_prwn_resnet12_sgd/`. Rapporter les 4 scores de test à
côté de ceux du run AdamW, avec le classement du balayage.
