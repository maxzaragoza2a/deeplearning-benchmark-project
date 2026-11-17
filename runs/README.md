# PRWN runs under the shared protocol

Light outputs of the PRWN runs (metrics, training curves, logs; no checkpoints).
All runs follow `common/README.md`: `ds_FSL_224x204` read as is, ImageNet normalization,
no augmentation, original `train` / `valid` / `test` folders, model selection on `valid`,
test once on the 1000 shared episodes of `common/test_episodes.json`.

Backbone: ResNet-12 (same as NCA) trained from scratch. Training episodes: 5-way 1-shot,
1 query per class, 20,000 steps; the random walk goes through the query embeddings.

## Results (test, 5-way, mean +/- 95% CI over 1000 episodes)

| Run | Optimizer | 1-shot | 5-shot | 1-shot + CL2N | 5-shot + CL2N | Best valid |
| --- | --- | --- | --- | --- | --- | --- |
| **`harmonized_prwn_resnet12_sgd`** (main result) | SGD lr 0.003, momentum 0.9, Nesterov, wd 5e-4, cosine | **84.32 +/- 0.99** | **92.82 +/- 0.73** | 84.16 +/- 1.01 | 92.82 +/- 0.72 | 87.20 (step 12,000) |
| `harmonized_prwn_resnet12` (variant) | AdamW lr 1e-3, wd 1e-4, cosine | 81.22 +/- 1.05 | 89.58 +/- 0.87 | 79.50 +/- 1.10 | 88.56 +/- 0.90 | 86.00 (step 8,500) |

Reference on the same episodes, with no training at all: raw pixels averaged into an
8x8 grid give 68.3 +/- 1.3 % (1-shot) and 80.5 +/- 1.1 % (5-shot).

## How the SGD learning rate was chosen: `sweep_optimizer/`

`scripts/sweep_optimizer.py`: 4,000-step runs, same episodes, ranked on `valid` only (the
test was never run for these). `summary.csv` holds the ranking:

| Candidate | Best valid (%) |
| --- | --- |
| AdamW 1e-3 | 85.08 |
| SGD 0.003 | 84.20 |
| SGD 0.0003 | 84.16 |
| SGD 0.001 | 83.96 |
| SGD 0.01 | 83.28 |
| SGD 0.1, cosine | 79.36 |
| SGD 0.1, x0.1 at 70% (exact NCA settings) | 77.92 |
| SGD 0.03 | 75.92 |

SGD 0.0003-0.003 tie (~84 %, within the CI); 0.003 was kept as the best and the only one
still improving at the end. The NCA learning rate (0.1, tuned for batches of 128 images) is
too high for 10-image episodes.

## Caveats

- One seed per run. The SGD recipe also changes the weight decay (5e-4 vs 1e-4): the gain
  is that of the whole recipe, not of SGD alone.
- The SGD run was accidentally launched twice at the same time (same config, same seed).
  Both runs reached their best validation (87.20 %) at step 12,000; the files here come
  from the second one, whose step-12,000 checkpoint is the one both runs tested. The two
  validation curves differ by 1-3 points at some steps: GPU non-determinism, a useful
  reminder that validation gaps below ~1.5 points are noise.
