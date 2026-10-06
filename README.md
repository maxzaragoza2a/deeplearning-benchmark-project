# deeplearning-benchmark-project
# Prototypical Learning Benchmark — Flying Insect Wing Classification

> Second-year engineering project (ENSEA, Signal Processing & Computer Science) carried out with the **ETIS laboratory**.
> Goal: benchmark three prototypical few-shot learning methods from the literature on a lab-owned dataset of insect wing images, and determine which one is the most effective.

---

## Table of contents

1. [Context & motivation](#context--motivation)
2. [Project pipeline](#project-pipeline)
3. [Phase 1 — Dataset analysis](#phase-1--dataset-analysis)
4. [Phase 2 — Embedding backbone](#phase-2--embedding-backbone)
5. [Phase 3 — Implementation of the three models](#phase-3--implementation-of-the-three-models)
6. [Phase 4 — Benchmark](#phase-4--benchmark)
7. [Repository structure](#repository-structure)
8. [Getting started](#getting-started)
9. [References](#references)

---

## Context & motivation

Prototypical learning is an **interpretable alternative to black-box deep learning classifiers**: each class is represented by one (or several) prototypes in an embedding space, and a sample is assigned to the class of its nearest prototype.

Previous work at ETIS on insect images suggested that **using the embedding space learned by a conventional CNN was more relevant than directly learning prototypes on the dataset**. This project builds a benchmark to test how well that observation holds, by comparing three reference prototypical methods on top of a shared embedding backbone.

The task: classify flying insects (mosquitoes, tsetse flies, sandflies, biting midges…) **at species level from images of their wings** — a fine-grained, heavily imbalanced, low-data problem that is a natural fit for few-shot learning.

---

## Project pipeline

```
 ┌──────────────────┐    ┌──────────────────────┐    ┌────────────────────────┐    ┌──────────────┐
 │ 1. Dataset audit │ ─▶ │ 2. Embedding backbone│ ─▶ │ 3. Three prototypical  │ ─▶ │ 4. Benchmark │
 │  integrity,      │    │  model survey →      │    │  models from the papers│    │  same splits,│
 │  splits, balance │    │  DINOv2, fine-tuned  │    │  ProtoNet / NCA / PRW  │    │  same metrics│
 └──────────────────┘    └──────────────────────┘    └────────────────────────┘    └──────────────┘
```

---

## Phase 1 — Dataset analysis

Before any training, the raw dataset (`ds_FSL`) was fully audited with a dedicated script, producing `images.csv` (per-image report), `classes.csv` (per-class counts) and `doublons.csv` (exact duplicates).

### Overview

| Property | Value |
|---|---|
| Images | **5,456** (all JPEG, RGB) |
| Classes (species) | **129**, across 20 genera |
| Split | train 1,818 / valid 1,819 / test 1,819 |
| Images per class | min **3**, median **10**, mean 42.3, max **608** |
| Resolution | 789 distinct sizes, ~1306 × 545 px on average (elongated wing crops) |

The most represented genera are *Glossina* (22 species, 1,742 images), *Phlebotomus* (11 species, 1,265), *Anopheles* (42 species, 843), *Culex* (10 species, 588) and *Aedes* (22 species, 494).

### Key findings

**1. Split by images, not by classes.**
All 129 classes appear in train, valid and test. This is a standard supervised split, *not* a few-shot protocol with disjoint base/novel classes. The episodic evaluation therefore required building our own class-disjoint splits (see Phase 4).

**2. Severe class imbalance (long tail).**
59 classes have fewer than 10 images, while 20 classes have 100+. *Glossina palpalis palpalis* alone accounts for 608 images (~11 % of the dataset). This long tail is the main motivation for a few-shot approach.

**3. Massive file corruption — 85 % of images truncated.**
4,664 images out of 5,456 fail to decode fully (`OSError: image file is truncated`). Every one of them has a file size that is an exact power of two (4,483 files at 262,144 B, 181 at 524,288 B) — a clear signature of an **interrupted copy / transfer**, not of corrupt acquisitions. Only 792 images are intact (JPEG EOI marker present). The corruption is spread evenly across splits (~14–15 % valid images in each).

**4. Minor issues.**
- 2 groups of exact duplicates (MD5), within the same split and class — no train/test leakage.
- Class name inconsistencies: a double space (`Culex  quinquefaciatus`), non-ASCII characters (`Culicoïdes`, `Glossina tachinoïdes`).
- Probable duplicate classes from typos: `Glossina caliginea` ~ `Glossina calliginea`, `Glossina pallidipes` ~ `Glossina pallidipides` (and `Glossina morsitans morsitans` ~ `submorsitans`, to be confirmed with the lab — these may be genuine subspecies).

> **Action taken:** <!-- TODO: describe the fix, e.g. "a clean copy of the dataset was re-exported by the lab" / "truncated images were decoded with PIL's LOAD_TRUNCATED_IMAGES" / "class names were normalised and merged" -->

---

## Phase 2 — Embedding backbone

All three prototypical methods rely on an embedding function *f*<sub>θ</sub> that maps an image to a feature vector. To isolate the contribution of each method, **the same backbone is shared across the benchmark**.

### Model survey

Several well-known pretrained vision models were first studied and compared in a separate repository:
➡️ **[<!-- TODO: repo name -->](<!-- TODO: link to the embedding-study repo -->)**

<!-- TODO: fill with the models actually compared and their scores -->
| Model | Architecture | Pretraining | Notes / result |
|---|---|---|---|
| DINOv2 | ViT | Self-supervised (LVD-142M) | **Selected** |
| … | … | … | … |

### Selected approach

**DINOv2** was retained as the embedding backbone, then **retrained and fine-tuned on the ETIS insect-wing dataset** so that the embedding space captures fine-grained venation and wing-pattern features specific to the task.

<!-- TODO: fine-tuning details: frozen layers, loss, input resolution / aspect-ratio handling, augmentations, epochs, embedding dimension -->

---

## Phase 3 — Implementation of the three models

The three methods from the reference bibliography were implemented on top of the shared backbone.

### 3.1 Prototypical Networks — Snell et al., NeurIPS 2017

The baseline. Training is **episodic**: each episode samples *N* classes with *K* labelled support images each. Each class prototype is the **mean of its support embeddings**, and query images are classified by a softmax over negative squared Euclidean distances to the prototypes.

- `src/models/protonet.py` <!-- TODO: adjust path -->

### 3.2 Episode-free training with NCA — Laenen & Bertinetto, NeurIPS 2021

This paper shows that episodic training is a **wasteful** way to learn with a prototypical-style loss, since it discards many useful pairwise comparisons within a batch. It replaces episodes with the **Neighbourhood Component Analysis (NCA)** loss, which exploits *all* pairwise distances in a standard mini-batch. At test time, the same nearest-centroid classifier is used.

- `src/models/nca.py` <!-- TODO: adjust path -->

### 3.3 Prototypical Random Walk Networks (PRWN) — Ayyad et al., PMLR 2021

A **semi-supervised** extension that leverages **unlabelled images**. On top of the prototypical loss, a *random-walk* loss encourages walks that start at a prototype, pass through unlabelled embeddings, and return to the same prototype — which pulls unlabelled points into compact class clusters. Particularly relevant here given the small number of labelled images for most species.

- `src/models/prwn.py` <!-- TODO: adjust path -->

---

## Phase 4 — Benchmark

### Protocol

<!-- TODO: adjust to the actual protocol -->
- **Class-disjoint splits** rebuilt from the audited dataset (base classes for training, novel classes for evaluation), since the original split is image-based.
- **Episodic evaluation**: *N*-way *K*-shot tasks (e.g. 5-way 1-shot and 5-way 5-shot), averaged over **<!-- TODO -->** test episodes, reported with 95 % confidence intervals.
- Same backbone, same preprocessing and same seeds for all three methods.
- Additional supervised baseline: nearest-centroid classifier on the frozen fine-tuned DINOv2 embeddings, to test the lab's hypothesis that a well-trained embedding space alone is already competitive.

### Results

<!-- TODO: fill in -->
| Method | 5-way 1-shot | 5-way 5-shot | Training cost | Uses unlabelled data |
|---|---|---|---|---|
| Nearest centroid on DINOv2 (baseline) | – | – | – | No |
| ProtoNet (Snell 2017) | – | – | – | No |
| NCA, episode-free (Laenen 2021) | – | – | – | No |
| PRWN (Ayyad 2021) | – | – | – | Yes |

### Conclusion

<!-- TODO: which of the three is the most effective, by how much, and under which conditions (shots, rare vs common classes) — and whether the lab's observation about CNN/pretrained embeddings is confirmed -->

---

## Repository structure

<!-- TODO: adjust to the actual tree -->
```
.
├── data_audit/          # Phase 1 — audit script + reports (images.csv, classes.csv, doublons.csv, rapport.txt)
├── embedding/           # Phase 2 — DINOv2 fine-tuning
├── src/
│   ├── models/          # Phase 3 — protonet.py, nca.py, prwn.py
│   ├── data/            # datasets, episodic samplers, splits
│   └── utils/
├── benchmark/           # Phase 4 — evaluation scripts and results
├── requirements.txt
└── README.md
```

---

## Getting started

```bash
git clone https://github.com/maxzaragoza2a/<!-- TODO: repo-name -->.git
cd <!-- TODO: repo-name -->
pip install -r requirements.txt
```

<!-- TODO: commands to run the audit, fine-tune the backbone, train each model and run the benchmark -->

> The insect-wing dataset belongs to the ETIS laboratory and is **not distributed** with this repository.

---

## References

1. J. Snell, K. Swersky, R. Zemel. *Prototypical Networks for Few-shot Learning.* NeurIPS 30, 2017.
2. S. Laenen, L. Bertinetto. *On Episodes, Prototypical Networks, and Few-Shot Learning.* NeurIPS 34, 2021, pp. 24581–24592.
3. A. Ayyad, Y. Li, R. Muaz, S. Albarqouni, M. Elhoseiny. *Semi-Supervised Few-Shot Learning with Prototypical Random Walks.* AAAI Workshop on Meta-Learning and MetaDL Challenge, PMLR 140, 2021, pp. 45–57.
4. M. Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR, 2024.

---

*Project carried out at ENSEA in collaboration with the ETIS laboratory (CY Cergy Paris Université / ENSEA / CNRS).*
