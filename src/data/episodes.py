"""Split par classes, split etiquete / non etiquete et tirage des episodes.

Tout est deterministe a graine fixee : deux runs qui partagent la config voient le
meme split de classes, les memes images etiquetees et les memes episodes de test,
ce qui rend leurs scores directement comparables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _stratified_draw(
    counts: dict[str, int], n_draw: int, n_bins: int, rng: np.random.Generator
) -> list[str]:
    """Tire `n_draw` classes en respectant la repartition des tailles de classe.

    Les classes sont triees par nombre d'images puis decoupees en `n_bins` tranches
    d'effectifs egaux. Chaque tranche fournit une part proportionnelle a sa taille
    (methode du plus fort reste pour que le total fasse exactement `n_draw`).
    """
    n_classes = len(counts)
    if not 0 < n_draw < n_classes:
        raise ValueError(f"n_draw doit etre dans ]0, {n_classes}[, recu {n_draw}")
    n_bins = min(n_bins, n_classes)

    # Tri par (taille, nom) : le nom departage les ex aequo de facon stable.
    ordered = sorted(counts, key=lambda name: (counts[name], name))
    bins = [list(chunk) for chunk in np.array_split(np.array(ordered, dtype=object), n_bins)]

    exact = [n_draw * len(chunk) / n_classes for chunk in bins]
    quotas = [int(np.floor(value)) for value in exact]
    remainders = sorted(range(n_bins), key=lambda i: (exact[i] - quotas[i], -i), reverse=True)
    for i in remainders[: n_draw - sum(quotas)]:
        quotas[i] += 1

    drawn: list[str] = []
    for chunk, quota in zip(bins, quotas):
        drawn.extend(str(name) for name in rng.choice(chunk, size=quota, replace=False))
    return drawn


def split_classes(
    counts: dict[str, int], n_val: int, n_test: int, n_bins: int = 5, seed: int = 0
) -> tuple[list[str], list[str], list[str]]:
    """Separe les classes en (entrainement, validation, test), stratifie par taille."""
    rng = np.random.default_rng(seed)
    test = set(_stratified_draw(counts, n_test, n_bins, rng))
    rest = {name: count for name, count in counts.items() if name not in test}
    val = set(_stratified_draw(rest, n_val, n_bins, rng))
    train = sorted(name for name in rest if name not in val)
    return train, sorted(val), sorted(test)


def split_labeled(labels: np.ndarray, ratio: float, seed: int = 0) -> np.ndarray:
    """Masque booleen des images dont on garde l'etiquette, tire classe par classe.

    Les autres forment la reserve non etiquetee. Chaque classe garde au moins une
    image etiquetee.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        positions = rng.permutation(np.flatnonzero(labels == label))
        n_labeled = len(positions) if ratio >= 1 else max(1, round(ratio * len(positions)))
        mask[positions[:n_labeled]] = True
    return mask


@dataclass
class Episode:
    """Indices d'images d'un episode ; -1 dans `unlabeled_class` = distracteur."""

    support: np.ndarray  # (n_way, k_shot)
    query: np.ndarray  # (n_query_total,)
    query_class: np.ndarray  # (n_query_total,) position de la classe dans l'episode
    unlabeled: np.ndarray  # (n_unlabeled_total,)
    unlabeled_class: np.ndarray  # (n_unlabeled_total,) position de la classe dans l'episode


class EpisodeSampler:
    """Tire des episodes N-way K-shot semi-supervises.

    Supports et requetes viennent de la reserve `labeled`, les points non etiquetes
    de la reserve `unlabeled` : une image n'apparait jamais des deux cotes. Pour
    l'evaluation, passer la meme reserve des deux cotes avec `shared_pool=True` :
    les trois groupes sont alors tires sans recouvrement dans les images de la classe.
    """

    def __init__(
        self,
        labeled: dict[int, np.ndarray],
        unlabeled: dict[int, np.ndarray],
        n_way: int,
        k_shot: int,
        n_query: int,
        n_unlabeled: int = 0,
        n_distractor_classes: int = 0,
        shared_pool: bool = False,
    ):
        self.labeled = labeled
        self.unlabeled = unlabeled
        self.n_way, self.k_shot, self.n_query = n_way, k_shot, n_query
        self.n_unlabeled = n_unlabeled
        self.n_distractors = n_distractor_classes if n_unlabeled > 0 else 0
        self.shared_pool = shared_pool

        needed = k_shot + n_query + (n_unlabeled if shared_pool else 0)
        self.eligible = sorted(
            c
            for c, pool in labeled.items()
            if len(pool) >= needed
            and (shared_pool or len(unlabeled.get(c, ())) >= n_unlabeled)
        )
        self.distractor_pool = sorted(
            c for c, pool in unlabeled.items() if len(pool) >= max(n_unlabeled, 1)
        )
        if len(self.eligible) < n_way:
            raise ValueError(
                f"Seulement {len(self.eligible)} classes eligibles pour n_way={n_way} "
                f"(k_shot={k_shot}, n_query={n_query}, n_unlabeled={n_unlabeled}). "
                "Reduis ces valeurs ou augmente le nombre de classes."
            )
        if self.n_distractors and len(self.distractor_pool) < n_way + self.n_distractors:
            raise ValueError("Pas assez de classes pour tirer les distracteurs demandes.")

    def sample(self, rng: np.random.Generator) -> Episode:
        chosen = rng.choice(self.eligible, size=self.n_way, replace=False)
        support, query, unlabeled, unlabeled_class, query_class = [], [], [], [], []
        for position, c in enumerate(chosen):
            n_labeled = self.k_shot + self.n_query
            if self.shared_pool:
                picked = rng.choice(self.labeled[c], size=n_labeled + self.n_unlabeled, replace=False)
                extra = picked[n_labeled:]
            else:
                picked = rng.choice(self.labeled[c], size=n_labeled, replace=False)
                extra = (
                    rng.choice(self.unlabeled[c], size=self.n_unlabeled, replace=False)
                    if self.n_unlabeled
                    else np.empty(0, dtype=np.int64)
                )
            support.append(picked[: self.k_shot])
            query.append(picked[self.k_shot : n_labeled])
            query_class.append(np.full(self.n_query, position))
            unlabeled.append(extra)
            unlabeled_class.append(np.full(self.n_unlabeled, position))

        if self.n_distractors:
            others = [c for c in self.distractor_pool if c not in set(chosen.tolist())]
            for c in rng.choice(others, size=self.n_distractors, replace=False):
                unlabeled.append(rng.choice(self.unlabeled[c], size=self.n_unlabeled, replace=False))
                unlabeled_class.append(np.full(self.n_unlabeled, -1))

        return Episode(
            support=np.stack(support),
            query=np.concatenate(query).astype(np.int64),
            query_class=np.concatenate(query_class).astype(np.int64),
            unlabeled=np.concatenate(unlabeled).astype(np.int64),
            unlabeled_class=np.concatenate(unlabeled_class).astype(np.int64),
        )
