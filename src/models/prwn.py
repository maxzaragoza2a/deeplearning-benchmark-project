"""Prototypical Random Walk Networks (Ayyad et al., 2021).

Objectif d'entrainement : L_S + lambda * L_RW, avec
- L_S  : entropie croisee prototypique sur les requetes etiquetees (Snell et al.) ;
- L_RW : un marcheur part d'un prototype, fait tau pas parmi les points non
  etiquetes et doit revenir sur son prototype de depart (L_walker), en visitant
  tous les points non etiquetes de facon uniforme (L_visit).

Ce module ne contient que des fonctions sur des embeddings : il ne depend pas du
backbone. Les similarites sont des distances euclidiennes au carre negatives.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def squared_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distances euclidiennes au carre entre lignes de `a` (n, d) et de `b` (m, d)."""
    # Forme developpee plutot que torch.cdist : pas de racine, donc pas de gradient
    # infini quand deux points coincident.
    distances = a.pow(2).sum(1, keepdim=True) + b.pow(2).sum(1) - 2 * a @ b.T
    return distances.clamp_min(0)


def compute_prototypes(support: torch.Tensor) -> torch.Tensor:
    """Prototypes (n_way, d) = moyenne des supports etiquetes (n_way, k_shot, d)."""
    return support.mean(dim=1)


def prototype_logits(points: torch.Tensor, prototypes: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Logits (n, n_way) d'appartenance de chaque point a chaque prototype."""
    return -scale * squared_distances(points, prototypes)


def landing_matrices(
    prototypes: torch.Tensor,
    unlabeled: torch.Tensor,
    n_steps: int = 3,
    smoothing: float = 0.1,
    scale: float = 1.0,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Matrices T^(0..n_steps) du marcheur et distribution de visite.

    T^(i) = G(p->x) . G(x->x)^i . G(x->p) ; T^(i)[a, b] est la probabilite de finir
    sur le prototype b en partant du prototype a apres i pas entre points non
    etiquetes. Renvoie aussi la probabilite de visite de chaque point (m,).

    Comme dans le code de reference, les transitions prototype <-> point sont
    lissees vers l'uniforme (`smoothing`), ce qui borne les logarithmes.
    """
    n_points, n_way = unlabeled.shape[0], prototypes.shape[0]
    if n_points < 2:
        raise ValueError("La marche aleatoire demande au moins 2 points non etiquetes.")

    point_proto = prototype_logits(unlabeled, prototypes, scale)  # A, (m, n_way)
    point_point = -scale * squared_distances(unlabeled, unlabeled)  # B, (m, m)
    # Pas de boucle sur soi-meme : le marcheur doit changer de point a chaque pas.
    self_loops = torch.eye(n_points, dtype=torch.bool, device=unlabeled.device)
    point_point = point_point.masked_fill(self_loops, float("-inf"))

    x_to_p = (1 - smoothing) * F.softmax(point_proto, dim=1) + smoothing / n_way
    p_to_x = ((1 - smoothing) * F.softmax(point_proto, dim=0) + smoothing / n_points).T
    x_to_x = F.softmax(point_point, dim=1)

    matrices = []
    walk = p_to_x  # (n_way, m) : position du marcheur apres i pas
    for step in range(n_steps + 1):
        matrices.append(walk @ x_to_p)
        if step < n_steps:
            walk = walk @ x_to_x
    return matrices, p_to_x.mean(dim=0)


def random_walk_loss(
    prototypes: torch.Tensor,
    unlabeled: torch.Tensor,
    n_steps: int = 3,
    decay: float = 0.7,
    smoothing: float = 0.1,
    visit_weight: float = 1.0,
    scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L_RW = L_walker + visit_weight * L_visit, et les probabilites de retour moyennes."""
    matrices, visit = landing_matrices(prototypes, unlabeled, n_steps, smoothing, scale)

    # L_walker = somme_i decay^i * H(I, T^(i)) : on pousse chaque T^(i) vers l'identite.
    walker = sum(
        decay**step * -torch.log(matrix.diagonal()).mean()
        for step, matrix in enumerate(matrices)
    )
    # L_visit = H(U, P) : entropie croisee entre l'uniforme et la distribution de visite.
    visit_loss = -torch.log(visit).mean()

    stats = {f"landing_{step}": matrix.diagonal().mean().item() for step, matrix in enumerate(matrices)}
    stats["walker"] = walker.item()
    stats["visit"] = visit_loss.item()
    return walker + visit_weight * visit_loss, stats


def prwn_loss(
    support: torch.Tensor,
    query: torch.Tensor,
    unlabeled: torch.Tensor | None,
    lambda_rw: float = 0.5,
    n_steps: int = 3,
    decay: float = 0.7,
    smoothing: float = 0.1,
    visit_weight: float = 1.0,
    scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss d'un episode.

    support   : (n_way, k_shot, d)
    query     : (n_way, n_query, d), rangees par classe
    unlabeled : (m, d) ou None ; ignore si `lambda_rw` vaut 0 (Prototypical Network).
    """
    n_way, n_query, dim = query.shape
    prototypes = compute_prototypes(support)
    logits = prototype_logits(query.reshape(-1, dim), prototypes, scale)
    targets = torch.arange(n_way, device=query.device).repeat_interleave(n_query)

    loss = F.cross_entropy(logits, targets)
    stats = {
        "supervised": loss.item(),
        "accuracy": (logits.argmax(dim=1) == targets).float().mean().item(),
    }
    if lambda_rw > 0 and unlabeled is not None:
        rw, rw_stats = random_walk_loss(
            prototypes, unlabeled, n_steps, decay, smoothing, visit_weight, scale
        )
        loss = loss + lambda_rw * rw
        stats.update(rw_stats)
    stats["loss"] = loss.item()
    return loss, stats


def refine_prototypes(
    prototypes: torch.Tensor, support: torch.Tensor, unlabeled: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """Inference semi-supervisee : un pas de soft k-means (Ren et al., 2018).

    Les supports comptent pour 1 dans leur classe, chaque point non etiquete est
    reparti entre les classes selon sa proximite aux prototypes.
    """
    assignment = F.softmax(prototype_logits(unlabeled, prototypes, scale), dim=1)  # (m, n_way)
    total = support.sum(dim=1) + assignment.T @ unlabeled
    weight = support.shape[1] + assignment.sum(dim=0)
    return total / weight.unsqueeze(1)
