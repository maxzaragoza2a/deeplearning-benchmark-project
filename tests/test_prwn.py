"""Tests de la loss PRW sur des embeddings synthetiques."""

import pytest
import torch

from src.models.prwn import (
    compute_prototypes,
    landing_matrices,
    prwn_loss,
    random_walk_loss,
    refine_prototypes,
    squared_distances,
)


def make_clusters(n_way=4, n_points=6, dim=8, spread=0.05, seed=0):
    """Prototypes bien separes et points non etiquetes groupes autour de chacun."""
    generator = torch.Generator().manual_seed(seed)
    centers = 5 * torch.randn(n_way, dim, generator=generator)
    points = centers.repeat_interleave(n_points, dim=0)
    points = points + spread * torch.randn(points.shape, generator=generator)
    return centers, points


def test_squared_distances_matches_cdist():
    a, b = torch.randn(5, 7), torch.randn(3, 7)
    assert torch.allclose(squared_distances(a, b), torch.cdist(a, b) ** 2, atol=1e-4)


def test_landing_matrices_are_stochastic():
    prototypes, points = make_clusters()
    matrices, visit = landing_matrices(prototypes, points, n_steps=3)
    assert len(matrices) == 4
    for matrix in matrices:
        assert matrix.shape == (4, 4)
        assert torch.allclose(matrix.sum(dim=1), torch.ones(4), atol=1e-5)
    assert visit.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_walker_returns_home_on_separated_clusters():
    prototypes, points = make_clusters()
    matrices, _ = landing_matrices(prototypes, points, n_steps=3, smoothing=0.0)
    for matrix in matrices:
        assert matrix.diagonal().min().item() > 0.99


def test_loss_is_lower_when_clusters_match_prototypes():
    prototypes, points = make_clusters()
    good, _ = random_walk_loss(prototypes, points)
    # Memes points, mais sans structure de classe autour des prototypes.
    scattered = 5 * torch.randn(points.shape, generator=torch.Generator().manual_seed(1))
    bad, _ = random_walk_loss(prototypes, scattered)
    assert good.item() < bad.item()


def test_visit_loss_penalises_ignored_points():
    prototypes, points = make_clusters()
    _, balanced = random_walk_loss(prototypes, points)
    # On eloigne la moitie des points : le marcheur ne les visite plus.
    far = points.clone()
    far[::2] += 100.0
    _, unbalanced = random_walk_loss(prototypes, far)
    assert unbalanced["visit"] > balanced["visit"]


def test_gradients_are_finite_even_with_duplicate_points():
    prototypes, points = make_clusters()
    points[1] = points[0]  # distance nulle entre deux points
    support = prototypes.unsqueeze(1).clone().requires_grad_(True)
    unlabeled = points.clone().requires_grad_(True)
    query = prototypes.unsqueeze(1).repeat(1, 3, 1)
    loss, stats = prwn_loss(support, query, unlabeled, lambda_rw=0.5)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(support.grad).all() and torch.isfinite(unlabeled.grad).all()
    assert unlabeled.grad.abs().sum().item() > 0
    assert {"supervised", "walker", "visit", "landing_0", "landing_3"} <= set(stats)


def test_lambda_zero_is_a_plain_prototypical_network():
    prototypes, points = make_clusters()
    support = prototypes.unsqueeze(1)
    query = prototypes.unsqueeze(1).repeat(1, 3, 1) + 0.1
    with_unlabeled, stats = prwn_loss(support, query, points, lambda_rw=0.0)
    without, _ = prwn_loss(support, query, None, lambda_rw=0.5)
    assert with_unlabeled.item() == pytest.approx(without.item())
    assert "walker" not in stats
    assert stats["accuracy"] == 1.0


def test_refine_prototypes_moves_towards_unlabeled_cluster():
    centers, points = make_clusters()
    support = (centers + 1.0).unsqueeze(1)  # supports decales par rapport au vrai centre
    prototypes = compute_prototypes(support)
    refined = refine_prototypes(prototypes, support, points)
    before = (prototypes - centers).norm(dim=1)
    after = (refined - centers).norm(dim=1)
    assert (after < before).all()
