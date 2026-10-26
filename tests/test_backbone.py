"""Tests de l'extraction d'embeddings, du cache et de la tete.

Hors ligne : un mini DINOv2 a poids aleatoires remplace le checkpoint pre-entraine.
"""

from pathlib import Path

import pytest
import torch
from transformers import Dinov2Config, Dinov2Model

from src.models import backbone
from src.models.backbone import ProjectionHead, cached_embeddings, extract_embeddings


def tiny_dinov2():
    torch.manual_seed(0)
    config = Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)
    return Dinov2Model(config).eval()


def test_extract_embeddings_takes_uint8_images():
    images = torch.randint(0, 256, (5, 3, 28, 42), dtype=torch.uint8)
    embeddings = extract_embeddings(tiny_dinov2(), images, "cpu", batch_size=2)
    assert embeddings.shape == (5, 32) and embeddings.dtype == torch.float32
    # Le decoupage en lots ne change pas le resultat.
    assert torch.allclose(embeddings, extract_embeddings(tiny_dinov2(), images, "cpu", batch_size=5), atol=1e-5)


def test_extract_embeddings_rejects_sizes_that_are_not_multiples_of_the_patch():
    with pytest.raises(ValueError, match="multiple de 14"):
        extract_embeddings(tiny_dinov2(), torch.zeros(1, 3, 28, 30, dtype=torch.uint8), "cpu")


def test_cache_is_reused_and_depends_on_images(monkeypatch, tmp_path):
    monkeypatch.setattr(backbone, "load_backbone", lambda name: tiny_dinov2())
    images = torch.randint(0, 256, (3, 3, 28, 28), dtype=torch.uint8)
    calls = []

    def load():
        calls.append(1)
        return images

    paths = [Path(f"img_{i}.jpg") for i in range(3)]
    first = cached_embeddings("dinov2_s", paths, (28, 28), load, "cpu", tmp_path)
    second = cached_embeddings("dinov2_s", paths, (28, 28), load, "cpu", tmp_path)
    assert len(calls) == 1 and torch.equal(first, second)
    cached_embeddings("dinov2_s", paths[::-1], (28, 28), load, "cpu", tmp_path)
    assert len(calls) == 2 and len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize("hidden_dim", [None, 16])
def test_head_is_trainable(hidden_dim):
    head = ProjectionHead(32, out_dim=8, hidden_dim=hidden_dim)
    out = head(torch.randn(4, 32))
    assert out.shape == (4, 8)
    out.sum().backward()
    assert all(p.grad is not None for p in head.parameters())
