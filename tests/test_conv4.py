"""Tests du backbone Conv-4, de son pretraitement et de son entrainement avec la loss PRW."""

import pytest
import torch

from src.models.conv4 import Conv4, ImagePreprocessor, channel_stats, embed_images
from src.models.prwn import prwn_loss


@pytest.mark.parametrize("size", [(84, 84), (224, 224), (204, 224)])
def test_embedding_dimension_does_not_depend_on_input_size(size):
    model = Conv4().eval()
    embeddings = model(torch.randn(2, 3, *size))
    assert embeddings.shape == (2, 1600) and model.out_dim == 1600


def test_without_pooling_84x84_gives_the_paper_dimension():
    # 84 -> 42 -> 21 -> 10 -> 5 : 64 x 5 x 5, comme dans Snell et al.
    model = Conv4(pool_size=None).eval()
    assert model(torch.randn(1, 3, 84, 84)).shape == (1, 1600) and model.out_dim is None


def test_channel_stats_match_a_direct_computation():
    images = torch.randint(0, 256, (7, 3, 6, 5), dtype=torch.uint8)
    mean, std = channel_stats(images, chunk=3)
    floats = images.double() / 255.0
    assert torch.allclose(mean.double(), floats.mean(dim=(0, 2, 3)), atol=1e-6)
    assert torch.allclose(std.double(), floats.std(dim=(0, 2, 3), unbiased=False), atol=1e-6)


def test_preprocessor_augments_only_in_training():
    images = torch.randint(0, 256, (4, 3, 16, 16), dtype=torch.uint8)
    preprocess = ImagePreprocessor(torch.zeros(3), torch.ones(3), hflip=True, max_shift=4)
    preprocess.eval()
    assert torch.equal(preprocess(images), images.float() / 255.0)
    preprocess.train()
    torch.manual_seed(0)
    augmented = preprocess(images)
    assert augmented.shape == images.shape and not torch.equal(augmented, images.float() / 255.0)


def test_embed_images_is_deterministic_and_batch_independent():
    torch.manual_seed(0)
    model = Conv4(hidden_channels=8)
    preprocess = ImagePreprocessor(torch.full((3,), 0.5), torch.full((3,), 0.25), max_shift=4)
    images = torch.randint(0, 256, (5, 3, 32, 32), dtype=torch.uint8)
    a = embed_images(model, preprocess, images, "cpu", batch_size=2)
    b = embed_images(model, preprocess, images, "cpu", batch_size=5)
    assert a.shape == (5, 8 * 25) and torch.allclose(a, b, atol=1e-5)


def test_prw_loss_trains_the_whole_backbone():
    torch.manual_seed(0)
    n_way, k_shot, n_query, n_unlabeled = 3, 1, 2, 2
    model = Conv4(hidden_channels=8).train()
    images = torch.randn(n_way * (k_shot + n_query + n_unlabeled), 3, 32, 32)
    embeddings = model(images)
    n_support, n_q = n_way * k_shot, n_way * n_query
    loss, stats = prwn_loss(
        embeddings[:n_support].view(n_way, k_shot, -1),
        embeddings[n_support : n_support + n_q].view(n_way, n_query, -1),
        embeddings[n_support + n_q :],
        lambda_rw=0.5,
    )
    loss.backward()
    assert torch.isfinite(loss) and "walker" in stats
    # Le gradient remonte jusqu'a la premiere convolution.
    first_conv = model.encoder[0][0]
    assert first_conv.weight.grad is not None and first_conv.weight.grad.abs().sum() > 0
