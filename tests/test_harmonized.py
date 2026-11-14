"""Tests of the shared protocol (common/) and of the harmonized pipeline.

None of them needs the dataset: images are synthetic, the episode file is read as committed.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from common.protocol import (
    EPISODES_FILE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, TRANSFORM, cl2n, episode_accuracies,
    load_episodes, load_image,
)
from src.data.dataset import DatasetIndex, load_images
from src.models.conv4 import ImagePreprocessor
from src.models.resnet12 import ResNet12


@pytest.fixture(scope="module")
def episodes():
    return load_episodes(EPISODES_FILE)


def random_image(size=IMAGE_SIZE, seed=0) -> Image.Image:
    pixels = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(pixels)


# --------------------------------------------------------------------------- episode file

def test_episode_file_matches_the_group_protocol(episodes):
    protocol = episodes["protocol"]
    assert protocol["n_way"] == 5 and protocol["shots"] == [1, 5] and protocol["n_query"] == 1
    assert protocol["n_episodes"] == len(episodes["episodes"]) == 1000
    assert protocol["image_size"] == list(IMAGE_SIZE)


def test_every_episode_is_well_formed(episodes):
    per_class = episodes["protocol"]["min_images_per_class"]
    n_images = len(episodes["images"])
    for episode in episodes["episodes"]:
        assert len(set(episode["classes"])) == 5
        flat = [i for picks in episode["images"] for i in picks]
        assert len(flat) == len(set(flat)) == 5 * per_class  # no image twice in an episode
        assert all(0 <= i < n_images for i in flat)
        for name, picks in zip(episode["classes"], episode["images"]):
            assert all(episodes["images"][i]["path"].startswith(name + "/") for i in picks)


# --------------------------------------------------------------------------- preprocessing

def test_transform_is_totensor_then_imagenet_normalize():
    image = random_image()
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
    expected = (pixels - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
    output = TRANSFORM(image)
    assert output.shape == (3, IMAGE_SIZE[1], IMAGE_SIZE[0])
    assert torch.allclose(output, expected, atol=1e-6)


def test_training_preprocessing_equals_the_shared_transform():
    # The training script normalizes uint8 tensors on the GPU: it must match TRANSFORM exactly.
    image = random_image(seed=1)
    preprocess = ImagePreprocessor(torch.tensor(IMAGENET_MEAN), torch.tensor(IMAGENET_STD), hflip=False).eval()
    uint8 = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0)
    assert torch.allclose(preprocess(uint8)[0], TRANSFORM(image), atol=1e-6)


def test_wrong_image_size_is_refused(tmp_path):
    path = tmp_path / "wrong.jpg"
    random_image(size=(224, 224)).save(path)
    with pytest.raises(ValueError, match="expected 224x204"):
        load_image(path)
    index = DatasetIndex([path], np.zeros(1, dtype=np.int64), ["a"], [])
    with pytest.raises(ValueError, match="expected 224x204"):
        load_images(index, IMAGE_SIZE, letterbox=False)


def test_images_at_the_right_size_are_loaded_untouched(tmp_path):
    path = tmp_path / "right.png"  # lossless, to compare pixels exactly
    image = random_image(seed=2)
    image.save(path)
    index = DatasetIndex([path], np.zeros(1, dtype=np.int64), ["a"], [])
    loaded = load_images(index, IMAGE_SIZE, letterbox=False)
    assert torch.equal(loaded[0], torch.from_numpy(np.array(image)).permute(2, 0, 1))


# --------------------------------------------------------------------------- evaluation

def test_separable_embeddings_give_perfect_accuracy(episodes):
    # One direction per class: every query is closest to its own prototype.
    classes = sorted({image["path"].split("/")[0] for image in episodes["images"]})
    label = {name: i for i, name in enumerate(classes)}
    embeddings = torch.eye(len(classes))[[label[image["path"].split("/")[0]] for image in episodes["images"]]]
    for n_shot in (1, 5):
        assert episode_accuracies(embeddings, episodes, n_shot).mean() == 1.0


def test_random_embeddings_give_chance_accuracy(episodes):
    embeddings = torch.randn(len(episodes["images"]), 32, generator=torch.Generator().manual_seed(0))
    assert abs(episode_accuracies(embeddings, episodes, 1).mean() - 0.2) < 0.03


def test_cl2n_centers_then_gives_unit_norm():
    embeddings = torch.randn(10, 8) + 5
    mean = embeddings.mean(dim=0, keepdim=True)
    out = cl2n(embeddings, mean)
    assert torch.allclose(out.norm(dim=1), torch.ones(10), atol=1e-6)
    assert torch.allclose(out, (embeddings - mean) / (embeddings - mean).norm(dim=1, keepdim=True), atol=1e-6)


# --------------------------------------------------------------------------- backbone

def test_resnet12_gives_640_dimensions_at_the_dataset_size():
    model = ResNet12().eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])).shape == (2, 640)
    assert model.out_dim == 640
