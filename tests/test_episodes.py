"""Tests du split par classes, du split etiquete et du tirage des episodes."""

import numpy as np
import pytest

from src.data.episodes import EpisodeSampler, split_classes, split_labeled


def make_counts(n_classes=40):
    return {f"classe_{i:02d}": 3 + 5 * i for i in range(n_classes)}


def test_split_classes_is_disjoint_complete_and_deterministic():
    counts = make_counts()
    train, val, test = split_classes(counts, n_val=6, n_test=12, seed=0)
    assert (len(train), len(val), len(test)) == (22, 6, 12)
    assert set(train) | set(val) | set(test) == set(counts)
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
    assert split_classes(counts, n_val=6, n_test=12, seed=0) == (train, val, test)
    assert split_classes(counts, n_val=6, n_test=12, seed=1) != (train, val, test)


def test_split_classes_is_stratified_by_size():
    counts = make_counts()
    _, _, test = split_classes(counts, n_val=5, n_test=10, n_bins=5, seed=0)
    ordered = sorted(counts, key=counts.get)
    # 10 classes sur 40 et 5 tranches de 8 : exactement 2 classes de test par tranche.
    for start in range(0, 40, 8):
        assert len(set(ordered[start : start + 8]) & set(test)) == 2


def test_split_labeled_keeps_the_requested_ratio_per_class():
    labels = np.repeat(np.arange(4), [10, 20, 5, 1])
    mask = split_labeled(labels, ratio=0.4, seed=0)
    assert [int(mask[labels == c].sum()) for c in range(4)] == [4, 8, 2, 1]
    assert split_labeled(labels, ratio=1.0).all()
    assert (split_labeled(labels, 0.4, seed=0) == mask).all()


def make_pools(n_classes=8, n_labeled=6, n_unlabeled=5):
    labeled, unlabeled, position = {}, {}, 0
    for c in range(n_classes):
        labeled[c] = np.arange(position, position + n_labeled)
        unlabeled[c] = np.arange(position + n_labeled, position + n_labeled + n_unlabeled)
        position += n_labeled + n_unlabeled
    return labeled, unlabeled


def test_episode_keeps_labeled_and_unlabeled_pools_apart():
    labeled, unlabeled = make_pools()
    sampler = EpisodeSampler(labeled, unlabeled, n_way=3, k_shot=1, n_query=4, n_unlabeled=3,
                             n_distractor_classes=2)
    all_labeled = set(np.concatenate(list(labeled.values())).tolist())
    all_unlabeled = set(np.concatenate(list(unlabeled.values())).tolist())
    class_of = {i: c for pools in (labeled, unlabeled) for c, pool in pools.items() for i in pool}
    rng = np.random.default_rng(0)

    for _ in range(50):
        episode = sampler.sample(rng)
        assert episode.support.shape == (3, 1) and episode.query.shape == (12,) and episode.query_class.shape == (12,)
        assert episode.unlabeled.shape == episode.unlabeled_class.shape == (15,)
        used = np.concatenate([episode.support.ravel(), episode.query.ravel(), episode.unlabeled])
        assert len(set(used.tolist())) == len(used)
        assert set(episode.support.ravel().tolist()) <= all_labeled
        assert set(episode.query.ravel().tolist()) <= all_labeled
        assert set(episode.unlabeled.tolist()) <= all_unlabeled

        classes = [class_of[i] for i in episode.support[:, 0]]
        assert len(set(classes)) == 3
        for i, position in zip(episode.query, episode.query_class):
            assert class_of[i] == classes[position]
        for i, position in zip(episode.unlabeled, episode.unlabeled_class):
            if position >= 0:
                assert class_of[i] == classes[position]
            else:
                assert class_of[i] not in classes
        assert (episode.unlabeled_class == -1).sum() == 6


def test_episode_without_unlabeled_points():
    labeled, _ = make_pools()
    empty = {c: np.empty(0, dtype=np.int64) for c in labeled}
    episode = EpisodeSampler(labeled, empty, n_way=3, k_shot=1, n_query=4).sample(np.random.default_rng(0))
    assert episode.unlabeled.size == 0


def test_shared_pool_draws_three_disjoint_groups():
    pools = {c: np.arange(10 * c, 10 * c + 10) for c in range(6)}
    sampler = EpisodeSampler(pools, pools, n_way=5, k_shot=2, n_query=4, n_unlabeled=3, shared_pool=True)
    episode = sampler.sample(np.random.default_rng(0))
    used = np.concatenate([episode.support.ravel(), episode.query.ravel(), episode.unlabeled])
    assert len(set(used.tolist())) == len(used) == 5 * 9


def test_too_few_eligible_classes_is_an_error():
    labeled, unlabeled = make_pools(n_classes=3)
    with pytest.raises(ValueError, match="classes eligibles"):
        EpisodeSampler(labeled, unlabeled, n_way=5, k_shot=1, n_query=4, n_unlabeled=3)
