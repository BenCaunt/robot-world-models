from __future__ import annotations

import numpy as np

from robot_world_models.adapters.base import CanonicalEpisode
from robot_world_models.training import split_episode_ids, transitions_from_episodes


def _episode(identifier: str, offset: float) -> CanonicalEpisode:
    states = np.array(
        [[offset, 0.0], [offset + 1.0, 1.0], [offset + 2.0, 2.0]],
        dtype=np.float32,
    )
    return CanonicalEpisode(
        episode_id=identifier,
        robot_wref="Robot/test",
        dataset_wref="Dataset/test",
        timestamps_seconds=[0.0, 0.1, 0.2],
        observations={"state": states},
        actions=np.zeros((3, 2), dtype=np.float32),
        task=None,
        modality_mask={},
    )


def test_transition_extraction_never_crosses_episode_boundaries() -> None:
    transitions = transitions_from_episodes([_episode("a", 0.0), _episode("b", 100.0)])

    assert len(transitions.states) == 4
    assert transitions.episode_ids.tolist() == ["a", "a", "b", "b"]
    assert transitions.targets[:, 0].tolist() == [1.0, 2.0, 101.0, 102.0]


def test_episode_split_is_seeded_and_disjoint() -> None:
    first = split_episode_ids(
        [str(index) for index in range(10)],
        seed=7,
        train_fraction=0.8,
        validation_fraction=0.1,
    )
    second = split_episode_ids(
        [str(index) for index in range(10)],
        seed=7,
        train_fraction=0.8,
        validation_fraction=0.1,
    )

    assert first == second
    assert {key: len(value) for key, value in first.items()} == {
        "train": 8,
        "validation": 1,
        "test": 1,
    }
    assert len(set(first["train"] + first["validation"] + first["test"])) == 10
