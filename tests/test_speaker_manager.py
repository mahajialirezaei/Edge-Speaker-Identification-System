import numpy as np
from unittest.mock import MagicMock
from src.services.speaker_manager import SpeakerManager


def test_cosine_similarity():
    mock_processor = MagicMock()
    manager = SpeakerManager(processor=mock_processor)

    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    vec3 = np.array([0.0, 1.0, 0.0])

    assert np.isclose(manager._cosine_similarity(vec1, vec2), 1.0)

    assert np.isclose(manager._cosine_similarity(vec1, vec3), 0.0)