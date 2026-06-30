import numpy as np
import pytest
from src.services.stream_manager import AudioStreamManager


def test_stream_manager_buffer_overflow():
    manager = AudioStreamManager(sample_rate=10, max_buffer_seconds=1)

    chunk1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    manager.add_chunk(chunk1)
    assert len(manager.get_full_buffer()) == 5

    chunk2 = np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float32)
    manager.add_chunk(chunk2)

    full_buffer = manager.get_full_buffer()
    assert len(full_buffer) == 10
    assert full_buffer[0] == 3.0
    assert full_buffer[-1] == 12.0


def test_stream_manager_invalid_input():
    manager = AudioStreamManager()
    invalid_chunk = np.array([[1.0, 2.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        manager.add_chunk(invalid_chunk)