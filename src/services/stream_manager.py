import numpy as np
from collections import deque

class AudioStreamManager:
    def __init__(self, sample_rate: int = 16000, max_buffer_seconds: int = 3):
        self.sample_rate = sample_rate
        self.max_samples = sample_rate * max_buffer_seconds
        self.buffer = deque(maxlen=self.max_samples)

    def add_chunk(self, chunk: np.ndarray):
        if chunk.ndim != 1:
            raise ValueError("Audio chunk must be a 1D numpy array")
        self.buffer.extend(chunk)

    def get_full_buffer(self) -> np.ndarray:
        return np.array(self.buffer, dtype=np.float32)

    def clear(self):
        self.buffer.clear()