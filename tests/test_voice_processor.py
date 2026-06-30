import os
import numpy as np
import soundfile as sf
import pytest
from src.services.voice_processor import VoiceProcessor
from src.config import MODEL_PATH


def is_valid_model():
    return MODEL_PATH.exists() and os.path.getsize(MODEL_PATH) > 1024 * 1024


@pytest.mark.skipif(not is_valid_model(),
                    reason="Real model file not found. Skipping to avoid ONNX graph errors in CI.")
def test_extract_embedding(tmp_path):
    processor = VoiceProcessor()

    dummy_wav = tmp_path / "dummy.wav"
    sample_rate = 16000
    dummy_data = np.random.uniform(-1, 1, sample_rate).astype(np.float32)
    sf.write(str(dummy_wav), dummy_data, sample_rate)

    embedding = processor.extract_embedding(str(dummy_wav))

    assert isinstance(embedding, np.ndarray), "Embedding must be a numpy array"
    assert embedding.dtype == np.float32, "Embedding dtype must be float32"
    assert len(embedding.shape) == 1, "Embedding must be a 1D array"