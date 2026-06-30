import numpy as np
import soundfile as sf
import pytest
from src.services.voice_processor import VoiceProcessor
from src.config import MODEL_PATH


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model file not found. Please download it first.")
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