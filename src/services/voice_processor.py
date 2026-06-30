import numpy as np
import sherpa_onnx
import soundfile as sf
from src.config import MODEL_PATH


class VoiceProcessor:
    def __init__(self, model_path: str = str(MODEL_PATH), num_threads: int = 1):
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model_path,
            num_threads=num_threads,
            debug=False
        )

        if not config.validate():
            raise ValueError(f"❌ model setting is invalid check path: {model_path}")

        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def _read_audio(self, file_path: str) -> tuple[np.ndarray, int]:
        data, sample_rate = sf.read(file_path, always_2d=True, dtype="float32")
        data = data[:, 0]
        samples = np.ascontiguousarray(data)
        return samples, sample_rate

    def extract_embedding(self, file_path: str) -> np.ndarray:
        samples, sample_rate = self._read_audio(file_path)

        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)

        embedding = self.extractor.compute(stream)
        return np.array(embedding, dtype=np.float32)