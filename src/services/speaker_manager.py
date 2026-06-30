# src/services/speaker_manager.py
import os
import numpy as np
from src.config import EMBEDDINGS_DIR
from src.services.voice_processor import VoiceProcessor


class SpeakerManager:
    def __init__(self, processor: VoiceProcessor = None):
        self.processor = processor or VoiceProcessor()

    def register_speaker(self, name: str, audio_path: str) -> bool:
        try:
            new_embedding = self.processor.extract_embedding(audio_path)
            save_path = EMBEDDINGS_DIR / f"{name}.npy"

            if save_path.exists():
                # Update existing embedding by averaging
                old_embedding = np.load(save_path)
                combined_embedding = (old_embedding + new_embedding) / 2
                np.save(save_path, combined_embedding)
                print(f"✅ Updated registration for '{name}' with new sample.")
            else:
                np.save(save_path, new_embedding)
                print(f"✅ Registered new user '{name}'.")

            return True
        except Exception as e:
            print(f"❌ Error registering speaker: {e}")
            return False

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def identify_speaker(self, audio_path: str, threshold: float = 0.6) -> tuple[str, float]:
        try:
            target_embedding = self.processor.extract_embedding(audio_path)

            best_match = "Unknown"
            highest_score = 0.0

            for file_name in os.listdir(EMBEDDINGS_DIR):
                if file_name.endswith(".npy"):
                    speaker_name = file_name.replace(".npy", "")
                    saved_embedding = np.load(EMBEDDINGS_DIR / file_name)

                    score = self._cosine_similarity(target_embedding, saved_embedding)

                    if score > highest_score:
                        highest_score = score
                        best_match = speaker_name

            if highest_score >= threshold:
                return best_match, highest_score
            else:
                return "Unknown", highest_score

        except Exception as e:
            print(f"❌ Error identifying speaker: {e}")
            return "Error", 0.0