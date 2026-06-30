from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

MODEL_PATH = BASE_DIR / "models" / "wespeaker_en_voxceleb_resnet34.onnx"
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
REGISTERED_DIR = DATA_DIR / "registered"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
REGISTERED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SAMPLE_RATE = 16000