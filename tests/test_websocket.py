from fastapi.testclient import TestClient
from src.api import app
import numpy as np

client = TestClient(app)


def test_websocket_streaming():
    with client.websocket_connect("/ws/stream/") as websocket:
        fake_audio_small = np.zeros(1000, dtype=np.int16).tobytes()
        websocket.send_bytes(fake_audio_small)

        response = websocket.receive_json()

        assert response["status"] == "listening"
        assert "speaker" in response
        assert "confidence" in response


def test_websocket_continuous_identification_flow():
    with client.websocket_connect("/ws/stream/") as websocket:
        fake_audio_large = np.zeros(16000, dtype=np.int16).tobytes()
        websocket.send_bytes(fake_audio_large)

        response = websocket.receive_json()

        assert response["status"] in ["searching", "identified"]