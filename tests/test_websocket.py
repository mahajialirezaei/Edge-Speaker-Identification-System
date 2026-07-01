from fastapi.testclient import TestClient
from src.api import app
import numpy as np

client = TestClient(app)


def test_websocket_streaming():
    with client.websocket_connect("/ws/stream/") as websocket:
        fake_audio = np.zeros(1000, dtype=np.int16).tobytes()

        websocket.send_bytes(fake_audio)

        response = websocket.receive_json()

        assert response["status"] == "receiving"
        assert response["buffer_size"] == 1000