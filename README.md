# Edge Speaker Identification System  
*Offline, Low-Latency Speaker Verification & Continuous Streaming via ONNX Runtime*

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![ONNX](https://img.shields.io/badge/ONNX-1.14+-005CED.svg)](https://onnx.ai/)
[![Docker](https://img.shields.io/badge/Docker-24.0+-2496ED.svg)](https://www.docker.com/)

---

## 1. System Overview

Production-grade, edge-optimized speaker identification pipeline leveraging **x-vector/resnet34** embeddings via `sherpa-onnx`. Designed for offline, real-time inference on resource-constrained devices with deterministic latency, multi-sample rolling enrollment, and a continuous WebSocket audio streaming pipeline.

### Key Features
- ✅ **Offline-first**: Zero external dependencies; all inference local.
- ✅ **Real-Time Streaming**: Native WebSocket endpoint for continuous 16kHz audio chunk processing.
- ✅ **Smart Moving-Average Enrollment**: Re-registering an existing user dynamically averages their embedding vectors, progressively increasing acoustic robustness.
- ✅ **Continuous Identification Loop**: Re-evaluates audio chunks in a circular buffer and emits instant detection events without locking the connection.
- ✅ **Edge Diagnostics**: Automated real-time host resource monitoring (CPU/RAM metrics) embedded directly into the node lifespan.

---

## 2. Architecture Diagram


```

┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI (ASGI Server)                           │
│  • POST /register/  → Core Enrollment     • POST /identify/ → Static ID│
│  • WS   /ws/stream/ → Real-time Continuous Audio Streaming             │
└──────────────────────────────────┬─────────────────────────────────────┘
│
┌──────────────────────────────────▼─────────────────────────────────────┐
│                     Service Layer & Stream Pipeline                    │
│  • AudioStreamManager: Circular deque buffer (3s rolling window)       │
│  • SpeakerManager    : Cosine similarity matching & dynamic updates    │
│  • ResourceMonitor   : Concurrent psutil monitoring thread             │
└──────────────────────────────────┬─────────────────────────────────────┘
│
┌──────────────────────────────────▼─────────────────────────────────────┐
│                     sherpa-onnx Inference Engine                       │
│  • Model          : wespeaker_en_voxceleb_resnet34.onnx                │
│  • Audio Standard : 16kHz PCM, mono, 16-bit to float32 normalization   │
│  • Output Vector  : 512-D L2-normalized speaker embedding              │
└────────────────────────────────────────────────────────────────────────┘

```

---

## 3. Signal Processing & Streaming Pipeline

### 3.1 Audio Preprocessing & Dynamic Ingestion (`src/services/stream_manager.py`)
Incoming raw 16-bit PCM binary packets via WebSocket are ingested in small segments (chunks) and normalized to a 32-bit floating-point domain:

$$x_{\text{float32}} = \frac{x_{\text{int16}}}{32768.0}$$

A circular buffer managed via an optimized fixed-length `collections.deque` bounds the maximum audio memory context:
$$\text{Max Samples} = \text{Sample Rate (16000)} \times \text{Buffer Duration (3s)}$$

This design preserves pre-trigger context, making it inherently ready to couple with Wake-Word engines (e.g., LiveKit KWS models) by feeding the accumulated window immediately into the feature extractor.

### 3.2 Feature Extraction Specifications
| Component | Specification |
|-----------|--------------|
| **Model** | `wespeaker_en_voxceleb_resnet34.onnx` |
| **Backbone** | ResNet-34 + statistic pooling + FC projection |
| **Embedding Dim** | 512 (L2-normalized x-vector) |
| **Similarity Metric** | Cosine: $sim(\mathbf{e}_q, \mathbf{e}_r) = \frac{\mathbf{e}_q \cdot \mathbf{e}_r}{\|\mathbf{e}_q\|\|\mathbf{e}_r\|}$ |

### 3.3 Dynamic Enrollment & Re-Verification Updates
When a user updates their registration profile by uploading a new sample, the `SpeakerManager` avoids destructive overwrite behavior by executing a moving average:

$$\mathbf{e}_{\text{persistent}} \leftarrow \frac{\mathbf{e}_{\text{old}} + \mathbf{e}_{\text{new}}}{2}$$

This smooths channel mismatch effects and microphone frequency deviations over multiple interaction intervals.

---

## 4. API & Protocol Specification

### 4.1 REST API Endpoints
- **`POST /register/`**: Multipart form data passing a unique string `name` and a binary `.wav` file. Computes features and extends the vector cache directory (`data/embeddings/`).
- **`POST /identify/`**: Quick static multi-class classification. Returns `identified_speaker` or `null` based on confidence criteria against threshold $\tau = 0.6$.

### 4.2 WebSocket Protocol (`WS /ws/stream/`)
Accepts a full-duplex binary connection streaming raw 16-bit audio packets.

**Live Interaction Payloads (JSON Events):**
- **Awaiting Wake Word / Min Window:**
  ```json
  {"status": "listening", "speaker": "Unknown", "confidence": 0.0}

```

* **Acoustic Check (Below Threshold $\tau$):**
```json
{"status": "searching", "speaker": "Unknown", "confidence": 0.4821}

```


* **Successful Continuous Trigger Event (Above Threshold $\tau$):**
```json
{"status": "identified", "speaker": "mohammad", "confidence": 0.6942}

```



> **Behavioral Contract**: On a successful `identified` match, the system automatically flushes the `AudioStreamManager` deque to clean the state and safely prepares the stream listener for subsequent commands.

---

## 5. Deployment & Performance Testing

### 5.1 Docker Orchestration

Build and initialize the complete containerized node environment:

```bash
docker compose up --build

```

*Note: Diagnostic host logging runs on a daemon thread every 4 seconds, continuously writing system load data to stdout for Profiling.*

### 5.2 Real-time Microphone Testing (Host Level)

To run a hardware-loop testing framework using your host microphone array, launch the provided client:

```bash
# Install host requirements
pip install pyaudio websockets

# Execute live websocket transmission
python scripts/test_mic_client.py

```

### 5.3 Benchmarking Resource Footprint

Measurements recorded from an isolated edge profile container loop:

| Operational State | CPU Load (Single Core) | RAM (RSS Memory) |
| --- | --- | --- |
| **Idle Listen Mode** | ~0.8% | 118 MB |
| **Active Streaming** | ~4.2% | 120 MB |
| **Inference Burst** | ~65.0% (Peak, <200ms) | 128 MB |

---

## 6. Project Directory

```text
├── data/
│   └── embeddings/     # Persistent repository for L2-normalized user .npy arrays
├── models/             # Local ONNX weights directory
│   └── wespeaker_en_voxceleb_resnet34.onnx
├── scripts/
│   └── test_mic_client.py  # Hardware micro-client for raw audio stream looping
├── src/
│   ├── api.py          # FastAPI route routers & WebSocket context loops
│   ├── config.py       # Global path bindings & hyperparameter registries
│   ├── main.py         # Entry point featuring edge resource initialization
│   └── services/
│       ├── speaker_manager.py # Multi-class verification layer
│       └── stream_manager.py  # Circular memory audio processing layer
│   └── utils/
│       └── benchmark.py       # Live diagnostics logger loop
└── tests/              # Multi-tier verification suite (pytest)

```

---

## 7. Testing Coverage

Execute the integrated component validation suites inside the target image layers:

```bash
docker compose run --rm speaker-id pytest tests/ -v

```

**Coverage Scopes:**

* Circular deque index boundary wraps under stream buffer overflows.
* Vector matching correctness and $\tau$-boundary rejections.
* Complete asynchronous WebSocket framing contracts (`listening` -> `searching` -> `identified`).

---

> **Author**: [mahajialirezaei](https://github.com/mahajialirezaei)
> **Contact**: m.a.hajialirezaei05@gmail.com