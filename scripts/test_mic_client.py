import asyncio
import websockets
import pyaudio
import json
import sys

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4000


async def stream_mic():
    uri = "ws://localhost:8000/ws/stream/"

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print(f"🎤 Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected! Start speaking...")

            async def sender():
                while True:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    await websocket.send(data)
                    await asyncio.sleep(0.01)

            async def receiver():
                while True:
                    response = await websocket.recv()
                    data = json.loads(response)
                    status = data.get("status")
                    speaker = data.get("speaker")
                    conf = data.get("confidence")

                    if status == "identified":
                        print(f"\n🎯 [DETECTED] Speaker: {speaker} | Confidence: {conf:.2f}")
                    elif status == "searching":
                        print("⏳ [Searching...] Analyzing voice...", end="\r")
                    elif status == "listening":
                        print("👂 [Listening...] Waiting for more audio...", end="\r")

            await asyncio.gather(sender(), receiver())

    except websockets.exceptions.ConnectionClosed:
        print("\n❌ Connection closed by server.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(stream_mic())
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
        sys.exit(0)