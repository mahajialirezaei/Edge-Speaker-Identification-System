# src/main.py
import uvicorn
import argparse

def main():
    parser = argparse.ArgumentParser(description="Speaker Identification Edge Node")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Port Number")
    args = parser.parse_args()

    print(f"🚀 Starting Edge API on {args.host}:{args.port}...")
    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()