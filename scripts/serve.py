#!/usr/bin/env python
"""
scripts/serve.py — Start the FastAPI server locally
Usage:
    python scripts/serve.py
    python scripts/serve.py --port 8000 --reload
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start the Churn Prediction API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"Starting API server at http://{args.host}:{args.port}")
    print(f"Docs at http://localhost:{args.port}/docs")

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
