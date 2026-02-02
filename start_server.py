#!/usr/bin/env python3
"""
LuxTTS API Server Launcher

This script starts the OpenAI-compatible TTS API server for LuxTTS.

Environment Variables:
    LUXTTS_DEVICE: Device to use (cuda/cpu/mps/auto). Default: auto
    LUXTTS_PORT: Port to run the server on. Default: 9999
    LUXTTS_HOST: Host to bind the server to. Default: 0.0.0.0
    LUXTTS_VOICE_CACHE: Directory to store voice cache. Default: ./voice_cache
    LUXTTS_DEFAULT_VOICE: Default voice ID to use. Optional.

Usage:
    python start_server.py
    python start_server.py --device cuda --port 9999
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Start the LuxTTS OpenAI-compatible API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with default settings (auto-detect device, port 9999)
    python start_server.py

    # Start on specific device
    python start_server.py --device cuda

    # Start on custom port
    python start_server.py --port 8080

    # Start with default voice
    python start_server.py --default-voice my_voice

For more information, see OPENAI_API_SETUP.md
        """,
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "mps", "auto"],
        default="auto",
        help="Device to run the model on (default: auto)",
    )

    parser.add_argument("--port", type=int, default=9999, help="Port to run the server on (default: 9999)")

    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to (default: 0.0.0.0)")

    parser.add_argument(
        "--voice-cache", default="./voice_cache", help="Directory to store voice cache (default: ./voice_cache)"
    )

    parser.add_argument("--default-voice", help="Default voice ID to use (optional)")

    parser.add_argument("--preload", action="store_true", default=True, help="Preload model on startup (default: True)")

    args = parser.parse_args()

    # Set environment variables
    os.environ["LUXTTS_DEVICE"] = args.device
    os.environ["LUXTTS_PORT"] = str(args.port)
    os.environ["LUXTTS_HOST"] = args.host
    os.environ["LUXTTS_VOICE_CACHE"] = args.voice_cache

    if args.default_voice:
        os.environ["LUXTTS_DEFAULT_VOICE"] = args.default_voice

    print("=" * 60)
    print("LuxTTS OpenAI API Server")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Voice Cache: {args.voice_cache}")
    if args.default_voice:
        print(f"Default Voice: {args.default_voice}")
    print("=" * 60)
    print()
    print("API Endpoints:")
    print(f"  - http://{args.host}:{args.port}/v1/audio/speech  (TTS)")
    print(f"  - http://{args.host}:{args.port}/v1/models        (List models)")
    print(f"  - http://{args.host}:{args.port}/v1/voices        (List voices)")
    print(f"  - http://{args.host}:{args.port}/health           (Health check)")
    print()
    print("Documentation: http://localhost:{}/docs".format(args.port))
    print("=" * 60)
    print()

    # Import and start the API
    try:
        from api import app
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
