#!/usr/bin/env python3
"""
LuxTTS Voice Upload Utility

Simple command-line tool to upload voice samples to the LuxTTS API.

Usage:
    python upload_voice.py my_voice.wav --id my_voice
    python upload_voice.py voice.mp3 --id narrator --duration 10
"""

import argparse
import requests
import sys
from pathlib import Path


def upload_voice(api_url: str, audio_path: str, voice_id: str, duration: int = 5, rms: float = 0.01):
    """Upload a voice sample to the API"""

    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"Error: File '{audio_path}' not found")
        return False

    print(f"Uploading '{audio_path}' as voice '{voice_id}'...")
    print(f"  Duration: {duration}s")
    print(f"  RMS: {rms}")

    url = f"{api_url}/v1/voices/upload"

    try:
        with open(audio_path, "rb") as f:
            files = {"audio_file": (audio_file.name, f, "audio/wav")}
            data = {"voice_id": voice_id, "duration": duration, "rms": rms}

            response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            print(f"✓ Success! Voice '{result['voice_id']}' uploaded")
            print(f"  Message: {result['message']}")
            return True
        else:
            print(f"✗ Error: HTTP {response.status_code}")
            try:
                error = response.json()
                print(f"  Detail: {error.get('detail', 'Unknown error')}")
            except:
                print(f"  Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Cannot connect to API at {api_url}")
        print("  Is the server running? Start it with: python start_server.py")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def list_voices(api_url: str):
    """List all uploaded voices"""
    url = f"{api_url}/v1/voices"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            voices = data.get("voices", [])

            if voices:
                print(f"Uploaded voices ({len(voices)}):")
                for voice in voices:
                    print(f"  - {voice['voice_id']}")
            else:
                print("No voices uploaded yet")
            return True
        else:
            print(f"Error: HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to API at {api_url}")
        return False


def test_tts(
    api_url: str, voice_id: str, text: str = "Hello, this is a test of LuxTTS!", output: str = "test_output.mp3"
):
    """Test TTS generation"""
    url = f"{api_url}/v1/audio/speech"

    payload = {"model": "luxtts", "input": text, "voice": voice_id, "response_format": "mp3", "speed": 1.0}

    print(f"Testing TTS with voice '{voice_id}'...")
    print(f"  Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")

    try:
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            with open(output, "wb") as f:
                f.write(response.content)

            # Get generation stats from headers
            gen_time = response.headers.get("X-Generation-Time", "N/A")
            rtf = response.headers.get("X-RTF", "N/A")

            print(f"✓ Success! Audio saved to '{output}'")
            print(f"  Generation time: {gen_time}s")
            print(f"  Real-time factor: {rtf}x")
            return True
        else:
            print(f"✗ Error: HTTP {response.status_code}")
            try:
                error = response.json()
                print(f"  Detail: {error.get('detail', 'Unknown error')}")
            except:
                print(f"  Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to API at {api_url}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload voice samples to LuxTTS API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Upload a voice
    python upload_voice.py my_voice.wav --id my_voice
    
    # Upload with custom duration
    python upload_voice.py voice.mp3 --id narrator --duration 10
    
    # List all voices
    python upload_voice.py --list
    
    # Test TTS with a voice
    python upload_voice.py --test my_voice --text "Hello world!"
        """,
    )

    parser.add_argument("audio_file", nargs="?", help="Path to audio file (WAV, MP3, etc.)")

    parser.add_argument("--id", help="Voice ID (name for the voice)")

    parser.add_argument(
        "--api-url", default="http://localhost:9999", help="API base URL (default: http://localhost:9999)"
    )

    parser.add_argument("--duration", type=int, default=5, help="Duration in seconds to use from audio (default: 5)")

    parser.add_argument("--rms", type=float, default=0.01, help="RMS volume normalization (default: 0.01)")

    parser.add_argument("--list", action="store_true", help="List all uploaded voices")

    parser.add_argument("--test", metavar="VOICE_ID", help="Test TTS generation with specified voice")

    parser.add_argument("--text", default="Hello, this is a test of LuxTTS!", help="Text for TTS test (with --test)")

    parser.add_argument(
        "--output", default="test_output.mp3", help="Output file for TTS test (default: test_output.mp3)"
    )

    args = parser.parse_args()

    # List voices
    if args.list:
        sys.exit(0 if list_voices(args.api_url) else 1)

    # Test TTS
    if args.test:
        sys.exit(0 if test_tts(args.api_url, args.test, args.text, args.output) else 1)

    # Upload voice
    if not args.audio_file:
        parser.print_help()
        sys.exit(1)

    if not args.id:
        # Auto-generate ID from filename
        args.id = Path(args.audio_file).stem
        print(f"Auto-generated voice ID: {args.id}")

    success = upload_voice(args.api_url, args.audio_file, args.id, args.duration, args.rms)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
