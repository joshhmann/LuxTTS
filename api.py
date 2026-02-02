"""
OpenAI-compatible API for LuxTTS
Fast, lightweight TTS with voice cloning - 150x realtime

Endpoints:
    - POST /v1/audio/speech - OpenAI-compatible TTS endpoint
    - GET  /v1/models - List available models
    - GET  /v1/voices - List uploaded voices
    - POST /v1/voices/upload - Upload a voice sample
    - POST /v1/voices/clone - Clone voice from URL or base64
    - GET  /health - Health check

Compatible with:
    - SillyTavern (OpenAI TTS plugin)
    - Discord bots
    - Any OpenAI TTS client

Environment Variables:
    - LUXTTS_DEVICE: cuda/cpu/mps (default: auto)
    - LUXTTS_PORT: Server port (default: 9999)
    - LUXTTS_HOST: Server host (default: 0.0.0.0)
    - LUXTTS_VOICE_CACHE: Voice cache directory (default: ./voice_cache)
    - LUXTTS_DEFAULT_VOICE: Default voice ID (optional)
"""

import io
import os
import base64
import tempfile
import time
import json
from typing import Optional, Literal
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import soundfile as sf
import numpy as np

# Import LuxTTS
try:
    from zipvoice.luxvoice import LuxTTS
except ImportError as e:
    print(f"Error importing LuxTTS: {e}")
    print("Make sure you're running from the project root and requirements are installed")
    raise

# Configuration from environment
DEVICE = os.getenv("LUXTTS_DEVICE", "auto")
PORT = int(os.getenv("LUXTTS_PORT", "9999"))
HOST = os.getenv("LUXTTS_HOST", "0.0.0.0")
VOICE_CACHE_DIR = Path(os.getenv("LUXTTS_VOICE_CACHE", "./voice_cache"))
DEFAULT_VOICE = os.getenv("LUXTTS_DEFAULT_VOICE", None)

# Create voice cache directory
VOICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Global model instance
lux_tts: Optional[LuxTTS] = None


def get_device() -> str:
    """Auto-detect best available device"""
    if DEVICE != "auto":
        return DEVICE

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model():
    """Load the LuxTTS model"""
    global lux_tts
    if lux_tts is None:
        device = get_device()
        print(f"Loading LuxTTS model on {device}...")
        lux_tts = LuxTTS("YatharthS/LuxTTS", device=device)
        print(f"LuxTTS loaded successfully on {device}")
    return lux_tts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for startup/shutdown"""
    # Startup
    print("=" * 50)
    print("Starting LuxTTS OpenAI API Server")
    print(f"Device: {get_device()}")
    print(f"Voice cache: {VOICE_CACHE_DIR.absolute()}")
    print(f"Default voice: {DEFAULT_VOICE or 'None (must upload first)'}")
    print("=" * 50)

    # Preload model
    load_model()

    yield

    # Shutdown
    print("Shutting down LuxTTS API Server")


# Create FastAPI app
app = FastAPI(
    title="LuxTTS API", description="OpenAI-compatible TTS API with voice cloning", version="1.0.0", lifespan=lifespan
)

# Add CORS middleware for SillyTavern and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class TTSSpeechRequest(BaseModel):
    """OpenAI TTS request model"""

    model: str = Field(default="luxtts", description="Model ID (always 'luxtts' for this server)")
    input: str = Field(..., description="Text to synthesize (max 4096 characters)", max_length=4096)
    voice: str = Field(default="default", description="Voice ID (must be uploaded first)")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3", description="Audio format"
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed multiplier")
    # LuxTTS-specific parameters (OpenAI extensions)
    num_steps: Optional[int] = Field(default=4, ge=1, le=50, description="Sampling steps (higher=quality, lower=speed)")
    t_shift: Optional[float] = Field(default=0.9, ge=0.1, le=1.0, description="Temperature-like parameter")
    return_smooth: Optional[bool] = Field(default=False, description="Smoother audio (may reduce metallic artifacts)")


class VoiceInfo(BaseModel):
    """Voice information"""

    voice_id: str
    name: str
    preview_url: Optional[str] = None


class VoicesListResponse(BaseModel):
    """List of voices response"""

    voices: list[VoiceInfo]


class ModelInfo(BaseModel):
    """Model information"""

    id: str
    name: str
    description: str
    max_input_length: int = 4096
    voices: list[str]


class ModelsListResponse(BaseModel):
    """List of models response"""

    models: list[ModelInfo]


class VoiceUploadResponse(BaseModel):
    """Voice upload response"""

    success: bool
    voice_id: str
    message: str
    duration_seconds: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model: str
    device: str
    gpu: Optional[str] = None
    voices_count: int
    version: str = "1.0.0"


def convert_audio_format(audio_np: np.ndarray, sample_rate: int, format: str) -> tuple[bytes, str]:
    """
    Convert audio numpy array to specified format

    Returns:
        Tuple of (audio_bytes, mime_type)
    """
    if format == "wav":
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read(), "audio/wav"

    elif format == "pcm":
        # Raw PCM 16-bit little-endian
        audio_int16 = (audio_np * 32767).astype(np.int16)
        return audio_int16.tobytes(), "audio/pcm"

    elif format in ["mp3", "opus", "aac", "flac"]:
        # For compressed formats, we need to use external tools or pydub
        try:
            from pydub import AudioSegment

            # First save as WAV in memory
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_np, sample_rate, format="WAV")
            wav_buffer.seek(0)

            # Load with pydub and convert
            audio_segment = AudioSegment.from_wav(wav_buffer)

            output_buffer = io.BytesIO()
            mime = "audio/wav"  # Default fallback

            if format == "mp3":
                audio_segment.export(output_buffer, format="mp3", bitrate="192k")
                mime = "audio/mpeg"
            elif format == "opus":
                audio_segment.export(output_buffer, format="opus")
                mime = "audio/opus"
            elif format == "aac":
                audio_segment.export(output_buffer, format="aac")
                mime = "audio/aac"
            elif format == "flac":
                audio_segment.export(output_buffer, format="flac")
                mime = "audio/flac"

            output_buffer.seek(0)
            return output_buffer.read(), mime

        except ImportError:
            # Fallback to WAV if pydub not available
            print(f"Warning: pydub not installed, falling back to WAV format")
            buffer = io.BytesIO()
            sf.write(buffer, audio_np, sample_rate, format="WAV")
            buffer.seek(0)
            return buffer.read(), "audio/wav"
        except Exception as e:
            print(f"Error converting to {format}: {e}, falling back to WAV")
            buffer = io.BytesIO()
            sf.write(buffer, audio_np, sample_rate, format="WAV")
            buffer.seek(0)
            return buffer.read(), "audio/wav"

    else:
        # Default to WAV
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read(), "audio/wav"


def get_voice_cache_path(voice_id: str) -> Path:
    """Get the cache path for a voice ID"""
    return VOICE_CACHE_DIR / f"{voice_id}.pt"


def load_voice(voice_id: str) -> Optional[dict]:
    """Load a cached voice encoding"""
    cache_path = get_voice_cache_path(voice_id)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")
    return None


def save_voice(voice_id: str, encoded_voice: dict):
    """Save a voice encoding to cache"""
    cache_path = get_voice_cache_path(voice_id)
    torch.save(encoded_voice, cache_path)


def list_cached_voices() -> list[str]:
    """List all cached voice IDs"""
    voices = []
    if VOICE_CACHE_DIR.exists():
        for file in VOICE_CACHE_DIR.glob("*.pt"):
            voices.append(file.stem)
    return voices


# API Endpoints


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "LuxTTS API",
        "version": "1.0.0",
        "description": "OpenAI-compatible TTS with voice cloning",
        "endpoints": {"tts": "/v1/audio/speech", "models": "/v1/models", "voices": "/v1/voices", "health": "/health"},
        "openai_compatible": True,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    device = get_device()
    gpu_name = None
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    voices = list_cached_voices()

    return HealthResponse(status="healthy", model="LuxTTS", device=device, gpu=gpu_name, voices_count=len(voices))


@app.get("/v1/models", response_model=ModelsListResponse)
async def list_models():
    """List available models - OpenAI compatible"""
    voices = list_cached_voices()

    # If no voices uploaded yet, show empty list but model is available
    available_voices = voices if voices else ["upload a voice first"]

    models = [
        ModelInfo(
            id="luxtts",
            name="LuxTTS",
            description="High-quality voice cloning TTS with 48kHz output",
            max_input_length=4096,
            voices=available_voices,
        )
    ]

    return ModelsListResponse(models=models)


@app.api_route("/v1/models/{model_id}", methods=["GET"])
async def get_model_info(model_id: str):
    """Get specific model info"""
    if model_id != "luxtts":
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    voices = list_cached_voices()
    available_voices = voices if voices else ["upload a voice first"]

    return ModelInfo(
        id="luxtts",
        name="LuxTTS",
        description="High-quality voice cloning TTS with 48kHz output",
        max_input_length=4096,
        voices=available_voices,
    )


@app.get("/v1/voices", response_model=VoicesListResponse)
async def list_voices():
    """List uploaded voices"""
    voices = list_cached_voices()
    voice_list = [VoiceInfo(voice_id=v, name=v) for v in voices]
    return VoicesListResponse(voices=voice_list)


@app.post("/v1/audio/speech")
async def create_speech(request: Request):
    """
    OpenAI-compatible TTS endpoint

    Supports both JSON and Form data (for maximum compatibility)
    """
    try:
        # Parse request
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            data = await request.json()
            tts_request = TTSSpeechRequest(**data)
        else:
            # Form data fallback
            form = await request.form()
            tts_request = TTSSpeechRequest(
                model=form.get("model", "luxtts"),
                input=form.get("input") or form.get("text", ""),
                voice=form.get("voice", "default"),
                response_format=form.get("response_format", "mp3"),
                speed=float(form.get("speed", 1.0)),
            )

        # Validate input
        if not tts_request.input:
            raise HTTPException(status_code=400, detail="Input text is required")

        if len(tts_request.input) > 4096:
            raise HTTPException(status_code=400, detail="Input text exceeds maximum length of 4096 characters")

        # Load voice
        voice_id = tts_request.voice

        # Try to load cached voice
        encoded_voice = load_voice(voice_id)

        # If voice not found and we have a default, use that
        if encoded_voice is None and DEFAULT_VOICE:
            encoded_voice = load_voice(DEFAULT_VOICE)
            if encoded_voice:
                voice_id = DEFAULT_VOICE

        if encoded_voice is None:
            available_voices = list_cached_voices()
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{voice_id}' not found. Available voices: {available_voices or 'None (upload a voice first)'}. Use /v1/voices/upload to add voices.",
            )

        # Generate speech
        model = load_model()

        start_time = time.time()

        final_wav = model.generate_speech(
            tts_request.input,
            encoded_voice,
            num_steps=tts_request.num_steps or 4,
            speed=tts_request.speed,
            t_shift=tts_request.t_shift or 0.9,
            return_smooth=tts_request.return_smooth or False,
        )

        # Convert to numpy
        final_wav = final_wav.numpy().squeeze()
        generation_time = time.time() - start_time

        # Calculate audio duration and RTF
        sample_rate = 48000
        audio_duration = len(final_wav) / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else 0

        print(f"Generated {audio_duration:.2f}s audio in {generation_time:.2f}s (RTF: {rtf:.2f}x)")

        # Convert to requested format
        audio_bytes, mime_type = convert_audio_format(final_wav, sample_rate, tts_request.response_format)

        # Build filename
        ext = tts_request.response_format
        filename = f"speech.{ext}"

        return Response(
            content=audio_bytes,
            media_type=mime_type,
            headers={
                "X-Generation-Time": str(generation_time),
                "X-RTF": str(rtf),
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Voice-ID": voice_id,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.post("/v1/voices/upload", response_model=VoiceUploadResponse)
async def upload_voice(
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)"),
    voice_id: str = Form(..., description="Unique voice identifier"),
    duration: int = Form(5, description="Duration in seconds to use from the audio"),
    rms: float = Form(0.01, description="RMS volume normalization (0.001-0.1 recommended)"),
    normalize: bool = Form(True, description="Apply volume normalization"),
):
    """Upload a voice sample for cloning"""
    try:
        model = load_model()

        # Validate file extension
        allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
        file_ext = Path(audio_file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file format: {file_ext}. Allowed: {allowed_extensions}"
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Encode the voice
            print(f"Encoding voice '{voice_id}' from {audio_file.filename}...")
            start = time.time()

            encoded_prompt = model.encode_prompt(tmp_path, duration=duration, rms=rms if normalize else 0.001)

            encoding_time = time.time() - start
            print(f"Voice encoded in {encoding_time:.2f}s")

            # Cache it
            save_voice(voice_id, encoded_prompt)

            return VoiceUploadResponse(
                success=True,
                voice_id=voice_id,
                message=f"Voice '{voice_id}' uploaded and cached successfully",
                duration_seconds=duration,
            )

        finally:
            os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading voice: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Voice upload failed: {str(e)}")


@app.post("/v1/voices/clone")
async def clone_voice_from_data(
    voice_id: str = Form(...),
    audio_base64: Optional[str] = Form(None),
    audio_url: Optional[str] = Form(None),
    duration: int = Form(5),
    rms: float = Form(0.01),
):
    """Clone voice from base64 data or URL"""
    try:
        if not audio_base64 and not audio_url:
            raise HTTPException(status_code=400, detail="Either audio_base64 or audio_url is required")

        model = load_model()
        temp_path = None

        try:
            if audio_base64:
                # Decode base64 audio
                audio_data = base64.b64decode(audio_base64)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_data)
                    temp_path = tmp.name
            elif audio_url:
                # Download from URL
                import urllib.request

                temp_path = tempfile.mktemp(suffix=".wav")
                urllib.request.urlretrieve(audio_url, temp_path)
            else:
                raise HTTPException(status_code=400, detail="Either audio_base64 or audio_url must be provided")

            # Encode the voice
            print(f"Cloning voice '{voice_id}'...")
            start = time.time()
            encoded_prompt = model.encode_prompt(temp_path, duration=duration, rms=rms)
            print(f"Voice cloned in {time.time() - start:.2f}s")

            # Cache it
            save_voice(voice_id, encoded_prompt)

            return VoiceUploadResponse(
                success=True,
                voice_id=voice_id,
                message=f"Voice '{voice_id}' cloned successfully",
                duration_seconds=duration,
            )

        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error cloning voice: {e}")
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a cached voice"""
    cache_path = get_voice_cache_path(voice_id)

    if not cache_path.exists():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")

    cache_path.unlink()
    return {"success": True, "message": f"Voice '{voice_id}' deleted"}


# Legacy/Compatibility endpoints


@app.post("/api/openai/custom/generate-voice")
async def legacy_openai_compat(request: Request):
    """Legacy endpoint for backward compatibility"""
    return await create_speech(request)


@app.post("/tts")
async def simple_tts(text: str = Form(...), ref_audio: UploadFile = File(...), speed: float = Form(1.0)):
    """Simple TTS endpoint - one-shot with reference audio"""
    try:
        model = load_model()

        # Save uploaded reference audio
        file_ext = Path(ref_audio.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await ref_audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Encode and generate in one go
            start = time.time()
            encoded_prompt = model.encode_prompt(tmp_path, duration=5, rms=0.01)
            final_wav = model.generate_speech(
                text, encoded_prompt, num_steps=4, speed=speed, t_shift=0.9, return_smooth=False
            )

            final_wav = final_wav.numpy().squeeze()
            total_time = time.time() - start
            audio_duration = len(final_wav) / 48000

            print(
                f"TTS completed in {total_time:.2f}s (audio: {audio_duration:.2f}s, RTF: {total_time / audio_duration:.2f})"
            )

            # Return audio as MP3
            audio_bytes, mime_type = convert_audio_format(final_wav, 48000, "mp3")

            return Response(
                content=audio_bytes,
                media_type=mime_type,
                headers={"Content-Disposition": "attachment; filename=speech.mp3"},
            )

        finally:
            os.unlink(tmp_path)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print(f"Starting LuxTTS OpenAI API Server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
