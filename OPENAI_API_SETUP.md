# LuxTTS OpenAI API Setup Guide

This guide will help you set up LuxTTS as an OpenAI-compatible TTS API server that works with SillyTavern, Discord bots, and any other OpenAI TTS client.

## Features

- **OpenAI API Compatible**: Drop-in replacement for OpenAI TTS
- **Voice Cloning**: Upload voice samples for custom voices
- **Multiple Formats**: MP3, Opus, AAC, FLAC, WAV, PCM
- **Fast**: 150x realtime generation
- **Flexible**: Works with SillyTavern, Discord bots, and custom clients

## Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/ysharma3501/LuxTTS.git
cd LuxTTS
pip install -r requirements.txt
```

### 2. Start the Server

```bash
# Start with default settings (auto-detects GPU)
python start_server.py

# Start with specific device
python start_server.py --device cuda

# Start on custom port
python start_server.py --port 8080

# Start with a default voice
python start_server.py --default-voice my_voice
```

### 3. Upload a Voice

Before generating speech, you need to upload a voice sample:

```bash
# Using curl
curl -X POST "http://localhost:9999/v1/voices/upload" \
  -F "audio_file=@your_voice_sample.wav" \
  -F "voice_id=my_voice" \
  -F "duration=5"
```

Or use the Python client (see examples below).

### 4. Test the API

```bash
# Generate speech
curl -X POST "http://localhost:9999/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "luxtts",
    "input": "Hello, this is a test of LuxTTS!",
    "voice": "my_voice",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output test.mp3
```

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/speech` | POST | Generate speech (OpenAI compatible) |
| `/v1/models` | GET | List available models |
| `/v1/voices` | GET | List uploaded voices |
| `/v1/voices/upload` | POST | Upload a voice sample |
| `/health` | GET | Health check |

### Request Format

**POST /v1/audio/speech**

```json
{
  "model": "luxtts",
  "input": "Text to speak (max 4096 chars)",
  "voice": "voice_id",
  "response_format": "mp3",
  "speed": 1.0,
  "num_steps": 4,
  "t_shift": 0.9,
  "return_smooth": false
}
```

**Parameters:**
- `model`: Always "luxtts"
- `input`: Text to synthesize (required, max 4096 chars)
- `voice`: Voice ID (must be uploaded first, or use default)
- `response_format`: Audio format - mp3, opus, aac, flac, wav, pcm (default: mp3)
- `speed`: Speech speed multiplier 0.25-4.0 (default: 1.0)
- `num_steps`: Sampling steps 1-50 (default: 4, higher = better quality)
- `t_shift`: Temperature-like parameter 0.1-1.0 (default: 0.9)
- `return_smooth`: Smoother audio if true (may reduce artifacts)

## SillyTavern Setup

### Method 1: Custom OpenAI Endpoint

1. In SillyTavern, go to **Extensions** → **TTS** → **OpenAI**
2. Set the **Base URL** to: `http://localhost:9999/v1`
3. Set the **API Key** to any non-empty string (e.g., "dummy")
4. Set the **Voice** to your uploaded voice ID (e.g., `my_voice`)
5. Select **Model**: `luxtts`

### Method 2: AllTalk TTS Extension

If SillyTavern has AllTalk TTS extension support:

1. Install the AllTalk TTS extension
2. Set the API URL to: `http://localhost:9999`
3. Select voice from the dropdown

## Discord Bot Setup

### Using discord.py

```python
import discord
from discord.ext import commands
import aiohttp
import io

# Configuration
TTS_API_URL = "http://localhost:9999/v1/audio/speech"
VOICE_ID = "my_voice"  # Your uploaded voice

class TTSBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
    
    async def generate_tts(self, text: str) -> bytes:
        """Generate TTS audio from text"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": "luxtts",
                "input": text,
                "voice": VOICE_ID,
                "response_format": "mp3",
                "speed": 1.0
            }
            
            async with session.post(TTS_API_URL, json=payload) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    error = await response.text()
                    raise Exception(f"TTS Error: {error}")

bot = TTSBot()

@bot.command()
async def speak(ctx, *, text: str):
    """Convert text to speech"""
    if len(text) > 4096:
        await ctx.send("Text too long! Maximum 4096 characters.")
        return
    
    async with ctx.typing():
        try:
            audio_data = await bot.generate_tts(text)
            
            # Send as file
            audio_file = discord.File(io.BytesIO(audio_data), filename="speech.mp3")
            await ctx.send(file=audio_file)
            
        except Exception as e:
            await ctx.send(f"Error generating speech: {e}")

@bot.command()
async def voices(ctx):
    """List available voices"""
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:9999/v1/voices") as response:
            if response.status == 200:
                data = await response.json()
                voice_list = [v["voice_id"] for v in data["voices"]]
                await ctx.send(f"Available voices: {', '.join(voice_list)}")
            else:
                await ctx.send("Error fetching voices")

# Run the bot
bot.run("YOUR_DISCORD_BOT_TOKEN")
```

### Using discord.js

```javascript
const { Client, GatewayIntentBits } = require('discord.js');
const axios = require('axios');
const fs = require('fs');

const TTS_API_URL = 'http://localhost:9999/v1/audio/speech';
const VOICE_ID = 'my_voice';

const client = new Client({
    intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages, GatewayIntentBits.MessageContent]
});

async function generateTTS(text) {
    const response = await axios.post(TTS_API_URL, {
        model: 'luxtts',
        input: text,
        voice: VOICE_ID,
        response_format: 'mp3',
        speed: 1.0
    }, {
        responseType: 'arraybuffer'
    });
    
    return Buffer.from(response.data);
}

client.on('messageCreate', async (message) => {
    if (message.author.bot) return;
    
    if (message.content.startsWith('!speak ')) {
        const text = message.content.slice(7);
        
        if (text.length > 4096) {
            return message.reply('Text too long! Maximum 4096 characters.');
        }
        
        try {
            const audioBuffer = await generateTTS(text);
            const attachment = { attachment: audioBuffer, name: 'speech.mp3' };
            await message.reply({ files: [attachment] });
        } catch (error) {
            console.error(error);
            await message.reply('Error generating speech!');
        }
    }
});

client.login('YOUR_DISCORD_BOT_TOKEN');
```

## Python Client Example

```python
import requests

class LuxTTSClient:
    def __init__(self, base_url="http://localhost:9999"):
        self.base_url = base_url
    
    def upload_voice(self, audio_path, voice_id, duration=5):
        """Upload a voice sample"""
        with open(audio_path, 'rb') as f:
            files = {'audio_file': f}
            data = {
                'voice_id': voice_id,
                'duration': duration,
                'rms': 0.01
            }
            response = requests.post(
                f"{self.base_url}/v1/voices/upload",
                files=files,
                data=data
            )
            return response.json()
    
    def list_voices(self):
        """List uploaded voices"""
        response = requests.get(f"{self.base_url}/v1/voices")
        return response.json()
    
    def generate_speech(self, text, voice_id, output_path, format="mp3", speed=1.0):
        """Generate speech from text"""
        payload = {
            "model": "luxtts",
            "input": text,
            "voice": voice_id,
            "response_format": format,
            "speed": speed
        }
        
        response = requests.post(
            f"{self.base_url}/v1/audio/speech",
            json=payload
        )
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False

# Usage
client = LuxTTSClient()

# Upload voice
client.upload_voice("my_voice_sample.wav", "my_voice")

# Generate speech
client.generate_speech(
    "Hello, this is a test!",
    "my_voice",
    "output.mp3"
)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LUXTTS_DEVICE` | Device (cuda/cpu/mps/auto) | `auto` |
| `LUXTTS_PORT` | Server port | `9999` |
| `LUXTTS_HOST` | Server host | `0.0.0.0` |
| `LUXTTS_VOICE_CACHE` | Voice cache directory | `./voice_cache` |
| `LUXTTS_DEFAULT_VOICE` | Default voice ID | `None` |

## Tips

### Voice Quality

- Use at least 3 seconds of clean audio for voice cloning
- WAV or FLAC files give the best results
- Avoid background noise in voice samples

### Performance

- **GPU (CUDA)**: 150x realtime on modern GPUs
- **CPU**: 10-20x realtime (slower but still usable)
- **MPS (Mac)**: Good performance on Apple Silicon

### Troubleshooting

**"Voice not found" error:**
- Upload a voice first using `/v1/voices/upload`
- Check available voices with `/v1/voices`

**Metallic artifacts:**
- Set `return_smooth: true` in the request
- Increase `num_steps` to 8-16 (slower but cleaner)

**Pronunciation issues:**
- Decrease `t_shift` to 0.7-0.8
- Trade-off: better pronunciation but slightly lower quality

**Out of memory:**
- Use CPU mode: `--device cpu`
- Reduce `num_steps` to 4 or less

## Docker (Optional)

If you have Docker set up:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9999

CMD ["python3", "start_server.py", "--host", "0.0.0.0"]
```

Build and run:
```bash
docker build -t luxtts-api .
docker run -p 9999:9999 --gpus all luxtts-api
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:9999/docs
- **ReDoc**: http://localhost:9999/redoc

## Support

For issues or questions:
- GitHub: https://github.com/ysharma3501/LuxTTS
- Email: yatharthsharma350@gmail.com
