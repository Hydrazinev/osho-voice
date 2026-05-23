"""
TTS Inference Service — FastAPI
POST /synthesize  { "text": "..." }  → returns audio/wav stream
GET  /health      → { "status": "ok" }

Uses F5-TTS fine-tuned on Osho's voice.
Drop your trained checkpoint at: ../f5-training/checkpoints/model_last.pt
"""

import io
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ.pop("PYTHONHASHSEED", None)

import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Osho TTS Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CKPT_DIR  = Path(__file__).parent.parent / "f5-training" / "checkpoints"
REFS_DIR  = Path(__file__).parent.parent / "voice-training" / "reference_clips"
REF_AUDIO = REFS_DIR / "ref_01.wav"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

tts_pipe = None


def find_checkpoint() -> Path | None:
    candidates = sorted(CKPT_DIR.glob("**/*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_model():
    global tts_pipe
    from f5_tts.api import F5TTS

    ckpt = find_checkpoint()
    if ckpt:
        print(f"Loading fine-tuned checkpoint: {ckpt}")
        tts_pipe = F5TTS(ckpt_file=str(ckpt), device=device)
    else:
        print("No fine-tuned checkpoint found — loading base F5-TTS model...")
        tts_pipe = F5TTS(device=device)

    print(f"F5-TTS loaded on {device}")


@app.on_event("startup")
async def startup():
    load_model()


class SynthRequest(BaseModel):
    text: str
    speed: float = 1.0


@app.get("/health")
def health():
    ckpt = find_checkpoint()
    return {
        "status": "ok",
        "device": device,
        "model_loaded": tts_pipe is not None,
        "checkpoint": str(ckpt) if ckpt else "base model",
    }


@app.post("/synthesize")
async def synthesize(req: SynthRequest):
    if not tts_pipe:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    if len(req.text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 chars)")

    ref = str(REF_AUDIO) if REF_AUDIO.exists() else None

    try:
        wav, sr, _ = tts_pipe.infer(
            ref_file=ref,
            ref_text="It is the mind that has been trained into Aristotelian logic.",
            gen_text=req.text,
            speed=req.speed,
            nfe_step=16,
        )

        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        wav = np.array(wav, dtype=np.float32)

        buf = io.BytesIO()
        sf.write(buf, wav, samplerate=sr, format="WAV")
        buf.seek(0)

        return StreamingResponse(buf, media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
