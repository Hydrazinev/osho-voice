import os
import io
import modal
import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Speaker registry ─────────────────────────────────────────────────────────
# To add a new speaker:
#   1. Train an F5-TTS model and upload to HuggingFace
#   2. Add an entry here with repo, model file, ref audio filename, and ref text
#   3. Re-deploy: modal deploy tts-service/modal_app.py
SPEAKERS = {
    "osho": {
        "repo": "Hydrazinenv/osho-tts-model",
        "model_file": "model_last.pt",
        "ref_audio": "ref_01.wav",
        "ref_text": "It is the mind that has been trained into Aristotelian logic.",
    },
    "morgan": {
        "repo": "Hydrazinenv/morgan-tts-model",
        "model_file": "model_last.pt",
        "ref_audio": "ref_morgan_short.wav",
        "ref_text": "How did the universe begin? We've all heard of the Big Bang, but how do we really know that's the way it was? I mean, after all, nobody was around to see it happen.",
        "nfe_step": 48,
    },
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["ffmpeg"])
    .pip_install(["f5-tts", "soundfile", "numpy", "huggingface_hub", "fastapi"])
)

volume = modal.Volume.from_name("osho-model-cache", create_if_missing=True)

app = modal.App("osho-tts")


@app.cls(
    gpu="T4",
    image=image,
    volumes={"/models": volume},
    scaledown_window=300,
    timeout=120,
)
class OshoTTS:
    @modal.enter()
    def load(self):
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        import shutil

        # Pre-download all speaker assets into /models/<speaker>/
        for speaker, cfg in SPEAKERS.items():
            speaker_dir = Path(f"/models/{speaker}")
            speaker_dir.mkdir(parents=True, exist_ok=True)

            ckpt_path = speaker_dir / cfg["model_file"]
            ref_path = speaker_dir / cfg["ref_audio"]

            if not ckpt_path.exists():
                print(f"Downloading {speaker} checkpoint from HF Hub...")
                src = hf_hub_download(cfg["repo"], cfg["model_file"])
                shutil.copy(src, ckpt_path)

            if not ref_path.exists():
                print(f"Downloading {speaker} reference audio from HF Hub...")
                src = hf_hub_download(cfg["repo"], cfg["ref_audio"])
                shutil.copy(src, ref_path)

        volume.commit()

        # Lazy model cache — models load on first request to conserve VRAM
        self._models: dict = {}
        print("Speaker assets ready. Models will load on first request.")

    def _get_model(self, speaker: str):
        if speaker not in self._models:
            from pathlib import Path
            from f5_tts.api import F5TTS

            cfg = SPEAKERS[speaker]
            ckpt_path = str(Path(f"/models/{speaker}") / cfg["model_file"])
            print(f"Loading {speaker} model into GPU...")
            self._models[speaker] = F5TTS(ckpt_file=ckpt_path, device="cuda")
            print(f"{speaker} model ready.")
        return self._models[speaker]

    @modal.asgi_app()
    def web(self):
        fastapi_app = FastAPI(title="Voice TTS")
        fastapi_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        get_model = self._get_model

        class SynthRequest(BaseModel):
            text: str
            speed: float = 1.0
            speaker: str = "osho"

        @fastapi_app.get("/health")
        def health():
            return {"status": "ok", "speakers": list(SPEAKERS.keys())}

        @fastapi_app.post("/synthesize")
        async def synthesize(req: SynthRequest):
            if not req.text.strip():
                raise HTTPException(400, "Text is empty")
            if len(req.text) > 5000:
                raise HTTPException(400, "Text too long (max 5000 chars)")
            if req.speaker not in SPEAKERS:
                raise HTTPException(400, f"Unknown speaker '{req.speaker}'. Valid: {list(SPEAKERS.keys())}")

            cfg = SPEAKERS[req.speaker]
            tts = get_model(req.speaker)
            ref_file = str(f"/models/{req.speaker}/{cfg['ref_audio']}")
            ref_text = cfg["ref_text"]

            wav, sr, _ = tts.infer(
                ref_file=ref_file,
                ref_text=ref_text,
                gen_text=req.text,
                speed=req.speed,
                nfe_step=cfg.get("nfe_step", 32),
            )
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            wav = np.array(wav, dtype=np.float32)
            buf = io.BytesIO()
            sf.write(buf, wav, samplerate=sr, format="WAV")
            buf.seek(0)
            return StreamingResponse(buf, media_type="audio/wav")

        return fastapi_app
