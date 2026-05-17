# Osho Voice Clone

An end-to-end AI voice cloning pipeline that fine-tunes [F5-TTS](https://github.com/SWivid/F5-TTS) on ~19 hours of Osho's speeches and serves real-time synthesis through a Next.js web app.

**Live demo:** _coming soon_

---

## What it does

Paste any text — a book chapter, an article, anything — and hear it read aloud in Osho's voice. The site streams audio chunk by chunk so playback starts immediately while the next segment is being synthesized.

---

## Architecture

```
Raw MP3s (19h)
    │
    ▼
voice-training/1_prepare_audio.py
    • Silence-split into 2–12s clips  (pydub)
    • Noise reduction                  (noisereduce)
    • Transcription                    (Whisper medium)
    │  6,676 labelled clips
    ▼
f5-training/prepare_dataset.py
    • Converts to F5-TTS format
    • Mel spectrogram extraction + duration.json
    │
    ▼
f5-training/train.py
    • Fine-tunes F5TTS_v1_Base on Google Colab A100
    • 20 epochs, lr=5e-6, batch=8
    │  5.4 GB checkpoint → Hugging Face Hub
    ▼
tts-service/modal_app.py  (FastAPI on Modal T4 GPU)
    • Loads checkpoint from HF Hub into Modal Volume
    • POST /synthesize  →  audio/wav stream
    │
    ▼
web/  (Next.js + Tailwind, deployed on Vercel)
    • Chunks text at sentence boundaries
    • Prefetches next chunk while current one plays
    • Pause / resume / stop controls
```

---

## Tech stack

| Layer | Tech |
|---|---|
| Voice model | F5-TTS (flow-matching TTS) |
| Fine-tuning | PyTorch · Accelerate · Google Colab A100 |
| Audio prep | Whisper · pydub · noisereduce |
| Inference API | FastAPI · Modal (serverless GPU) |
| Model storage | Hugging Face Hub |
| Frontend | Next.js 15 · Tailwind CSS · TypeScript |
| Deployment | Vercel (frontend) · Modal (backend) |

---

## Repo structure

```
├── voice-training/
│   ├── 1_prepare_audio.py   — segment, denoise, transcribe MP3s
│   └── 3_prepare_new.py     — append new recordings to dataset
│
├── f5-training/
│   ├── prepare_dataset.py   — convert to F5-TTS format
│   └── train.py             — fine-tuning script
│
├── tts-service/
│   ├── modal_app.py         — serverless GPU inference (Modal)
│   └── main.py              — local FastAPI server
│
└── web/                     — Next.js frontend
    └── app/page.tsx
```

---

## Run locally

**Backend**
```bash
cd tts-service
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py          # runs at http://localhost:8000
```

**Frontend**
```bash
cd web
npm install
echo "NEXT_PUBLIC_TTS_URL=http://localhost:8000" > .env.local
npm run dev             # runs at http://localhost:3000
```

---

## Key design decisions

- **F5-TTS over XTTS**: flow-matching model produces more natural prosody with less data
- **Serverless GPU (Modal)**: zero idle cost — GPU spins up per request, model cached in a Volume
- **Streaming playback**: text is chunked at sentence boundaries; audio plays as soon as the first chunk is ready instead of waiting for the full text
- **Whisper medium for transcription**: ran on Colab A100 (~1.4s/clip) to build a clean labelled dataset from raw MP3s
