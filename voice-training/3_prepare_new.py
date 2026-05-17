"""
Process only the new Dang Dang Doko Dang files and append to existing dataset.
Starts clip numbering from 3035 to avoid overwriting existing clips.
"""

import os
import sys
import csv
import numpy as np
import noisereduce as nr
import soundfile as sf
import whisper
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm

sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

SOURCE_DIR  = Path(__file__).parent.parent / "Osho Rec"
DATASET_DIR = Path(__file__).parent / "dataset"
WAVS_DIR    = DATASET_DIR / "wavs"
SAMPLE_RATE = 22050
MIN_CLIP_MS = 2000
MAX_CLIP_MS = 12000

NEW_FILES = [f for f in sorted(SOURCE_DIR.glob("Dang Dang Doko Dang*.mp3"))]
START_INDEX = 3035


def segment_audio(mp3_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
    chunks = split_on_silence(
        audio,
        min_silence_len=400,
        silence_thresh=-40,
        keep_silence=150,
    )
    merged, current = [], None
    for chunk in chunks:
        if current is None:
            current = chunk
        elif len(current) + len(chunk) <= MAX_CLIP_MS:
            current += chunk
        else:
            if len(current) >= MIN_CLIP_MS:
                merged.append(current)
            current = chunk if len(chunk) <= MAX_CLIP_MS else None
    if current and len(current) >= MIN_CLIP_MS:
        merged.append(current)
    return merged


def main():
    print(f"Found {len(NEW_FILES)} new files to process:")
    for f in NEW_FILES:
        print(f"  {f.name}")

    print("\nLoading Whisper medium model...")
    model = whisper.load_model("medium", device="cpu")
    print("Whisper loaded.")

    clip_index = START_INDEX
    all_wav_paths = []

    for mp3_path in NEW_FILES:
        print(f"\nProcessing: {mp3_path.name}")
        chunks = segment_audio(mp3_path)
        print(f"  {len(chunks)} segments")

        for chunk in chunks:
            wav_name = f"osho_{clip_index:05d}"
            wav_path = WAVS_DIR / f"{wav_name}.wav"
            chunk.export(str(wav_path), format="wav")
            data, sr = sf.read(str(wav_path))
            if data.ndim > 1:
                data = data[:, 0]
            noise_sample = data[:int(sr * 0.3)]
            denoised = nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample, prop_decrease=0.75)
            sf.write(str(wav_path), denoised, sr)
            all_wav_paths.append(wav_path)
            clip_index += 1

        print(f"  Saved {len(chunks)} clips (up to osho_{clip_index-1:05d})")

    print(f"\nTotal new clips: {len(all_wav_paths)}")
    print("Starting Whisper transcription...")

    metadata_path = DATASET_DIR / "metadata.csv"
    kept = 0
    with open(metadata_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        for i, wav_path in enumerate(tqdm(all_wav_paths)):
            result = model.transcribe(str(wav_path), language="en", fp16=False)
            text = result["text"].strip()
            if len(text.split()) >= 3:
                writer.writerow([wav_path.stem, text, text])
                kept += 1
            else:
                wav_path.unlink()
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(all_wav_paths)} transcribed, kept {kept}")

    print(f"\nDone! Added {kept} new clips to dataset.")
    print(f"Total dataset now at: {metadata_path}")


if __name__ == "__main__":
    main()
