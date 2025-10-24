# AI Music Improver 🎵

Dual-engine AI system that can:
- **Improve symbolic music (MIDI)** using a Transformer model.
- **Enhance real audio (MP3/WAV)** using a super-resolution CNN model.
- Accessible through a clean **web UI**.
- GPU-accelerated and Docker-ready.

---

## Components
| Engine | Description |
|--------|--------------|
| **SMT (Symbolic Music Transformer)** | Improves compositions, timing, melody |
| **ASRNet (Audio Super-Resolution)** | Converts low-quality 64 kbps audio into high-fidelity “4 K” sound |

---

## Quick start with Docker (recommended)

1. Build the Docker image (requires Docker & NVIDIA Container Toolkit for GPU):
```bash
docker build -t ai-music-improver .
```
2. Create uploads folder locally and run the container (GPU):
```bash
mkdir -p server/uploads server/uploads/outputs
docker run --rm -p 8000:8000 -v $(pwd)/server/uploads:/app/server/uploads --gpus all ai-music-improver
```
If you don't have GPUs, omit `--gpus all` (image will run on CPU but much slower):
```bash
docker run --rm -p 8000:8000 -v $(pwd)/server/uploads:/app/server/uploads ai-music-improver
```

Open the app in your browser: `http://localhost:8000`

---

## Local development (no Docker)

1. Create virtualenv and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Start the server:
```bash
cd server
python app.py
```

3. Open `http://localhost:8000` to access the web UI.

---

## How the web UI works
- Upload `.mid`, `.wav`, or `.mp3` files.
- Choose mode: **Music Improve** (symbolic) or **Audio Upscale** (audio super-resolution) or **Auto**.
- After processing, outputs appear under `server/uploads/outputs` and can be downloaded or played from the UI.

---

## Where to place music files for training
- Symbolic (MIDI) training data: put `.mid`/`.midi` files in `server/training_data/midi/`.
- Audio training data (for AudioSR): put high-quality `.wav` files in `server/training_data/audio/high/` and corresponding low-quality versions in `server/training_data/audio/low/` (or the training script will generate low-quality versions by downsampling/encoding).

Example structure:
```
server/training_data/
├─ midi/
│  ├─ file1.mid
│  └─ file2.mid
└─ audio/
   ├─ high/
   │  ├─ song1.wav
   │  └─ song2.wav
   └─ low/   # optional (can be generated)
      ├─ song1_64kbps.mp3
      └─ song2_64kbps.mp3
```

---

## Training tips
- Start with small subsets (10–100 files) and small model sizes to iterate quickly.
- For symbolic model, edit `config.yaml` training params under `training` and run `python server/symbolic_model/train.py --config ../config.yaml`.
- For audio SR model, edit `config.yaml` `audio_sr` settings and run `python server/audio_model/train_audio_sr.py --config ../config.yaml`.
- Use a GPU where possible (`--gpus all` with Docker or run on cloud GPU) for faster training.

---

## File locations
- Server code: `server/`
- Frontend static files (built): `server/static/`
- Uploaded files: `server/uploads/`
- Output files: `server/uploads/outputs/`
- Checkpoints: `server/checkpoints/`

---

If you want, I can also provide a prebuilt small demo checkpoint (tiny weights) and an example MIDI/audio file included in the zip. Ask and I will add them.