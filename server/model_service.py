import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from torchaudio.transforms import Resample

# ===============================================================
# High-quality audio loader (fix for 16000 vs 32000 mismatch)
# ===============================================================
def load_audio_high_quality(file_path, target_sr=32000, max_len=5.0):
    """Load, resample, normalize, and pad/trim audio safely (preserving quality)."""
    waveform, sr = torchaudio.load(file_path)

    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample only if needed
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # Normalize to prevent clipping
    waveform = waveform / waveform.abs().max().clamp_min(1e-6)

    # Pad or trim to fixed max length for consistency
    max_samples = int(target_sr * max_len)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    else:
        pad_len = max_samples - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad_len))

    return waveform, target_sr


# ===============================================================
# Simple Conv1D Autoencoder for audio enhancement
# ===============================================================
class AudioAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# ===============================================================
# TRAINING FUNCTION (uses high-quality loader)
# ===============================================================
def train_model(data_dir="/app/server/uploads", mode="music", epochs=5, lr=1e-4, target_sr=32000):
    print(f"[TRAIN] Loading audio data from {data_dir}")
    data_dir = Path(data_dir)
    audio_files = list(data_dir.glob("*.wav")) + list(data_dir.glob("*.mp3"))
    if not audio_files:
        raise RuntimeError("No audio files found for training")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TRAIN] Using device: {device}")

    model = AudioAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Prepare training data
    data = []
    for f in audio_files:
        waveform, sr = load_audio_high_quality(f, target_sr)
        data.append(waveform)
    dataset = torch.cat(data, dim=1)  # concatenate along time
    n_segments = dataset.shape[1] // (target_sr * 2)  # 2 sec segments
    print(f"[TRAIN] {len(audio_files)} files, {n_segments} segments")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(n_segments):
            start = i * target_sr * 2
            end = start + target_sr * 2
            x = dataset[:, start:end].unsqueeze(0).to(device)
            # add synthetic noise to learn denoising
            noisy = x + 0.05 * torch.randn_like(x)
            out = model(noisy)

            # --- 🔧 Fix: Ensure output matches input length ---
            if out.shape[2] > x.shape[2]:
                out = out[:, :, :x.shape[2]]
            elif out.shape[2] < x.shape[2]:
                pad_len = x.shape[2] - out.shape[2]
                out = F.pad(out, (0, pad_len))
            # --------------------------------------------------

            loss = loss_fn(out, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(n_segments, 1)
        print(f"[TRAIN] Epoch {epoch+1}/{epochs}, loss={avg_loss:.6f}")

    # Save trained model
    ckpt_path = Path(__file__).parent / "checkpoints" / "trained_model.pt"
    os.makedirs(ckpt_path.parent, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"[TRAIN] Saved model to {ckpt_path}")


# ===============================================================
# INFERENCE FUNCTION (uses high-quality loader)
# ===============================================================
def improve_music_or_audio(input_path, output_path, mode="auto", target_sr=32000):
    print(f"[IMPROVE] Enhancing {input_path} -> {output_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AudioAutoencoder().to(device)
    ckpt_path = Path(__file__).parent / "checkpoints" / "trained_model.pt"
    if not ckpt_path.exists():
        raise RuntimeError("No trained model found. Please run training first.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    waveform, sr = load_audio_high_quality(input_path, target_sr)
    with torch.no_grad():
        enhanced = model(waveform.unsqueeze(0).to(device))
    enhanced = enhanced.squeeze(0).cpu()

    # --- 🔧 Fix: Align output length to input length ---
    if enhanced.shape[1] > waveform.shape[1]:
        enhanced = enhanced[:, :waveform.shape[1]]
    elif enhanced.shape[1] < waveform.shape[1]:
        pad_len = waveform.shape[1] - enhanced.shape[1]
        enhanced = F.pad(enhanced, (0, pad_len))
    # ----------------------------------------------------

    torchaudio.save(output_path, enhanced, target_sr)
    print(f"[IMPROVE] Saved enhanced audio to {output_path}")

